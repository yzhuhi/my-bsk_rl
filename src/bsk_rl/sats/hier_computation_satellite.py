# -*- coding: utf-8 -*-
"""
HierComputationSatellite: 分层连续路由方案实现。

该类继承自 ComputationSatellite，提供分层 RL 专用的动作空间和观测空间配置。
自有路由方法 ``process_slice_queue_hier`` 和 ``schedule_collaboration_action_hier``
均定义在本子类中，支持分层任务切分与切片路由。

设计说明：
    - Level-1（任务切分）：对 raw_task_queue 执行，生成切片加入 slice_queue
    - Level-2（切片路由）：对 slice_queue 执行概率采样路由
    - 支持两种执行语义：
      * delayed: Level-1 新切片下一个 step 再参与 Level-2
      * immediate: Level-1 新切片本 step 即可参与 Level-2

动作空间（与旧版差异）：
    - 旧版 STINContinuousAction: 16D
    - 新版 STINHierarchicalHybridAction: K + 7 (K=4 时 11D，L2 含 self)
    - Level-1 任务切分: 4D（UD/Cloud/本地/路由）

观测空间（与旧版差异）：
    - 旧版 ComputationSatellite.observation_spec: ~40D (SatProperties + STINRelativeObservations)
    - 新版 STINHierarchicalObservations: (F+1)K (F=8, K=4 时 36D)
    - 使用统一扁平向量，支持可配置属性
    - 新增邻居可达性掩码 (K维)

队列流程::

    raw_task_queue → [Level-1 切分] → slice_queue → [Level-2 路由判断]
                                                          ├→ 本地: task_queue → [计算] → result_queue
                                                          └→ 邻居: pop 并转发到邻居的 slice_queue

主要方法:
    - ``schedule_collaboration_action_hier``: Level-1 任务切分
    - ``process_slice_queue_hier``: Level-2 切片路由（概率采样/argmax）
    - ``_mask_and_normalize_routing_probs``: 路由概率 mask + 归一化

注意: 本类方法不在父类 ComputationSatellite 中定义，仅供 HierComputationSatellite 使用。
"""

import copy
import logging
from typing import TYPE_CHECKING, Any, List, Optional

import numpy as np
from bsk_rl.sim import dyn, fsw
from bsk_rl import obs
from bsk_rl.sats.computation_satellite import ComputationSatellite, TaskSlice, TaskStatus
from Basilisk.utilities.orbitalMotion import REQ_EARTH

from bsk_rl.act.stin_hier_hybrid_actions import STINHierarchicalHybridAction
from bsk_rl.utils.constants import calculate_isl_rate

if TYPE_CHECKING:  # pragma: no cover
    from bsk_rl.act.actions import Action

logger = logging.getLogger(__name__)


class HierComputationSatellite(ComputationSatellite):
    """分层 RL 专用卫星实现。

    继承 ComputationSatellite，使用 STINHierarchicalHybridAction 作为动作空间。
    观测空间采用 SatProperties + STINHierarchicalObservations + Time/Eclipse。
    
    核心方法（本类定义）：
        - ``schedule_collaboration_action_hier``: Level-1 任务切分
        - ``process_slice_queue_hier``: Level-2 切片路由（概率采样/argmax）
        - ``_mask_and_normalize_routing_probs``: 路由概率 mask + 归一化
    
    设计意图：
        通过 routing_execution_mode 在 delayed / immediate 语义间切换，
        兼顾兼容性（旧实验）与即时路由可学习性（新实验）。
    
    动作空间 (11D, K=4):
        - Level-1 (4D): [alpha_ud, alpha_cloud, alpha_sat_local, alpha_sat_offload]
        - Level-2 (7D): [cpu, tx, route_n0...route_nK-1, route_self]
    
    观测空间 (与父类相同风格，约 40D):
        - SatProperties: 卫星自身状态（电池、队列、CPU 等）
        - STINRelativeObservations: 邻居状态
        - Time + Eclipse
    
    Attributes:
        action_spec: 动作空间规范列表 (包含 STINHierarchicalHybridAction)
        observation_spec: 观测空间规范列表 (与父类风格一致)
        max_neighbors: 最大邻居数量，决定动作空间维度
    
    Args:
        max_neighbors: 最大邻居数量，决定动作空间维度 (默认 4)
        *args: 传递给父类的位置参数
        **kwargs: 传递给父类的关键字参数
    """
    # --- 类属性: 模型类型定义 ---
    dyn_type = dyn.LoSComputationDynaModel
    fsw_type = fsw.SteeringImagerFSWModel
    
    action_spec: List["Action"] = [STINHierarchicalHybridAction(max_neighbors=4)]
    # --- 类属性: 观测空间定义 (使用分层专用观测类) ---
    # 自身状态 (8D): 由 SatProperties 处理 (包含位置速度向量) (CPU, Power, Queue, Workload)
    # 队列统计 (8D): 由 SatProperties 处理 (详细队列统计)
    # 邻居状态 (36D): 由 STINHierarchicalObservations 处理 (8特征 x 4邻居 + 4Mask)
    # 环境状态 (2D): Time + Eclipse
    # 默认总维度: 8 + 10 + 36 + 2 = 56D
    observation_spec = [
        # 1. 自身状态 (SatProperties - 支持 module)
        obs.SatProperties(
            dict(prop="battery_charge_fraction", module="dynamics"),
            dict(prop="storage_level_fraction", module="dynamics"),
            dict(prop="r_BN_P", module="dynamics", norm=REQ_EARTH * 1e3),
            dict(prop="v_BN_P", module="dynamics", norm=7616.5),
            name="sat_state" 
        ),
        # 2. 资源与队列状态 (SatProperties - 使用 lambda 或属性)
        obs.SatProperties(
            dict(prop="current_cpu_freq", fn=lambda sat: sat.current_cpu_freq / 2e9),
            dict(prop="current_tx_power", fn=lambda sat: sat.current_tx_power / 30.0), # 新增
            dict(prop="task_queue_size", fn=lambda sat: min(len(sat.task_queue), 50) / 50.0),
            dict(prop="queue_workload", fn=lambda sat: min(sum(t.remaining_workload for t in sat.task_queue) / 1e9, 100.0) / 100.0), # 任务队列负载 [0, 1] ✅ Fix1: 归一化+截断
            name="resource_state"
        ),
        # 3. 队列统计 (SatProperties)
        obs.SatProperties(
            dict(prop="avg_task_data_size", fn=lambda sat: min(float(np.nanmean([t.data_size for t in sat.task_queue])) / 1e6, 500.0) / 500.0 if sat.task_queue else 0.0),  # [Mb] 平均数据量 [0, 1]
            dict(prop="avg_task_workload", fn=lambda sat: min(float(np.nanmean([t.workload for t in sat.task_queue])) / 1e3, 10.0) / 10.0 if sat.task_queue else 0.0),     # [kcycles/bit] 平均复杂度 [0, 1]
            dict(prop="total_queue_data", fn=lambda sat: min(sum(t.data_size for t in sat.task_queue) / 1e9, 50.0) / 50.0 if sat.task_queue else 0.0),
            dict(prop="avg_priority", fn=lambda sat: float(np.nanmean([getattr(t, 'priority', 0.5) for t in sat.task_queue])) if sat.task_queue else 0.5),
            dict(prop="min_remaining_time", fn=lambda sat: min(float(np.nanmin([t.get_remaining_time(sat.simulator.sim_time) for t in sat.task_queue])), 300.0) / 300.0 if sat.task_queue else 1.0),
            dict(prop="raw_queue_size", fn=lambda sat: min(len(sat.raw_task_queue), 50) / 50.0),  # 原始任务数量 [0, 1]
            dict(prop="raw_queue_data", fn=lambda sat: min(sum(t.data_size for t in sat.raw_task_queue) / 1e9, 50.0) / 50.0 if sat.raw_task_queue else 0.0),  # [Gb] 原始任务总数据量 [0, 1]
            dict(prop="raw_avg_priority", fn=lambda sat: float(np.nanmean([getattr(t, 'priority', 0.5) for t in sat.raw_task_queue])) if sat.raw_task_queue else 0.5),  # [0, 1] 平均优先级
            dict(prop="slice_queue_size", fn=lambda sat: min(len(getattr(sat, "slice_queue", [])), 50) / 50.0),
            dict(prop="slice_queue_data", fn=lambda sat: min(sum(t.data_size for t in getattr(sat, "slice_queue", [])) / 1e9, 50.0) / 50.0 if getattr(sat, "slice_queue", []) else 0.0),
            name="queue_stats"
        ),
        # 4. 邻居状态 (STINHierarchicalObservations)
        obs.STINHierarchicalObservations(
            dict(prop="get_task_queue_size", norm=50.0, name="task_queue_size"),      # 队列长度，归一化到 50
            dict(prop="get_avg_task_priority", norm=1.0, name="avg_task_priority"),   # 平均优先级 [0, 1] （独有）
            dict(prop="get_battery_fraction", norm=1.0, name="battery_fraction"),     # 电池电量 [0, 1]
            dict(prop="get_isl_distance", norm=1e7, name="isl_distance"),             # ISL 距离 [m]，归一化到 10,000 km
            dict(prop="get_isl_channel_quality", norm=1.0, name="isl_channel_quality"), # 信道质量 [0, 1]
            dict(prop="get_cpu_freq_ratio", norm=1.0, name="cpu_freq_ratio"),         # CPU 频率比例 [0, 1]
            dict(prop="get_queue_workload", norm=1e10, name="queue_workload"),        # 队列工作量 [cycles]
            dict(prop="get_downstream_queue", norm=50.0, name="downstream_queue"),    # 🆕 下游拥塞信号 (Backpressure)
            max_neighbors=4,
        ),
        # 5. 环境
        obs.Time(),
        obs.Eclipse(norm=5700.0),
    ]
    
    def __init__(self, *args, **kwargs) -> None:
        # 提取 max_neighbors 参数（动作空间维度依赖此值）
        max_neighbors = kwargs.pop('max_neighbors', 4)
        max_route_hops = kwargs.pop('max_route_hops', None)
        self.routing_mode = kwargs.pop('routing_mode', 'sample')
        self.routing_execution_mode = str(
            kwargs.pop("routing_execution_mode", "delayed")
        ).lower()
        level1_norm_mode = kwargs.pop('level1_norm_mode', 'ratio_normalize')

        # 在父类初始化前设置 spec，确保 Builder 使用正确维度
        self.action_spec = [
            STINHierarchicalHybridAction(
                max_neighbors=max_neighbors,
                level1_norm_mode=level1_norm_mode,
                routing_execution_mode=self.routing_execution_mode,
            )
        ]

        # 观测空间动态调整 (仅需调整 STINHierarchicalObservations 的 max_neighbors)
        if max_neighbors != 4:
            # 原始邻居属性配置
            neighbor_props = [
                dict(prop="get_task_queue_size", norm=50.0, name="task_queue_size"),
                dict(prop="get_avg_task_priority", norm=1.0, name="avg_task_priority"),
                dict(prop="get_battery_fraction", norm=1.0, name="battery_fraction"),
                dict(prop="get_isl_distance", norm=1e7, name="isl_distance"),
                dict(prop="get_isl_channel_quality", norm=1.0, name="isl_channel_quality"),
                dict(prop="get_cpu_freq_ratio", norm=1.0, name="cpu_freq_ratio"),
                dict(prop="get_queue_workload", norm=1e10, name="queue_workload"),
                dict(prop="get_downstream_queue", norm=50.0, name="downstream_queue"),  # 🆕 下游拥塞
            ]
            # 复制类默认配置并替换邻居观测部分
            new_obs_spec = list(self.observation_spec)
            # 找到 STINHierarchicalObservations 实例并替换
            replaced = False
            for i, obs_item in enumerate(new_obs_spec):
                if isinstance(obs_item, obs.STINHierarchicalObservations):
                    new_obs_spec[i] = obs.STINHierarchicalObservations(
                        *neighbor_props,
                        max_neighbors=max_neighbors,
                    )
                    replaced = True
                    break
            if not replaced:
                raise ValueError(
                    f"max_neighbors={max_neighbors} requires STINHierarchicalObservations in "
                    "observation_spec, but none was found. Check observation_spec definition "
                    "or set max_neighbors=4."
                )
            self.observation_spec = new_obs_spec

        super().__init__(*args, **kwargs)
        # 一致性检查：动作/观测的 max_neighbors 必须一致
        for act in getattr(self, "action_spec", []):
            if hasattr(act, "max_neighbors") and act.max_neighbors != max_neighbors:
                raise ValueError(
                    f"Action max_neighbors mismatch: action={act.max_neighbors}, "
                    f"config={max_neighbors}."
                )
        for obs_item in getattr(self, "observation_spec", []):
            if isinstance(obs_item, obs.STINHierarchicalObservations):
                if obs_item.max_neighbors != max_neighbors:
                    raise ValueError(
                        f"Observation max_neighbors mismatch: obs={obs_item.max_neighbors}, "
                        f"config={max_neighbors}."
                    )
        if max_route_hops is None:
            max_route_hops = getattr(self, "sat_args", {}).get("maxRouteHops", 3)
        self.max_route_hops = max_route_hops
        # 限制队列诊断日志次数，避免刷屏
        self._queue_diag_remaining = 200
        # 路由后处理差异诊断（base_ratios vs masked probs）
        self._routing_diff_log_remaining = 300
        self._routing_diff_log_threshold = 0.15

        # 队列级诊断（每步由 gym._step 重置）
        self.new_slice_count_this_step = 0
        self.new_slice_data_this_step = 0.0
        self.level2_processed_count_this_step = 0
        self.level2_candidate_slices_this_step = 0

        self.logger.info(
            f"HierComputationSatellite init finish "
            f"(max_neighbors={max_neighbors}, max_route_hops={self.max_route_hops}, "
            f"action_dim={max_neighbors + 7}, routing_execution_mode={self.routing_execution_mode})"
        )

    # ======================== 分层连续路由动作空间专用方法 ========================

    def schedule_collaboration_action_hier(
        self,
        task_split_ratios: np.ndarray,
        neighbor_satellites: List["ComputationSatellite"],
    ) -> None:
        """Level-1 任务切分（分层 RL 专用）。

        对 raw_task_queue 中的每个任务，按 task_split_ratios 比例切分：
        - [0] alpha_ud: UD 本地处理
        - [1] alpha_cloud: 云端处理
        - [2] alpha_sat_local: 接入卫星本地计算队列
        - [3] alpha_sat_offload: 接入卫星切片队列（Level-2 决定路由邻居）

        Args:
            task_split_ratios: 切分比例 [alpha_ud, alpha_cloud, alpha_sat_local, alpha_sat_offload]
                               已归一化，sum = 1
            neighbor_satellites: 邻居卫星列表（长度 K）
        """
        max_tasks_per_step = 40
        tasks_processed = 0

        task_split_ratios = np.asarray(task_split_ratios, dtype=np.float64)
        if task_split_ratios.size < 4:
            padded = np.zeros(4, dtype=np.float64)
            padded[: task_split_ratios.size] = task_split_ratios
            task_split_ratios = padded

        task_split_ratios = np.clip(task_split_ratios, 0.0, 1.0)
        total = np.sum(task_split_ratios)
        if total > 1e-6:
            task_split_ratios = task_split_ratios / total
        else:
            task_split_ratios = np.zeros_like(task_split_ratios)
            task_split_ratios[2] = 1.0  # alpha_sat_local = self

        alpha_ud = float(task_split_ratios[0])
        alpha_cloud = float(task_split_ratios[1])
        alpha_sat_local = float(task_split_ratios[2])
        alpha_sat_offload = float(task_split_ratios[3])
        alpha_sat_total = alpha_sat_local + alpha_sat_offload

        if self.raw_task_queue:
            sat_split_ratios = np.array(
                [alpha_sat_local, alpha_sat_offload], dtype=np.float64
            )
            self.logger.warning (
                f"[COLLAB] α_local={alpha_ud:.2%}, α_cloud={alpha_cloud:.2%}, "
                f"α_sat={alpha_sat_total:.2%} | split_ratios={[f'{r:.2f}' for r in sat_split_ratios]}"
            )

        while self.raw_task_queue and tasks_processed < max_tasks_per_step:
            current_raw_task = self.raw_task_queue[0]
            tasks_processed += 1

            raw_data_size = current_raw_task.data_size
            task_workload = current_raw_task.workload
            task_id = current_raw_task.task_id

            if task_id not in self.slice_registry:
                self.slice_registry[task_id] = {
                    "total": 0,
                    "completed": 0,
                    "expired": 0,
                    "creation_time": current_raw_task.creation_time,
                    "last_completion_time": 0.0,
                    "t_ud_path": 0.0,
                    "t_cloud_path": 0.0,
                }

            # 1) UD 本地处理部分
            if alpha_ud > 1e-6:
                data_ud = raw_data_size * alpha_ud
                allow_ud_complete = alpha_sat_total <= 1e-6
                self._handle_ud_processing(
                    current_raw_task, data_ud, task_workload, allow_complete=allow_ud_complete
                )

            # 2) 云端处理部分
            if alpha_cloud > 1e-6:
                data_cloud = raw_data_size * alpha_cloud
                self._handle_cloud_processing(current_raw_task, data_cloud)

            # 3) 接入卫星本地处理部分 -> 直接入计算队列
            if alpha_sat_local > 1e-6:
                data_self = raw_data_size * alpha_sat_local
                self._create_slice_for_target(
                    current_raw_task, data_self, task_workload, self, local_to_task_queue=True
                )

            # 4) 接入卫星切片队列（待路由到邻居）
            if alpha_sat_offload > 1e-6:
                data_offload = raw_data_size * alpha_sat_offload
                self._create_slice_for_target(
                    current_raw_task, data_offload, task_workload, self, local_to_task_queue=False
                )

            self.raw_task_queue.pop(0)

        if tasks_processed > 0:
            self.requires_retasking = True

    def _handle_ud_processing(
        self, task, data_size: float, workload: float, allow_complete: bool = False
    ) -> None:
        """处理 UD 本地计算部分（即时完成判定）。"""
        ud_cpu_ratio = self.ud_config.get("ud_cpu_ratio", 0.5)
        ud_cpu_freq = self.dynamics.cpu_min_frequency * ud_cpu_ratio
        ud_process_time = (data_size * workload) / ud_cpu_freq
        remaining_time = task.get_remaining_time(self.simulator.sim_time)

        task_id = task.task_id
        self.slice_registry[task_id]["t_ud_path"] = ud_process_time

        if ud_process_time <= remaining_time:
            if allow_complete:
                if task_id not in self.completed_task_ids:
                    self.completed_task_ids.add(task_id)
                    ud_slice = TaskSlice(
                        task_id=task_id,
                        data_size=data_size,
                        workload=workload,
                        max_delay=task.max_delay,
                        origin_position=task.origin_position,
                        uplink_distance=task.uplink_distance,
                        uplink_rate=task.uplink_rate,
                    )
                    ud_slice.status = TaskStatus.COMPLETED
                    ud_slice.priority = getattr(task, "priority", 1.0)
                    ud_slice.t_ud_path = ud_process_time
                    ud_slice.creation_time = task.creation_time
                    ud_slice.compute_end_time = self.simulator.sim_time
                    ud_slice.access_satellite = self.name
                    ud_slice.hop_count = 0
                    self.completed_tasks_buffer.append(ud_slice)
        else:
            if task_id not in self.expired_task_ids:
                self.expired_tasks_count += 1
                self.expired_task_ids.add(task_id)
                expired_slice = TaskSlice(
                    task_id=task_id,
                    data_size=data_size,
                    workload=workload,
                    max_delay=task.max_delay,
                    origin_position=task.origin_position,
                    uplink_distance=task.uplink_distance,
                    uplink_rate=task.uplink_rate,
                )
                expired_slice.status = TaskStatus.EXPIRED
                expired_slice.priority = getattr(task, "priority", 1.0)
                expired_slice.origin_satellite = self.name
                self.expired_tasks_buffer.append(expired_slice)
                self._log_task_expired_once(
                    expired_slice.task_id, reason="ud_timeout"
                )

    def _handle_cloud_processing(self, task, data_size: float) -> None:
        """处理云端部分（记录时延，实际由云端完成）。"""
        t_cloud_path = task.calculate_cloud_path_delay(
            sgl_distance=self._get_downlink_distance(task, "GS"),
            tx_power=self.current_tx_power,
        )
        task_id = task.task_id
        self.slice_registry[task_id]["t_cloud_path"] = t_cloud_path

    def _create_slice_for_target(
        self,
        task,
        data_size: float,
        workload: float,
        target: "ComputationSatellite",
        local_to_task_queue: bool = False,
    ) -> None:
        """为目标卫星创建切片并发送/入队。"""
        isl_tx_delay = 0.0
        isl_prop_delay = 0.0
        if target != self:
            isl_distance = self._compute_isl_distance(target)
            isl_rate = calculate_isl_rate(
                self.current_tx_power,
                isl_distance,
                max_distance=self._get_isl_max_distance_m(),
            )
            isl_tx_delay = data_size / isl_rate
            isl_prop_delay = isl_distance / TaskSlice.SPEED_OF_LIGHT

        new_slice = TaskSlice(
            task_id=task.task_id,
            data_size=data_size,
            workload=workload,
            max_delay=task.max_delay,
            origin_position=task.origin_position,
            uplink_distance=task.uplink_distance,
            uplink_rate=task.uplink_rate,
            priority=getattr(task, "priority", 0.5),
        )

        base_tx_isl_total = getattr(task, "t_tx_isl_total", 0.0)
        base_prop_isl_total = getattr(task, "t_prop_isl_total", 0.0)
        base_hop_count = getattr(task, "hop_count", 0)
        base_route_hop = getattr(task, "current_hop", 0)
        new_slice.t_tx_isl_total = base_tx_isl_total
        new_slice.t_prop_isl_total = base_prop_isl_total
        new_slice.hop_count = base_hop_count
        new_slice.current_hop = base_route_hop
        new_slice.creation_time = task.creation_time
        arrival_time = getattr(task, "arrival_time", 0.0)
        if arrival_time <= 0.0:
            arrival_time = self.simulator.sim_time
        new_slice.arrival_time = arrival_time
        new_slice.fiber_distance = getattr(task, "fiber_distance", 0.0)
        new_slice.cloud_cpu_freq = getattr(task, "cloud_cpu_freq", 10e9)
        new_slice.ud_cpu_freq = getattr(task, "ud_cpu_freq", 1e9)
        new_slice.is_slice = True
        new_slice.isl_tx_power = self.current_tx_power
        new_slice.origin_satellite = self.name
        new_slice.access_satellite = getattr(task, "access_satellite", "") or self.name
        visited = set(getattr(task, "visited_sats", []))
        visited.add(self.name)
        new_slice.visited_sats = visited

        if not new_slice.offload_chain:
            new_slice.offload_chain = [self.name]

        if data_size > 0:
            self.new_slice_count_this_step += 1
            self.new_slice_data_this_step += float(data_size)

        # 🆕 辅助函数：检查本地存储容量
        def _check_local_storage(slice_to_add: TaskSlice) -> bool:
            """检查是否有足够的本地存储空间。返回 True 表示可以存储。"""
            current_total_data = (
                sum(t.data_size for t in self.raw_task_queue) +
                sum(t.data_size for t in self.slice_queue)
            )
            storage_capacity = getattr(self.dynamics, 'data_storage_capacity', float('inf'))
            return current_total_data + slice_to_add.data_size <= storage_capacity

        def _mark_as_expired(slice_to_expire: TaskSlice) -> None:
            """将切片标记为过期并加入过期缓冲区。"""
            slice_to_expire.status = TaskStatus.EXPIRED
            if slice_to_expire.task_id not in self.expired_task_ids:
                self.expired_tasks_count += 1
                self.expired_task_ids.add(slice_to_expire.task_id)
            self.expired_tasks_buffer.append(slice_to_expire)
            self._log_task_expired_once(
                slice_to_expire.task_id, reason="storage_full"
            )
            self.logger.debug(
                f"[STORAGE FULL] Dropping slice {slice_to_expire.task_id}: "
                f"{slice_to_expire.data_size/1e6:.2f} Mb"
            )

        if target == self:
            # 🆕 本地分配也需要检查存储容量
            if _check_local_storage(new_slice):
                if local_to_task_queue:
                    self.task_queue.append(new_slice)
                else:
                    self.slice_queue.append(new_slice)
            else:
                _mark_as_expired(new_slice)
            self.register_slice(task.task_id, 1, creation_time=new_slice.creation_time)
        else:
            new_slice.hop_count = base_hop_count + 1
            new_slice.current_hop = base_route_hop + 1
            accepted = target.process_incoming_slice(new_slice)
            if not accepted:
                new_slice.hop_count = base_hop_count
                new_slice.current_hop = base_route_hop
                # 🆕 邻居拒绝后退回也需要检查本地存储
                if _check_local_storage(new_slice):
                    self.slice_queue.append(new_slice)
                else:
                    _mark_as_expired(new_slice)
            else:
                new_slice.t_tx_isl_total = base_tx_isl_total + isl_tx_delay
                new_slice.t_prop_isl_total = base_prop_isl_total + isl_prop_delay
                self.offloaded_data_total += data_size
                self.first_hop_offloaded += data_size
                self.offloaded_task_ids.add(task.task_id)
            self.register_slice(task.task_id, 1, creation_time=new_slice.creation_time)

    def process_incoming_slice(self, task_slice: "TaskSlice") -> bool:
        """接收切片后进入切片队列，允许继续路由决策。"""
        return super().process_incoming_slice(task_slice)

    def process_slice_queue_hier(
        self,
        high_split_ratios: np.ndarray,
        low_split_ratios: np.ndarray,
        priority_threshold: float,
        neighbor_satellites: List["ComputationSatellite"],
        max_slices: int = None,
        use_argmax: bool = False,
        # epsilon: float = 0.9,  # 添加参数
    ) -> int:
        """Level-2 切片路由（分层连续路由版）。

        路由逻辑：
        - 根据切片优先级选择 high/low 路由分布
        - mask 不可达邻居后归一化
        - 路由模式：
            * sample: 对每个切片按分布采样
            * sample_batch: 对当前 step 的切片做一次批量分配（低方差、近似确定性）
            * flow: 确定性按比例分配（加权轮询）
            * argmax: 始终选择最大概率邻居

        Args:
            high_split_ratios: 高优先级切片路由比例 [n0, n1, ..., nK-1, self]
            low_split_ratios: 低优先级切片路由比例 [n0, n1, ..., nK-1, self]
            priority_threshold: 高/低优先级分界阈值 [0, 1]
            neighbor_satellites: 邻居卫星列表
            max_slices: 最多处理的切片数量（轮询预算）
            use_argmax: 是否使用 argmax 路由（True=确定性，False=按 routing_mode）

        Returns:
            本步实际处理的切片数（包括本地处理与转发）。
        """
        max_forwards_per_step = 60
        forwards_count = 0
        processed_count = 0
        # Reset per-step ISL forwarding budget cache.
        self._isl_step_budget = {}
        mode = "argmax" if use_argmax else self.routing_mode
        flow_probs_high = None
        flow_probs_low = None
        flow_accum_high = None
        flow_accum_low = None
        batch_high = None
        batch_low = None
        batch_high_idx = 0
        batch_low_idx = 0
        high_batch_count = 0
        low_batch_count = 0

        def _build_batch_samples(probs: np.ndarray, count: int) -> List[int]:
            """批量低方差分配：用加权轮询近似期望比例（确定性）。"""
            if count <= 0:
                return []
            probs = np.asarray(probs, dtype=np.float64)
            total = float(np.sum(probs))
            if total <= 1e-8:
                return []
            probs = probs / total
            accum = np.zeros_like(probs, dtype=np.float64)
            indices: List[int] = []
            for _ in range(count):
                accum += probs
                winner = int(np.argmax(accum))
                accum[winner] -= 1.0
                indices.append(winner)
            return indices

        def _log_routing_diff(tag: str, base_ratios: np.ndarray, probs: np.ndarray) -> None:
            if probs is None:
                return
            base = np.asarray(base_ratios, dtype=np.float64)
            base_clean = np.maximum(
                np.nan_to_num(base, nan=0.0, posinf=0.0, neginf=0.0), 0.0
            )
            base_sum = float(np.sum(base_clean))
            if base_sum > 1e-8:
                base_norm = base_clean / base_sum
            else:
                base_norm = np.zeros_like(base_clean)
            probs_arr = np.asarray(probs, dtype=np.float64)
            diff = base_norm - probs_arr
            l1 = float(np.sum(np.abs(diff))) if diff.size else 0.0
            max_delta = float(np.max(np.abs(diff))) if diff.size else 0.0
            zeros = int(np.sum(probs_arr <= 0)) if probs_arr.size else 0
            neighbor_len = len(neighbor_satellites) if neighbor_satellites is not None else -1
            valid_neighbors = (
                int(sum(1 for n in neighbor_satellites if n is not None))
                if neighbor_satellites is not None
                else -1
            )
            snapshot_neighbors = getattr(self, "_neighbor_snapshot", None)
            snapshot_len = len(snapshot_neighbors) if snapshot_neighbors is not None else -1
            mismatch = (
                (snapshot_len >= 0 and neighbor_len >= 0 and snapshot_len != neighbor_len)
                or (valid_neighbors >= 0 and neighbor_len >= 0 and valid_neighbors != neighbor_len)
            )
            anomaly = (
                neighbor_len <= 0
                or valid_neighbors <= 0
                or snapshot_len == 0
                or max_delta > self._routing_diff_log_threshold
                or mismatch
            )
            if anomaly and self._routing_diff_log_remaining > 0:
                self.logger.warning(
                    "[ROUTING DIFF] sat=%s tag=%s l1=%.3e max=%.3e zeros=%d "
                    "neighbor_len=%d valid_neighbors=%d snapshot_neighbors=%d "
                    "base_sum=%.3e probs_sum=%.3e",
                    self.name,
                    tag,
                    l1,
                    max_delta,
                    zeros,
                    neighbor_len,
                    valid_neighbors,
                    snapshot_len,
                    base_sum,
                    float(np.sum(probs_arr)) if probs_arr.size else 0.0,
                )
                self._routing_diff_log_remaining -= 1

        # 🔧 FIX: 预计算 sample_batch 模式的 probs，确保所有切片使用相同分布
        precomputed_high_probs = None
        precomputed_low_probs = None
        if mode == "sample_batch":
            # 预计算 high/low probs（使用当前邻居状态）
            precomputed_high_probs = self._mask_and_normalize_routing_probs(
                base_ratios=high_split_ratios,
                neighbor_satellites=neighbor_satellites,
            )
            precomputed_low_probs = self._mask_and_normalize_routing_probs(
                base_ratios=low_split_ratios,
                neighbor_satellites=neighbor_satellites,
            )
            _log_routing_diff("pre_high", high_split_ratios, precomputed_high_probs)
            _log_routing_diff("pre_low", low_split_ratios, precomputed_low_probs)
            max_count = max_slices if max_slices is not None else len(self.slice_queue)
            for idx, task_slice in enumerate(self.slice_queue):
                if idx >= max_count:
                    break
                if task_slice.is_expired(self.simulator.sim_time):
                    continue
                if getattr(task_slice, "priority", 0.5) >= priority_threshold:
                    high_batch_count += 1
                else:
                    low_batch_count += 1

        while self.slice_queue and forwards_count < max_forwards_per_step:
            if max_slices is not None and processed_count >= max_slices:
                break

            current_slice = self.slice_queue[0]

            # 检查超时
            if current_slice.is_expired(self.simulator.sim_time):
                expired_slice = self.slice_queue.pop(0)
                expired_slice.status = TaskStatus.EXPIRED
                if expired_slice.task_id not in self.expired_task_ids:
                    self.expired_tasks_count += 1
                    self.expired_task_ids.add(expired_slice.task_id)
                self.expired_tasks_buffer.append(expired_slice)
                self._log_task_expired_once(
                    expired_slice.task_id, reason="slice_queue_timeout"
                )
                if getattr(self, "_queue_diag_remaining", 0) > 0:
                    self._queue_diag_remaining -= 1
                    raw_len = len(getattr(self, "raw_task_queue", []))
                    slice_len = len(getattr(self, "slice_queue", []))
                    task_len = len(getattr(self, "task_queue", []))
                    result_len = len(getattr(self, "result_queue", []))
                    self.logger.warning(
                        f"[QUEUE DIAG] t={self.simulator.sim_time:.2f} "
                        f"sat={self.name} raw={raw_len} slice={slice_len} "
                        f"task={task_len} result={result_len}"
                    )
                processed_count += 1
                continue

            # 记录一次 Level-2 路由决策（非过期切片）
            self.level2_decisions_this_step += 1

            # 根据优先级选择路由比例
            slice_priority = getattr(current_slice, "priority", 0.5)
            is_high_priority = slice_priority >= priority_threshold
            base_ratios = high_split_ratios if is_high_priority else low_split_ratios
            num_targets = len(base_ratios)
            has_self = num_targets == len(neighbor_satellites) + 1
            self_index = num_targets - 1 if has_self else None

            # mask 不可达邻居并归一化（无有效邻居则回退本地）
            if mode == "flow":
                if is_high_priority:
                    if flow_probs_high is None:
                        flow_probs_high = self._mask_and_normalize_routing_probs(
                            base_ratios=high_split_ratios,
                            neighbor_satellites=neighbor_satellites,
                        )
                        flow_accum_high = np.zeros_like(flow_probs_high, dtype=np.float64)
                        _log_routing_diff("flow_high", high_split_ratios, flow_probs_high)
                    probs = flow_probs_high
                    if probs.size == 0 or np.sum(probs) <= 1e-8:
                        self.level2_self_selected_this_step += 1
                        slice_to_process = self.slice_queue.pop(0)
                        self.task_queue.append(slice_to_process)
                        processed_count += 1
                        continue
                    flow_accum_high += probs
                    winner_index = int(np.argmax(flow_accum_high))
                    flow_accum_high[winner_index] -= 1.0
                else:
                    if flow_probs_low is None:
                        flow_probs_low = self._mask_and_normalize_routing_probs(
                            base_ratios=low_split_ratios,
                            neighbor_satellites=neighbor_satellites,
                        )
                        flow_accum_low = np.zeros_like(flow_probs_low, dtype=np.float64)
                        _log_routing_diff("flow_low", low_split_ratios, flow_probs_low)
                    probs = flow_probs_low
                    if probs.size == 0 or np.sum(probs) <= 1e-8:
                        self.level2_self_selected_this_step += 1
                        slice_to_process = self.slice_queue.pop(0)
                        self.task_queue.append(slice_to_process)
                        processed_count += 1
                        continue
                    flow_accum_low += probs
                    winner_index = int(np.argmax(flow_accum_low))
                    flow_accum_low[winner_index] -= 1.0
            else:
                probs = self._mask_and_normalize_routing_probs(
                    base_ratios=base_ratios,
                    neighbor_satellites=neighbor_satellites,
                )
                _log_routing_diff("step", base_ratios, probs)
                if probs.size == 0 or np.sum(probs) <= 1e-8:
                    self.level2_self_selected_this_step += 1
                    slice_to_process = self.slice_queue.pop(0)
                    self.task_queue.append(slice_to_process)
                    processed_count += 1
                    continue
                if mode == "argmax":
                    winner_index = int(np.argmax(probs))
                elif mode == "sample_batch":
                    # 🔧 FIX: 使用预计算的 probs，确保所有切片使用相同分布
                    if is_high_priority:
                        batch_probs = precomputed_high_probs if precomputed_high_probs is not None else probs
                        if batch_high is None:
                            batch_high = _build_batch_samples(batch_probs, high_batch_count)
                            batch_high_idx = 0
                        if batch_high_idx < len(batch_high):
                            winner_index = int(batch_high[batch_high_idx])
                            batch_high_idx += 1
                        else:
                            winner_index = int(np.argmax(batch_probs))
                    else:
                        batch_probs = precomputed_low_probs if precomputed_low_probs is not None else probs
                        if batch_low is None:
                            batch_low = _build_batch_samples(batch_probs, low_batch_count)
                            batch_low_idx = 0
                        if batch_low_idx < len(batch_low):
                            winner_index = int(batch_low[batch_low_idx])
                            batch_low_idx += 1
                        else:
                            winner_index = int(np.argmax(batch_probs))
                else:
                    winner_index = int(np.random.choice(len(probs), p=probs))  # 探索

            print(f"[ROUTING SAMPLE] Slice {current_slice.task_id} from {self.name} routed to index {winner_index}") # 调试信息，路由没有变都是同一个

                # Debug print removed to reduce log noise.

            # 执行路由：允许 self（本地处理）或邻居
            if self_index is not None and winner_index == self_index:
                self.level2_self_selected_this_step += 1
                slice_to_process = self.slice_queue.pop(0)
                self.task_queue.append(slice_to_process)
                processed_count += 1
                continue
            if winner_index >= len(neighbor_satellites):
                self.level2_self_selected_this_step += 1
                slice_to_process = self.slice_queue.pop(0)
                self.task_queue.append(slice_to_process)
                processed_count += 1
                continue
            target_neighbor = neighbor_satellites[winner_index]
            if target_neighbor is None:
                self.level2_self_selected_this_step += 1
                slice_to_process = self.slice_queue.pop(0)
                self.task_queue.append(slice_to_process)
                processed_count += 1
                continue

            # 回环检测：已访问过则强制本地处理
            # 🔧 FIX: 统一 visited_sats 为 set 类型并确保深拷贝
            visited_raw = getattr(current_slice, "visited_sats", None)
            if visited_raw is None:
                visited = set()
            elif isinstance(visited_raw, set):
                visited = visited_raw.copy()  # 避免修改原始引用
            else:
                visited = set(visited_raw)
            if target_neighbor.name in visited:
                self.logger.debug(
                    f"[LOOP] Skip visited target for slice {current_slice.task_id} "
                    f"from {self.name} to {target_neighbor.name}"
                )
                self.level2_self_selected_this_step += 1
                slice_to_process = self.slice_queue.pop(0)
                self.task_queue.append(slice_to_process)
                processed_count += 1
                continue

            # TTL：超过最大路由跳数则强制本地处理
            current_hop = getattr(current_slice, "current_hop", 0)
            if self.max_route_hops is not None and current_hop >= self.max_route_hops:
                self.level2_self_selected_this_step += 1
                slice_to_process = self.slice_queue.pop(0)
                self.task_queue.append(slice_to_process)
                processed_count += 1
                continue

            # 🛑 防 ping-pong：禁止立即回传给上一跳
            prev_hop = None
            chain = getattr(current_slice, "offload_chain", [])
            if len(chain) >= 2:
                prev_hop = chain[-2]
            if prev_hop and target_neighbor.name == prev_hop:
                self.logger.debug(
                    f"[PINGPONG] Skip backtrack for slice {current_slice.task_id} "
                    f"from {self.name} to {target_neighbor.name}"
                )
                # 本地处理：移至 task_queue
                self.level2_self_selected_this_step += 1
                slice_to_process = self.slice_queue.pop(0)
                self.task_queue.append(slice_to_process)
                processed_count += 1
                continue

            self.offload_attempts_this_step += 1
            # ISL 传播时延
            isl_distance = self._compute_isl_distance(target_neighbor)
            isl_prop_delay = isl_distance / TaskSlice.SPEED_OF_LIGHT

            # ISL 传输时延（使用物理模型计算速率）
            try:
                # 获取当前发射功率
                tx_power = getattr(self, "current_tx_power", 0.0)
                if tx_power <= 0:
                    # 回退到默认值
                    tx_power = abs(self.sat_args.get("transmitterPowerDraw", -15.0))

                # 使用物理模型计算 ISL 速率
                isl_rate = calculate_isl_rate(
                    tx_power,
                    isl_distance,
                    max_distance=self._get_isl_max_distance_m(),
                )
            except Exception:
                # 🔧 FIX: 回退到配置的默认速率而非硬编码值
                default_baud = abs(self.sat_args.get("transmitterBaudRate", 100e6))
                isl_rate = default_baud if default_baud > 0 else 100e6  # 100 Mbps 默认

            # --- Scheme A: 链路容量截断与残留 (Backlog) ---
            # Use per-step forwarding budget to avoid repeated over-commit and queue explosion.
            if not hasattr(self, "_isl_step_budget"):
                self._isl_step_budget = {}
            budget_key = getattr(target_neighbor, "name", str(id(target_neighbor)))
            if budget_key not in self._isl_step_budget:
                self._isl_step_budget[budget_key] = max(0.0, isl_rate * self.sim_step)
            max_transferable = max(0.0, self._isl_step_budget[budget_key])

            # 容量检查（创建 backlog 前不修改切片元数据）
            is_congested = False
            backlog_slice = None
            if current_slice.data_size > max_transferable:
                is_congested = True
                remaining_data = current_slice.data_size - max_transferable

                # 1. 创建残留切片 (Backlog) - 保持原始 hop/visited 状态
                backlog_slice = copy.deepcopy(current_slice)
                backlog_slice.data_size = remaining_data
                backlog_slice.priority = getattr(current_slice, "priority", 0.5)
                backlog_slice.is_slice = True

                # 2. 修改当前切片为截断后的大小（将被转发）
                current_slice.data_size = max_transferable

            # 更新转发元数据（只作用于本次实际转发部分）
            visited.add(target_neighbor.name)
            current_slice.visited_sats = visited
            current_slice.hop_count += 1
            current_slice.current_hop = getattr(current_slice, "current_hop", 0) + 1
            current_slice.current_holder = target_neighbor.name
            if hasattr(current_slice, "offload_chain"):
                if not current_slice.offload_chain or current_slice.offload_chain[-1] != self.name:
                    current_slice.offload_chain.append(self.name)
                current_slice.offload_chain.append(target_neighbor.name)

            current_slice.t_prop_isl_total = getattr(current_slice, "t_prop_isl_total", 0.0) + isl_prop_delay

            isl_tx_delay = current_slice.data_size / isl_rate if isl_rate > 0 else 0.0
            current_slice.t_tx_isl_total = (
                getattr(current_slice, "t_tx_isl_total", 0.0) + isl_tx_delay
            )
            current_slice.last_isl_tx_delay = isl_tx_delay

            # Offload metrics (count actual transmitted bits)
            if hasattr(self, "offloaded_data_total"):
                self.offloaded_data_total += current_slice.data_size
                task_origin = getattr(current_slice, "origin_satellite", "")
                is_first_hop = (task_origin == "" or task_origin == self.name)
                if is_first_hop:
                    if hasattr(self, "first_hop_offloaded"):
                        self.first_hop_offloaded += current_slice.data_size
                    if hasattr(self, "offloaded_task_ids"):
                        self.offloaded_task_ids.add(current_slice.task_id)
                else:
                    if hasattr(self, "relay_offloaded"):
                        self.relay_offloaded += current_slice.data_size
                    if hasattr(self, "relay_task_ids"):
                        self.relay_task_ids.add(current_slice.task_id)

            # 统计
            self.sent_data_this_step += current_slice.data_size
            self.tx_time_this_step += isl_tx_delay
            self._isl_step_budget[budget_key] = max(
                0.0, self._isl_step_budget[budget_key] - current_slice.data_size
            )

            # 转发
            target_neighbor.slice_queue.append(current_slice)
            
            # 关键：移除当前已处理的切片（或者是其截断后的部分）
            self.slice_queue.pop(0) 
            
            if is_congested and backlog_slice is not None:
                # 如果发生了拥塞，将残留切片插入回头部
                self.slice_queue.insert(0, backlog_slice)
                # 停止当前 step 的队列处理（队首阻塞，且链路已满）
                # 这完美体现了“链路最大通信能力”的限制，并产生自然排队惩罚（任务滞留）
                break

            forwards_count += 1
            processed_count += 1

        return int(processed_count)

    def _mask_and_normalize_routing_probs(
        self,
        base_ratios: np.ndarray,
        neighbor_satellites: List["ComputationSatellite"],
    ) -> np.ndarray:
        """对路由概率进行 mask 并归一化（支持 self 作为候选）。

        - 不可达邻居置 0
        - 若总和为 0，返回全 0（由调用方决定回退策略）
        """
        probs = np.asarray(base_ratios, dtype=np.float64).copy()
        # Align with non-hier routing: negative values are treated as 0 to avoid invalid normalization.
        probs = np.maximum(np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
        num_targets = len(probs)

        # 归一化 base_ratios
        base_sum = np.sum(probs)
        if base_sum > 1e-8:
            probs = probs / base_sum
        else:
            probs = np.zeros(num_targets, dtype=np.float64)

        # Mask 不可达邻居（如含 self，则最后一维为 self）
        has_self = num_targets == len(neighbor_satellites) + 1
        self_index = num_targets - 1 if has_self else None
        mask_snapshot = getattr(self, "_neighbor_mask_snapshot", None)
        for i in range(num_targets):
            if self_index is not None and i == self_index:
                continue
            if i >= len(neighbor_satellites) or neighbor_satellites[i] is None:
                probs[i] = 0.0
                continue
            if (
                mask_snapshot is not None
                and i < len(mask_snapshot)
                and float(mask_snapshot[i]) <= 0.0
            ):
                probs[i] = 0.0

        total = np.sum(probs)
        if total < 1e-8:
            # Keep execution consistent with policy-side fallback: self-only routing.
            # Self is expected at the last index when present.
            fallback = np.zeros(num_targets, dtype=np.float64)
            if num_targets > 0:
                fallback[-1] = 1.0
            return fallback

        return probs / total


__doc_title__ = "Hierarchical Computation Satellite"
__all__ = [
    "HierComputationSatellite",
]
