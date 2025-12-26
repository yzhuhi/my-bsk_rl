"""
STINScenario: 定义 STIN 场景中的计算任务分布。

设计说明:
    - 继承自 Scenario 基类（参考 UniformTargets 的设计）
    - 只负责生成计算任务（用户终端请求），不负责卫星轨道配置
    - 卫星轨道配置应通过 `sat_arg_randomizer=walker_delta_args(...)` 完成
    - 计算任务包含：数据量、计算复杂度、最大时延等属性

使用示例:
    >>> from bsk_rl import ConstellationTasking
    >>> from bsk_rl.utils.orbital import walker_delta_args
    >>> from bsk_rl.scene import STINTaskScenario
    >>> from bsk_rl.sats import ComputationSatellite
    >>>
    >>> env = ConstellationTasking(
    ...     satellites=[ComputationSatellite() for _ in range(6)],
    ...     sat_arg_randomizer=walker_delta_args(  # 星座在这里配置！
    ...         n_planes=1,
    ...         altitude=800,
    ...         inc=60,
    ...         clusterspacing=5,
    ...     ),
    ...     scenario=STINTaskScenario(n_tasks=10),  # 场景只负责任务
    ... )
"""

import logging
from typing import TYPE_CHECKING, Callable, Optional, Union
from dataclasses import dataclass

import numpy as np
from scipy.stats import truncnorm
from Basilisk.utilities import orbitalMotion

from bsk_rl.scene import Scenario
from bsk_rl.utils import vizard
from bsk_rl.utils.orbital import lla2ecef
from bsk_rl.utils.constants import DEFAULT_UPLINK_RATE, DEFAULT_FIBER_DISTANCE, DEFAULT_CLOUD_CPU_FREQ, DEFAULT_UD_CPU_FREQ

if TYPE_CHECKING:
    from bsk_rl.sats import Satellite

logger = logging.getLogger(__name__)


@dataclass
class ComputationTask:
    """STIN 场景中的计算任务。
    
    表示用户终端（UD）发起的计算请求。
    
    时延模型所需属性:
        - origin_position: 任务来源位置 (= r_LP_P)
        - uplink_distance: 上行链路距离（需要在任务分配时计算）
        
    并行路径时延模型:
        T_total = max(T_UD, T_SAT, T_CLOUD)
        
        - T_UD: UD 本地处理时延 = (alpha_local × data × workload) / f_ud
        - T_SAT: 卫星路径时延 = T_tx_up(α_sat) + T_prop_up + T_queue + T_compute + T_isl + T_prop_down
        - T_CLOUD: 云端路径时延 = T_tx_up(α_cloud) + T_tx_sgl(α_cloud) + 2×(T_prop_up + T_prop_sgl + T_fiber) + T_compute_cloud
    """
    name: str
    r_LP_P: np.ndarray        # [m] 任务来源位置（Planet-fixed）
    data_size: float          # [bits] 输入数据量
    workload: float           # [cycles/bit] 计算复杂度
    max_delay: float          # [s] 最大容忍时延
    priority: float           # 优先级 [0, 1]，详见下方说明 / 12-26 改动
    # ------------------------------------------------------------------
    # priority 字段设计说明：
    # 
    # ✅ 已实现功能：
    #   - 可视化标记大小 (markerScale = sqrt(priority))
    #   - 任务到达优先级：高优先级任务更可能先到达 (process_task_arrivals)
    #   - 分配优先级：高优先级任务分配给资源更充足的卫星 (assign_tasks_to_satellites)
    #   - Agent 观测：current_task_priority 已加入观测空间 (41D)
    #   - 奖励函数：超时惩罚已使用 priority 加权
    # 
    # 🔮 Future Work（卫星层优先级队列）：
    #   - 卫星端按优先级处理：在 task_queue 中使用优先级队列替代 FIFO
    #     实现方式：max(task_queue, key=lambda t: t.priority) 或 heapq
    #   - 奖励函数权重：完成高优先级任务获得更高奖励 （已实现）
    #   - 分级处理 (SLA tiers)：根据优先级设置不同的 max_delay 乘数
    #   - 抛弃策略：资源紧张时优先抛弃低优先级任务
    # ------------------------------------------------------------------
    uplink_rate: float = DEFAULT_UPLINK_RATE  # [bps] 上行传输速率
    
    # 云端路径参数
    fiber_distance: float = DEFAULT_FIBER_DISTANCE    # [m] 网关到云服务器的光纤距离，默认 500 km
    cloud_cpu_freq: float = DEFAULT_CLOUD_CPU_FREQ       # [Hz] 云端 CPU 频率，默认 10 GHz
    
    # UD 本地参数
    ud_cpu_freq: float = DEFAULT_UD_CPU_FREQ           # [Hz] UD 本地 CPU 频率，默认 1 GHz
    
    @property
    def origin_position(self) -> np.ndarray:
        """任务来源位置别名（用于 TaskSlice 兼容性）。"""
        return self.r_LP_P
    
    @property
    def id(self) -> str:
        """获取唯一标识符。"""
        try:
            return self._id
        except AttributeError:
            self._id = f"{self.name}_{id(self)}"
            return self._id
    
    @property
    def task_id(self) -> int:
        """数值型任务 ID（从名称中提取）。"""
        try:
            return int(self.name.split('-')[-1])
        except (ValueError, IndexError):
            return hash(self.name) % 100000
    
    @property
    def total_cycles(self) -> float:
        """计算任务所需的总 CPU 周期数。"""
        return self.data_size * self.workload
    
    def __repr__(self) -> str:
        return f"ComputationTask({self.name}, {self.data_size/1e6:.1f}Mb)"


class STINTaskScenario(Scenario):
    """STIN 计算任务场景：定义计算任务的分布。
    
    任务属性包括：
    - 地理位置（用户终端位置）
    - 数据量分布 截断正态分布
    - 计算复杂度分布
    - 时延要求分布
    
    注意：卫星轨道配置应通过 `sat_arg_randomizer` 参数完成，
    而不是在此场景中定义。
    """

    def __init__(
        self,
        n_tasks: Union[int, tuple[int, int]] = 10,
        data_size_range: tuple[float, float] = (1e6, 10e6),  # (a, b) 截断范围 [bits]
        data_size_mean: Optional[float] = None,  # 均值 μ，默认为范围中心
        data_size_std: Optional[float] = None,   # 标准差 σ，默认为范围的1/4
        workload_range: tuple[float, float] = (100, 1000),    # cycles/bit
        max_delay_range: tuple[float, float] = (1.0, 10.0),   # 1-10 seconds
        priority_distribution: Optional[Callable] = None, # 12-26 改动：优先级生成函数被真正输入
        task_arrival_rate: float = 0.1,  # 任务到达率 (tasks/second)
        radius: float = orbitalMotion.REQ_EARTH * 1e3,
    ) -> None:
        """初始化 STIN 计算任务场景。
        
        Args:
            n_tasks: 每个 episode 的任务数量，可以是固定值或范围 (low, high)。
            data_size_range: [bits] 任务数据量截断范围 (a, b)，对应论文公式 (3)。
            data_size_mean: [bits] 数据量均值 μ，默认为 (a+b)/2。
            data_size_std: [bits] 数据量标准差 σ，默认为 (b-a)/4。
            workload_range: [cycles/bit] 计算复杂度范围。
            max_delay_range: [s] 最大时延要求范围。
            priority_distribution: 优先级生成函数，默认为 uniform(0, 1)。 
            task_arrival_rate: [tasks/s] 任务到达率（泊松过程）。
            radius: [m] 任务来源位置的地球半径。 
        """
        super().__init__()
        self._n_tasks = n_tasks
        self.data_size_range = data_size_range
        # 截断正态分布参数 (论文公式 3) 12-26 改动：允许自定义均值和标准差
        a, b = data_size_range
        self.data_size_mean = data_size_mean if data_size_mean is not None else (a + b) / 2
        self.data_size_std = data_size_std if data_size_std is not None else (b - a) / 4
        
        self.workload_range = workload_range
        self.max_delay_range = max_delay_range
        self.task_arrival_rate = task_arrival_rate
        self.radius = radius
        
        if priority_distribution is None:
            priority_distribution = lambda: np.random.rand()
        self.priority_distribution = priority_distribution
        
        # 任务列表
        self.tasks: list[ComputationTask] = []
        # 泊松到达机制：任务池（待到达）和已到达队列
        self.task_pool: list[ComputationTask] = []  # 待到达任务池
        self.arrived_tasks: list[ComputationTask] = []  # 已到达任务队列

    def reset_overwrite_previous(self) -> None:
        """清除上一个 episode 的任务列表。"""
        self.tasks = []
        self.task_pool = []
        self.arrived_tasks = []

    def reset_pre_sim_init(self) -> None:
        """在仿真初始化前生成任务集。
        
        注意：任务生成后放入task_pool，等待泊松过程动态到达
        """
        # 确定任务数量
        if isinstance(self._n_tasks, int):
            self.n_tasks = self._n_tasks
        else:
            self.n_tasks = np.random.randint(self._n_tasks[0], self._n_tasks[1] + 1)
        
        logger.info(f"Generating {self.n_tasks} computation tasks for Poisson arrival")
        self._regenerate_tasks()
        
        # 将所有任务放入待到达池
        self.task_pool = self.tasks.copy()
        self.arrived_tasks = []
        
        # 注意：不再在这里注册任务的access checking
        # 任务将在到达时动态注册（在process_task_arrivals中）

    def reset_during_sim_init(self) -> None:
        """在仿真初始化期间可视化任务。"""
        for task in self.tasks:
            self._visualize_task(task)

    @vizard.visualize
    def _visualize_task(self, task, vizSupport=None, vizInstance=None):
        """在 Vizard 中可视化任务位置。
        
        此方法由 @vizard.visualize 装饰器控制，仅在 Vizard 可视化模式启用时执行。
        任务位置会在地球表面显示为标记点，优先级高的任务标记更大。
        
        使用示例（无需手动调用，由 reset_during_sim_init 自动触发）::
        
            # 启用 Vizard 可视化
            >>> env = ConstellationTasking(
            ...     satellites=[...],
            ...     scenario=STINTaskScenario(n_tasks=100),
            ...     viz_args=dict(openBrowserOnRun=True),  # 启用 Vizard
            ... )
            >>> env.reset()  # 任务位置自动可视化
        
        Args:
            task: 要可视化的 ComputationTask 对象。
            vizSupport: Vizard 支持模块（由装饰器注入）。
            vizInstance: Vizard 实例（由装饰器注入）。
        
        Note:
            任务优先级 (priority) 用于设置可视化标记大小：
            markerScale = sqrt(priority)，优先级越高标记越大。
        """
        vizSupport.addLocation(
            vizInstance,
            stationName=task.name,
            parentBodyName="earth",
            r_GP_P=list(task.r_LP_P),
            fieldOfView=np.arctan(500 / 800),
            color=vizSupport.toRGBA255("cyan"),
            range=1000.0 * 1000,
            markerScale=np.sqrt(task.priority),
        )
        if vizInstance.settings.showLocationCones == 0:
            vizInstance.settings.showLocationCones = -1
        if vizInstance.settings.showLocationLabels == 0:
            vizInstance.settings.showLocationLabels = -1

    def _regenerate_tasks(self) -> None:
        """生成均匀分布的计算任务。
        
        可以在子类中重写此方法以实现其他分布（如城市分布）。
        """
        self.tasks = []
        
        # 计算截断正态分布的标准化边界 (论文公式 3) 12-26 改动
        a, b = self.data_size_range
        mu, sigma = self.data_size_mean, self.data_size_std
        a_std = (a - mu) / sigma  # 标准化下界
        b_std = (b - mu) / sigma  # 标准化上界
        
        for i in range(self.n_tasks):
            # 均匀分布的地理位置
            x = np.random.normal(size=3)
            x *= self.radius / np.linalg.norm(x)
            
            # 截断正态分布采样数据量 (论文公式 3)
            data_size = truncnorm.rvs(a_std, b_std, loc=mu, scale=sigma)
            data_size = np.clip(data_size, a, b)  # 确保在范围内
            
            # 其他属性仍用均匀分布
            workload = np.random.uniform(*self.workload_range)
            max_delay = np.random.uniform(*self.max_delay_range)
            
            task = ComputationTask(
                name=f"task-{i}",
                r_LP_P=x,
                data_size=data_size,
                workload=workload,
                max_delay=max_delay,
                priority=self.priority_distribution(),
            )
            self.tasks.append(task)

    def sample_task(self) -> Optional[ComputationTask]:
        """根据优先级加权采样一个任务。
        
        **当前状态**: 已实现但未被调用（保留接口）。
        
        **设计用途**: 
            实现优先级驱动的任务选择，高优先级任务被选中的概率更高。
        
        **启用方法**:
            在 process_task_arrivals() 中使用此方法替代随机选择::
            
                # 当前代码（随机选择）:
                arrived_indices = np.random.choice(len(self.task_pool), ...)
                
                # 改为优先级驱动:
                for _ in range(actual_arrivals):
                    task = self.sample_task_from_pool()  # 需要实现此变体
                    ...
        
        Returns:
            采样的任务，如果没有任务则返回 None。
            
        Note:
            采样概率 P(task_i) = priority_i / sum(priorities)
        """
        if not self.tasks:
            return None
        
        # 按优先级加权采样
        priorities = np.array([t.priority for t in self.tasks])
        probabilities = priorities / priorities.sum()
        idx = np.random.choice(len(self.tasks), p=probabilities)
        return self.tasks[idx]
    
    def process_task_arrivals(self, step_duration: float) -> int:
        """ 改
        处理当前步的任务到达（泊松过程）。
        
        根据task_arrival_rate和step_duration，使用泊松分布采样本步到达的任务数，
        并将任务从task_pool移到arrived_tasks，同时注册到卫星的access checking。
        
        Args:
            step_duration: 当前步的时长(秒)
        
        Returns:
            本步到达的任务数
        """
        if len(self.task_pool) == 0:
            return 0
        
        # 计算期望到达数：λ = task_arrival_rate (tasks/s) × step_duration (s)
        expected_arrivals = self.task_arrival_rate * step_duration
        
        # 泊松采样实际到达数
        actual_arrivals = np.random.poisson(expected_arrivals)
        # 不能超过任务池剩余数量
        actual_arrivals = min(actual_arrivals, len(self.task_pool))
        
        # 从任务池中按优先级加权选择到达的任务 (创新点: 优先级驱动)
        if actual_arrivals > 0:
            # 按优先级加权采样（高优先级任务更可能先到达）
            priorities = np.array([t.priority for t in self.task_pool])
            probabilities = priorities / priorities.sum()
            arrived_indices = np.random.choice(
                len(self.task_pool), 
                size=actual_arrivals, 
                replace=False,
                p=probabilities
            )
            
            # 从后往前删除，避免索引变化
            for idx in sorted(arrived_indices, reverse=True):
                task = self.task_pool.pop(idx)
                self.arrived_tasks.append(task)
                
                # 注册任务到所有卫星的access checking
                for satellite in self.satellites:
                    if hasattr(satellite, "add_location_for_access_checking"):
                        min_elev = satellite.sat_args_generator.get(
                            "taskMinimumElevation",
                            satellite.sat_args_generator.get(
                                "imageTargetMinimumElevation", 0.0
                            )
                        )
                        satellite.add_location_for_access_checking(
                            object=task,
                            r_LP_P=task.r_LP_P,
                            min_elev=min_elev,
                            type="task",
                        )
            
            logger.debug(
                f"Poisson arrival: {actual_arrivals} tasks arrived "
                f"(pool: {len(self.task_pool)} remaining, "
                f"arrived: {len(self.arrived_tasks)} total)"
            )
        
        return actual_arrivals
    
    def assign_tasks_to_satellites(self) -> None:
        """ 改
        将任务分配给具有可见性的卫星（优先级驱动）。
        
        创新点：
        1. 按优先级排序处理任务（高优先级任务优先分配）
        2. 优先级感知的卫星选择（高优先级任务分配给资源更充足的卫星）
        """
        current_time = self.satellites[0].simulator.sim_time
        
        # 统计本次分配
        assigned_count = 0
        
        # 按优先级降序排列待分配任务（高优先级优先）
        pending_tasks = [
            t for t in self.arrived_tasks 
            if not (hasattr(t, '_assigned') and t._assigned)
        ]
        pending_tasks.sort(key=lambda t: t.priority, reverse=True)
        
        for task in pending_tasks:
            # 从任务获取 UD 位置
            task_position = getattr(task, 'r_LP_P', None)
            if task_position is None:
                continue
                
            # 找到能看到这个任务的卫星列表
            visible_satellites = []
            for satellite in self.satellites:
                if not hasattr(satellite, 'receive_task_from_scenario'):
                    continue
                if not hasattr(satellite, '_check_ud_visibility'):
                    continue
                    
                # 检查可见性
                if satellite._check_ud_visibility(task_position):
                    visible_satellites.append(satellite)
            
            # 如果有可见的卫星，根据优先级选择最优卫星
            if visible_satellites:
                # 高优先级任务：选择资源最充足的卫星（队列短 + 电量高）
                # 低优先级任务：选择队列最短的卫星（简单负载均衡）
                if task.priority > 0.7:  # 高优先级阈值
                    # 综合评分：队列越短越好，电量越高越好
                    def sat_score(s):
                        queue_len = len(s.task_queue) if hasattr(s, 'task_queue') else 0
                        battery_ratio = getattr(s, 'battery_charge_ratio', 1.0)
                        return queue_len - battery_ratio * 10  # 电量每10%抵消1个队列任务
                    best_sat = min(visible_satellites, key=sat_score)
                else:
                    # 普通任务：简单负载均衡
                    best_sat = min(
                        visible_satellites,
                        key=lambda s: len(s.task_queue) if hasattr(s, 'task_queue') else 0
                    )
                best_sat.receive_task_from_scenario(task)
                task._assigned = True
                assigned_count += 1
        
        if assigned_count > 0:
            logger.warning(
                f"[DIAG] ✓ Assigned {assigned_count} tasks (priority-driven) at t={current_time:.2f}s"
            )



class CityTaskScenario(STINTaskScenario):
    """基于城市分布的 STIN 计算任务场景。
    
    任务来源于人口密集区域（城市），更符合实际的用户终端分布。
    """

    def __init__(
        self,
        n_tasks: Union[int, tuple[int, int]] = 10,
        n_select_from: Optional[int] = 100,  # 从前 N 个最大城市中选择
        location_offset: float = 50000,  # [m] 位置偏移
        **kwargs,
    ) -> None:
        """初始化城市分布的计算任务场景。
        
        Args:
            n_tasks: 任务数量。
            n_select_from: 从最大的 N 个城市中采样，None 表示全部。
            location_offset: [m] 任务位置相对城市中心的随机偏移。
            **kwargs: 传递给 STINTaskScenario 的其他参数。
        """
        super().__init__(n_tasks=n_tasks, **kwargs)
        self.n_select_from = n_select_from
        self.location_offset = location_offset

    def _regenerate_tasks(self) -> None:
        """基于城市分布生成计算任务。"""
        import os
        import sys
        from pathlib import Path
        import pandas as pd
        
        self.tasks = []
        
        # 加载城市数据库
        cities = pd.read_csv(
            Path(os.path.realpath(__file__)).parent.parent
            / "_dat"
            / "simplemaps_worldcities"
            / "worldcities.csv",
        )
        
        n_select = self.n_select_from
        if n_select is None or n_select > len(cities):
            n_select = len(cities)
        
        # 计算截断正态分布的标准化边界 (论文公式 3)
        a, b = self.data_size_range
        mu, sigma = self.data_size_mean, self.data_size_std
        a_std = (a - mu) / sigma
        b_std = (b - mu) / sigma
        
        # 从前 N 个城市中随机选择
        for i in np.random.choice(n_select, self.n_tasks, replace=False):
            city = cities.iloc[i]
            location = lla2ecef(city["lat"], city["lng"], self.radius)
            
            # 添加随机偏移
            if self.location_offset > 0:
                offset = np.random.normal(size=3)
                offset /= np.linalg.norm(offset)
                offset *= self.location_offset * np.random.rand()
                location += offset
                location /= np.linalg.norm(location)
                location *= self.radius
            
            # 截断正态分布采样数据量 (论文公式 3)
            data_size = truncnorm.rvs(a_std, b_std, loc=mu, scale=sigma)
            data_size = np.clip(data_size, a, b)
            
            # 其他属性仍用均匀分布
            workload = np.random.uniform(*self.workload_range)
            max_delay = np.random.uniform(*self.max_delay_range)
            
            task = ComputationTask(
                name=f"{city['city']}, {city['iso2']}".replace("'", ""),
                r_LP_P=location,
                data_size=data_size,
                workload=workload,
                max_delay=max_delay,
                priority=self.priority_distribution(),
            )
            self.tasks.append(task)


__doc_title__ = "STIN Task Scenarios"
__all__ = ["ComputationTask", "STINTaskScenario", "CityTaskScenario"]