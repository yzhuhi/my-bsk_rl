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
from Basilisk.utilities import orbitalMotion

from bsk_rl.scene import Scenario
from bsk_rl.utils import vizard
from bsk_rl.utils.orbital import lla2ecef

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
    priority: float           # 优先级（用于调度决策）
    uplink_rate: float = 10e6 # [bps] 上行传输速率（默认 10 Mbps）
    
    # 云端路径参数
    fiber_distance: float = 500e3     # [m] 网关到云服务器的光纤距离，默认 500 km
    cloud_cpu_freq: float = 10e9      # [Hz] 云端 CPU 频率，默认 10 GHz
    
    # UD 本地参数
    ud_cpu_freq: float = 1e9          # [Hz] UD 本地 CPU 频率，默认 1 GHz
    
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
    - 数据量分布
    - 计算复杂度分布
    - 时延要求分布
    
    注意：卫星轨道配置应通过 `sat_arg_randomizer` 参数完成，
    而不是在此场景中定义。
    """

    def __init__(
        self,
        n_tasks: Union[int, tuple[int, int]] = 10,
        data_size_range: tuple[float, float] = (1e6, 10e6),  # 1-10 Mb
        workload_range: tuple[float, float] = (100, 1000),    # cycles/bit
        max_delay_range: tuple[float, float] = (1.0, 10.0),   # 1-10 seconds
        priority_distribution: Optional[Callable] = None,
        task_arrival_rate: float = 0.1,  # 任务到达率 (tasks/second)
        radius: float = orbitalMotion.REQ_EARTH * 1e3,
    ) -> None:
        """初始化 STIN 计算任务场景。
        
        Args:
            n_tasks: 每个 episode 的任务数量，可以是固定值或范围 (low, high)。
            data_size_range: [bits] 任务数据量范围。
            workload_range: [cycles/bit] 计算复杂度范围。
            max_delay_range: [s] 最大时延要求范围。
            priority_distribution: 优先级生成函数，默认为 uniform(0, 1)。
            task_arrival_rate: [tasks/s] 任务到达率（泊松过程）。
            radius: [m] 任务来源位置的地球半径。
        """
        super().__init__()
        self._n_tasks = n_tasks
        self.data_size_range = data_size_range
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
        """在 Vizard 中可视化任务位置。"""
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
        for i in range(self.n_tasks):
            # 均匀分布的地理位置
            x = np.random.normal(size=3)
            x *= self.radius / np.linalg.norm(x)
            
            # 随机化任务属性
            data_size = np.random.uniform(*self.data_size_range)
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
        """根据优先级采样一个任务。
        
        可用于任务到达逻辑。
        
        Returns:
            采样的任务，如果没有任务则返回 None。
        """
        if not self.tasks:
            return None
        
        # 按优先级加权采样
        priorities = np.array([t.priority for t in self.tasks])
        probabilities = priorities / priorities.sum()
        idx = np.random.choice(len(self.tasks), p=probabilities)
        return self.tasks[idx]
    
    def process_task_arrivals(self, step_duration: float) -> int:
        """处理当前步的任务到达（泊松过程）。
        
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
        
        # 从任务池中随机选择到达的任务
        if actual_arrivals > 0:
            # 随机选择到达的任务索引
            arrived_indices = np.random.choice(
                len(self.task_pool), 
                size=actual_arrivals, 
                replace=False
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
        """将任务分配给具有可见性的卫星。
        
        在每个 step 中调用，检查哪些任务可以被哪些卫星接入。
        使用 AccessSatellite 的 opportunities 机制。
        
        注意：只处理已到达的任务(arrived_tasks)，不处理待到达池(task_pool)
        """
        current_time = self.satellites[0].simulator.sim_time
        
        for task in self.arrived_tasks:  # 只处理已到达的任务
            # 检查任务是否已被分配
            if hasattr(task, '_assigned') and task._assigned:
                continue
            
            # 检查是否有卫星可见此任务
            for satellite in self.satellites:
                if not hasattr(satellite, 'receive_task_from_scenario'):
                    continue
                
                # 检查任务是否在卫星的可见窗口内
                opportunities = satellite.find_next_opportunities(
                    n=1, types="task", pad=False
                )
                
                for opp in opportunities:
                    if opp.get('object') == task:
                        window = opp.get('window', (float('inf'), float('inf')))
                        # 检查当前时间是否在窗口内
                        if window[0] <= current_time <= window[1]:
                            # 分配任务给此卫星
                            satellite.receive_task_from_scenario(task)
                            task._assigned = True
                            logger.info(
                                f"Task {task.name} assigned to {satellite.name} "
                                f"at t={current_time:.2f}s"
                            )
                            break
                
                if hasattr(task, '_assigned') and task._assigned:
                    break


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
            
            # 随机化任务属性
            data_size = np.random.uniform(*self.data_size_range)
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