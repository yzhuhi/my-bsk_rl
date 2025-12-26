# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: MIT

from typing import Callable, Dict, List, Optional
from dataclasses import dataclass, MISSING
import copy
import numpy as np
from tensordict import TensorDictBase

from benchmarl.environments.common import Task, TaskClass
from benchmarl.utils import DEVICE_TYPING

from torchrl.data import CompositeSpec
from torchrl.envs import EnvBase, PettingZooWrapper
from torchrl.envs.transforms import DoubleToFloat

from bsk_rl import ConstellationTasking
from bsk_rl.data import STINTaskReward
from bsk_rl.scene import CityTaskScenario
from bsk_rl.sats import ComputationSatellite
from bsk_rl.comm import LOSMultiCommunication
from bsk_rl.utils.orbital import walker_delta_args


# ------------------ 环境与任务类 ------------------
class LoSComEnvTask(Task):
    """LoSComEnv 任务枚举 (配置从 conf/task/mylossatenv 加载)"""
    TASK_1 = None  # 默认任务配置
    TASK_2 = None  # 可选任务配置 (如不同星座规模/任务密度)
    TASK_3 = None  # 可选任务配置

    @staticmethod
    def associated_class():
        return LoSComEnvClass


class LoSComEnvClass(TaskClass):
    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        config = copy.deepcopy(self.config)
        
        def env_creator():
            # 星座配置 (从 config 读取)
            constellation_args = walker_delta_args(
                n_planes=config["n_planes"],
                altitude=config["altitude"],
                inc=config["inclination"],
                clusterspacing=config["cluster_spacing"],
            )
            
            # 任务场景配置 (从 config 读取)
            # 优先级分布函数（从配置读取范围，生成均匀分布）
            priority_min = config.get("priority_min", 0.1)
            priority_max = config.get("priority_max", 1.0)
            priority_fn = lambda: np.random.uniform(priority_min, priority_max)
            
            task_scenario = CityTaskScenario(
                n_tasks=config["n_tasks"],
                n_select_from=config["n_select_from"],
                data_size_range=(config["data_size_min"], config["data_size_max"]),
                data_size_mean=config.get("data_size_mean"),  # 截断正态分布均值
                data_size_std=config.get("data_size_std"),    # 截断正态分布标准差
                workload_range=(config["workload_min"], config["workload_max"]),
                max_delay_range=(config["max_delay_min"], config["max_delay_max"]),
                priority_distribution=priority_fn,  # 优先级分布函数
                task_arrival_rate=config["task_arrival_rate"],
            )
            
            # 奖励函数配置 (从 config 读取)
            reward_system = STINTaskReward(
                base_reward=config["base_reward"],
                timeout_penalty=config["timeout_penalty"],
                delay_weight=config["delay_weight"],
                energy_weight=config["energy_weight"],
                balance_weight=config["balance_weight"],
                latency_penalty=config["latency_penalty"],
                battery_penalty=config["battery_penalty"],
                battery_safety_threshold=config["battery_safety_threshold"],
                collaboration_weight=config["collaboration_weight"],
            )
            
            # 卫星硬件参数 (从 config 读取)
            sat_args = {
                "cpuMaxFrequency": config["cpu_max_frequency"],
                "cpuMinFrequency": config["cpu_min_frequency"],
                "cpuWorkload": config["cpu_workload"],
                "cpuPowerDraw": config["cpu_power_draw"],
                "transmitterBaudRate": config["transmitter_baud_rate"],
                "transmitterPowerDraw": config["transmitter_power_draw"],
                "basePowerDraw": config["base_power_draw"],
                "batteryStorageCapacity": config["battery_capacity"],
                "storedCharge_Init": config["initial_charge"],
                "panelArea": config["panel_area"],
                "taskMinimumElevation": config["task_min_elevation"],
                "dataStorageCapacity": config["data_storage_capacity"],
                # 航天器物理参数（bsk_rl 会自动计算转动惯量）
                "mass": config.get("sat_mass", 680.0),
                "width": config.get("sat_width", 1.8),
                "depth": config.get("sat_depth", 1.2),
                "height": config.get("sat_height", 1.2),
                "u_max": config.get("rw_u_max", 1.0),
                "maxWheelSpeed": config.get("max_wheel_speed", 6000.0),
                # MRP Steering 控制器参数 (SteeringImagerFSWModel)
                "K1": config.get("K1", 0.25),
                "K3": config.get("K3", 3.0),
                "omega_max": config.get("omega_max", 0.035),  # [rad/s] 5 deg/s
                "servo_Ki": config.get("servo_Ki", 5.0),
                "servo_P": config.get("servo_P", 5.0),
            }


            # 实例化 bsk_rl 环境
            pz_env = ConstellationTasking(
                satellites=[
                    ComputationSatellite(
                        name=f"sat-{i}",
                        sat_args=sat_args,
                    )
                    for i in range(config["n_satellites"])
                ],
                scenario=task_scenario,
                rewarder=reward_system,
                communicator=LOSMultiCommunication(),
                sat_arg_randomizer=constellation_args,
                max_step_duration=config["max_step_duration"],
                time_limit=config["time_limit"],
                log_level=config["log_level"],
            )
            
            # 使用 TorchRL 的 PettingZooWrapper 包装
            # 关键：传入 group_map 使 TorchRL 按 group 组织 specs (BenchMARL 需要这种结构)
            n_sats = config["n_satellites"]
            return PettingZooWrapper(
                env=pz_env,
                device=device,
                group_map={"agents": [f"sat-{i}" for i in range(n_sats)]},
            )

        return env_creator

    # --- 接口配置 ---

    def supports_continuous_actions(self) -> bool:
        return True # ComputationSatellite 使用连续动作 (CPU频率、传输功率等)

    def supports_discrete_actions(self) -> bool:
        return False

    def has_render(self, env: EnvBase) -> bool:
        return False
    
    def max_steps(self, env: EnvBase) -> int:
        return self.config.get("max_steps", 600)

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        # PettingZooWrapper 创建时传入了 group_map，直接使用它
        if hasattr(env, "group_map"):
            return env.group_map
        # 回退: 使用配置文件中的卫星数量来生成 agent 列表
        n_sats = self.config.get("n_satellites", 6)
        return {"agents": [f"sat-{i}" for i in range(n_sats)]}

    def observation_spec(self, env: EnvBase) -> CompositeSpec:
        # PettingZooWrapper with group_map 会按 group 组织 specs
        # 结构变成: {group_name: {observation: ..., other_keys: ...}}
        observation_spec = env.observation_spec.clone()
        for group in self.group_map(env):
            if group in observation_spec.keys():
                group_obs_spec = observation_spec[group]
                for key in list(group_obs_spec.keys()):
                    if key != "observation":
                        del group_obs_spec[key]
        if "state" in observation_spec.keys():
            del observation_spec["state"]
        return observation_spec

    def action_spec(self, env: EnvBase) -> CompositeSpec:
        return env.full_action_spec

    def state_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        return None

    def action_mask_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        # 如果 bsk_rl 提供了 action_mask，这里可以保留，否则返回 None
        # 目前 ImagingSatellite 默认可能不提供 mask，视具体实现而定
        return None

    def info_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        return None

    # --- 关键：数据类型转换 ---
    def get_env_transforms(self, env: EnvBase):
        # bsk_rl 输出 float64，必须转为 float32 才能输入神经网络
        return [DoubleToFloat()]

    @staticmethod
    def env_name() -> str:
        return "mylossatenv"

    def log_info(self, batch: TensorDictBase) -> Dict[str, float]:
        """Extract additional metrics from batch for wandb logging.
        
        Note: Most physical metrics (completed_tasks, expired_tasks, energy_consumed, 
        etc.) are automatically extracted by logger.py from batch["next", "info"].
        
        Automatic logging prefix: collection/agents/info/{metric_name}
        
        This method is reserved for custom metrics that need unit conversion
        or additional processing beyond what auto-extraction provides.
        """
        # 所有指标已通过 logger.py 的自动机制从 info 中提取
        # 自动记录格式: collection/agents/info/{key} 和 collection/info/{key}
        # 因此这里返回空字典，避免冗余
        return {}

