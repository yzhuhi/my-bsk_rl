"""STIN MARL Environment Usage Example.

This example demonstrates how to set up and use the STIN (Satellite-Terrestrial 
Integrated Network) multi-agent reinforcement learning environment.
"""

from bsk_rl import ConstellationTasking
from bsk_rl.data import STINTaskReward
from bsk_rl.scene import CityTaskScenario
from bsk_rl.sats import ComputationSatellite
from bsk_rl.utils.orbital import walker_delta_args

# ============================================================================
# 1. 创建 STIN MARL 环境
# ============================================================================

def create_stin_environment():
    """创建完整的 STIN MARL 环境。"""
    '''
    constellation_args = walker_delta_args(
        n_planes=6,           # 铱星有 6 个轨道面
        n_sats_per_plane=11,  # 每个面 11 颗工作卫星 (总计 66 颗)
        altitude=781,         # [km] 标准高度约为 781 km
        inc=86.4,             # [deg] 轨道倾角 (近极地轨道)
        clusterspacing=2,     # [F参数] 铱星的相位因子是 2 (Walker 66/6/2)
    )
    '''
    # Walker Delta 星座配置（6 颗卫星，800km 高度，60° 倾角）
    constellation_args = walker_delta_args(
        n_planes=2,           # 2 个轨道面
        n_sats_per_plane=3,   # 每个面 3 颗卫星
        altitude=800,         # [km] 轨道高度
        inc=60,               # [deg] 轨道倾角
        clusterspacing=5,     # [deg] 卫星间相位差
    )
    
    # 计算任务场景（基于城市分布）
    task_scenario = CityTaskScenario(
        n_tasks=20,                          # 20 个任务
        n_select_from=100,                   # 从前 100 大城市中选择
        data_size_range=(1e6, 10e6),         # 1-10 Mb
        workload_range=(100, 1000),          # 100-1000 cycles/bit
        max_delay_range=(5.0, 20.0),         # 5-20 秒时延约束
        task_arrival_rate=0.1,               # 0.1 tasks/s 到达率
    )
    
    # 奖励函数配置
    reward_system = STINTaskReward(
        base_reward=1.0,        # 基础完成奖励
        timeout_penalty=2.0,    # 超时惩罚
        delay_weight=0.5,       # 时延权重
        energy_weight=0.1,      # 能耗权重
        balance_weight=0.05,    # 负载均衡权重
    )
    
    # 创建环境
    env = ConstellationTasking(
        satellites=[
            ComputationSatellite(
                name=f"sat-{i}",
                sat_args={
                    "cpuMaxFrequency": 2.0e9,     # 2 GHz
                    "cpuMinFrequency": 0.5e9,     # 0.5 GHz
                    "cpuWorkload": 500.0,         # 500 cycles/bit
                    "batteryStorageCapacity": 80.0 * 3600,  # 80 Wh
                    "taskMinimumElevation": 10.0 * 3.14159 / 180,  # 10°
                }
            )
            for i in range(6)
        ],
        sat_arg_randomizer=constellation_args,
        scenario=task_scenario,
        data=reward_system,
        sim_rate=1.0,
        max_step_duration=60.0,  # 60 秒 per step
        time_limit=3600.0,       # 1 小时 episode
    )
    
    return env


# ============================================================================
# 2. 使用环境进行训练
# ============================================================================

def example_training_loop():
    """示例训练循环（使用随机动作）。"""
    
    env = create_stin_environment()
    
    # 重置环境
    env.reset()
    
    done = False
    total_reward = 0.0
    step_count = 0
    
    while not done:
        # 随机动作（实际训练中使用 RL 算法）
        actions = env.action_space.sample()
        
        # 执行动作
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        # 累计奖励
        step_reward = sum(rewards.values())
        total_reward += step_reward
        step_count += 1
        
        # 打印步骤信息
        print(f"Step {step_count}: Reward = {step_reward:.3f}, "
              f"Cumulative = {total_reward:.3f}")
        
        # 检查是否结束
        done = any(terminated.values()) or any(truncated.values())
    
    print(f"\nEpisode finished after {step_count} steps")
    print(f"Total reward: {total_reward:.3f}")
    
    env.close()


# ============================================================================
# 3. 分析任务完成情况
# ============================================================================

def analyze_task_completion(env):
    """分析任务完成情况和性能指标。"""
    
    total_completed = 0
    total_expired = 0
    total_delay = []
    
    for sat in env.satellites.values():
        if hasattr(sat, 'completed_tasks_count'):
            total_completed += sat.completed_tasks_count
            total_expired += sat.expired_tasks_count
            
            print(f"\n{sat.name} Statistics:")
            print(f"  Completed: {sat.completed_tasks_count}")
            print(f"  Expired: {sat.expired_tasks_count}")
            print(f"  Processed Data: {sat.processed_data_total/1e6:.2f} Mb")
            print(f"  Offloaded Data: {sat.offloaded_data_total/1e6:.2f} Mb")
    
    print(f"\n=== Global Statistics ===")
    print(f"Total Completed: {total_completed}")
    print(f"Total Expired: {total_expired}")
    completion_rate = total_completed / (total_completed + total_expired) * 100
    print(f"Completion Rate: {completion_rate:.2f}%")



if __name__ == "__main__":
    # 运行示例
    print("Creating STIN MARL Environment...")
    env = create_stin_environment()
    
    print("\nEnvironment created successfully!")
    print(f"Number of satellites: {len(env.satellites)}")
    print(f"Number of tasks: {len(env.scenario.tasks)}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # 运行一个 episode
    print("\n" + "="*60)
    print("Running example training loop...")
    print("="*60)
    example_training_loop()
