from dataclasses import dataclass, MISSING


@dataclass
class TaskConfig:
    """Configuration for MyTask environment."""
    # 这些参数会被加载到 self.config 中
    """LoSComEnv 环境配置参数 (通过 conf/task/mylossatenv/*.yaml 加载)"""
    
    # 任务名称 (必须)
    task: str = MISSING
    
    # --- 星座配置 ---
    n_satellites: int = MISSING              # 卫星数量
    n_planes: int = MISSING                   # 轨道面数量
    altitude: float = MISSING             # [km] 轨道高度
    inclination: float = MISSING           # [deg] 轨道倾角
    cluster_spacing: int = MISSING            # Walker delta F 参数
    
    # --- 任务场景配置 ---
    n_tasks: int = MISSING                 # 任务池大小
    n_select_from: int = MISSING             # 每步候选任务数
    data_size_min: float = MISSING          # [bits] 最小数据量 (1MB)
    data_size_max: float = MISSING         # [bits] 最大数据量 (5MB)
    data_size_mean: float = MISSING         # [bits] 可选：截断正态分布均值 (100 Mbits)
    data_size_std: float = MISSING    # [bits] 可选：截断正态分布标准差 (30 Mbits)
    workload_min: float = MISSING         # [cycles/bit] 最小计算负载
    workload_max: float = MISSING        # [cycles/bit] 最大计算负载
    max_delay_min: float = MISSING         # [s] 最小时延约束
    max_delay_max: float = MISSING         # [s] 最大时延约束
    task_arrival_rate: float = MISSING      # [tasks/s] 任务到达率
    priority_min: float = MISSING         # 任务优先级最小值 (影响奖励权重和调度顺序)
    priority_max: float = MISSING         # 任务优先级最大值

    # --- 奖励配置 ---
    base_reward: float = MISSING            # 基础完成奖励
    timeout_penalty: float = MISSING        # 超时惩罚
    delay_weight: float = MISSING           # 时延权重
    energy_weight: float = MISSING          # 能耗权重
    balance_weight: float = MISSING        # 负载均衡权重
    latency_penalty: float = MISSING      # 禁用二次时延惩罚（或设为 0.01）
    battery_penalty: float = MISSING       #  禁用电池安全惩罚（或设为 0.5）
    battery_safety_threshold: float = MISSING   # 降低阈值到 10%
    collaboration_weight: float = MISSING    # 协作奖励：卸载方获得任务完成奖励的 50%
    # --- 卫星硬件配置 ---
    cpu_max_frequency: float = MISSING   # [Hz] CPU 最大频率
    cpu_min_frequency: float = MISSING   # [Hz] CPU 最小频率
    cpu_workload: float = MISSING        # [cycles/bit] CPU 工作负载
    cpu_power_draw: float = MISSING       # [W] CPU 功耗
    transmitter_baud_rate: float = MISSING  # [baud] 传输速率
    transmitter_power_draw: float = MISSING  # [W] 发射机功耗
    base_power_draw: float = MISSING      # [W] 基础功耗
    battery_capacity: float = MISSING  # [J] 电池容量 (500Wh)   
    initial_charge: float = MISSING    # [J] 初始电量 (50%)
    panel_area: float = MISSING             # [m^2] 太阳能板面积
    task_min_elevation: float = MISSING  # [rad] 最小仰角 (15度)
    data_storage_capacity: float = MISSING # [bits] 数据存储容量 (1TB)
    sat_mass: float = MISSING                   # [kg] 卫星质量 (Iridium 类)
    sat_width: float = MISSING                  # [m] 卫星宽度
    sat_depth: float = MISSING                  # [m] 卫星深度
    sat_height: float = MISSING                 # [m] 卫星高度
    rw_u_max: float = MISSING                   # [N*m] 反作用轮最大扭矩
    max_wheel_speed: float = MISSING            # [RPM] 反作用轮最大转速
    
    # --- 仿真配置 ---
    max_step_duration: float = MISSING    # [s] 最大步长
    time_limit: float = MISSING          # [s] episode 时长
    max_steps: int = MISSING               # 最大步数
    log_level: str = MISSING            # 日志级别

    # --- MRP Steering 姿态控制参数 ---
    K1: float = MISSING          # MRP Steering 增益
    K3: float = MISSING          # MRP Steering 增益
    omega_max: float = MISSING   # [rad/s] 最大角速率
    servo_Ki: float = MISSING    # Servo 积分增益
    servo_P: float = MISSING     # Servo 比例增益
