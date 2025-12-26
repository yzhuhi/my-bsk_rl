"""
bsk_rl.utils.constants: 物理常量和系统默认值定义

此模块集中管理 STIN 仿真中使用的物理常量和默认参数，
避免在代码中硬编码，便于维护和配置。
"""

import numpy as np

# =============================================================================
# 物理常量
# =============================================================================

# 光速 [m/s]
SPEED_OF_LIGHT = 299792458.0

# 有效开关电容系数 [J/cycle²] - 用于计算 CPU 能耗
# E = KAPPA * f² * cycles
KAPPA = 1e-24

# =============================================================================
# 通信链路默认参数
# =============================================================================
# 注意: 在动态速率模式下，ISL 和 SGL 速率由 calculate_isl_rate() 和
#       calculate_sgl_rate() 函数动态计算。以下固定值仅用于:
#       1. Uplink (UD → 卫星): 固定速率，UD 发射功率不受 Agent 控制
#       2. 向后兼容或备用模式

# 上行链路默认速率 [bps] - 固定值（UD 功率不受 Agent 控制）
DEFAULT_UPLINK_RATE = 50e6  # 50 Mbps

# ISL（星间链路）默认速率 [bps] - 已被动态模型替代
DEFAULT_ISL_RATE = 100e6  # 100 Mbps (备用)

# SGL（卫星-地面链路）默认速率 [bps] - 已被动态模型替代
DEFAULT_SGL_RATE = 50e6  # 50 Mbps (备用)

# =============================================================================
# 云端路径默认参数
# =============================================================================

# 网关到云服务器的光纤距离 [m]
DEFAULT_FIBER_DISTANCE = 500e3  # 500 km

# 云端 CPU 频率 [Hz]
DEFAULT_CLOUD_CPU_FREQ = 10e9  # 10 GHz

# =============================================================================
# 用户终端 (UD) 默认参数
# =============================================================================

# UD 本地 CPU 频率 [Hz]
DEFAULT_UD_CPU_FREQ = 1e9  # 1 GHz

# =============================================================================
# 任务处理默认参数
# =============================================================================

# 结果数据量与输入数据量的比例
DEFAULT_RESULT_RATIO = 0.1  # 结果数据 = 10% 输入数据

# =============================================================================
# 警告阈值
# =============================================================================

# 电池安全阈值 [SOC, 0-1]
BATTERY_SAFETY_THRESHOLD = 0.2  # 20%

# 电池严重警告阈值 [SOC, 0-1]
BATTERY_CRITICAL_THRESHOLD = 0.1  # 10%

# =============================================================================
# 动态速率模型参数 (基于真实 LEO 卫星系统物理参数)
# =============================================================================
# 参考: Iridium NEXT, ITU Ka-band ISL 标准, 典型 LEO 通信系统

# --- ISL 链路参数 (公式 5, 6) ---
# 基于 Iridium NEXT Ka-band ISL 和 ITU 标准
ISL_BANDWIDTH = 100e6           # [Hz] B_I - ISL 带宽 (100 MHz, Ka-band 典型值)
ISL_CARRIER_FREQ = 23e9         # [Hz] f_c - Ka-band 载波频率 (Iridium: 22.55-23.55 GHz)
ISL_TX_ANTENNA_GAIN = 1000.0    # g_I^tr - 发射天线增益 (30 dBi, 中等定向天线)
ISL_RX_ANTENNA_GAIN = 1000.0    # g_I^rc - 接收天线增益 (30 dBi)

# --- SGL 链路参数 (公式 10) ---
# 基于典型 LEO 星地链路
SGL_BANDWIDTH = 50e6            # [Hz] B_S - SGL 带宽 (50 MHz)
SGL_CHANNEL_GAIN = 500.0        # h_S - 等效信道增益 (含下行路径损耗补偿)

# --- 噪声参数 ---
NOISE_POWER_DENSITY = 4e-21     # [W/Hz] N₀ - 噪声功率谱密度 (典型 290K 系统温度)

# --- ISL 最大通信距离 ---
ISL_MAX_DISTANCE = 5000e3       # [m] Dis_I^max - 5000 km (Iridium 跨面链路)


def calculate_isl_path_loss(distance: float, carrier_freq: float = ISL_CARRIER_FREQ) -> float:
    """计算 ISL 路径损耗 - 论文公式 (5)
    
    L_I = (4π × Dis_I × f_c / c)²
    
    Args:
        distance: [m] 星间距离 Dis_I
        carrier_freq: [Hz] 载波频率 f_c
        
    Returns:
        路径损耗 L_I (无量纲)
    """
    if distance <= 0:
        return float('inf')
    return (4 * np.pi * distance * carrier_freq / SPEED_OF_LIGHT) ** 2


def calculate_isl_rate(tx_power: float, distance: float) -> float:
    """计算 ISL 传输速率 - 论文公式 (6)
    
    R_I = B_I × log₂(1 + (p_I × g_I^tr × g_I^rc) / (L_I × N₀ × B_I))
    
    Args:
        tx_power: [W] 发射功率 p_I (Agent 可控)
        distance: [m] 星间距离
        
    Returns:
        传输速率 [bps]，限制在 [1 kbps, 1 Gbps]
    """
    if tx_power <= 0 or distance <= 0:
        return 1e3  # 最低 1 kbps
    
    # 检查是否超过最大通信距离
    if distance > ISL_MAX_DISTANCE:
        return 1e3  # 超出范围，返回最低速率
    
    path_loss = calculate_isl_path_loss(distance)
    
    # SNR = (p_I × g_I^tr × g_I^rc) / (L_I × N₀ × B_I)
    numerator = tx_power * ISL_TX_ANTENNA_GAIN * ISL_RX_ANTENNA_GAIN
    denominator = path_loss * NOISE_POWER_DENSITY * ISL_BANDWIDTH
    
    if denominator <= 0:
        return 1e3
    
    snr = numerator / denominator
    
    # R_I = B_I × log₂(1 + SNR)
    rate = ISL_BANDWIDTH * np.log2(1 + snr)
    
    return float(np.clip(rate, 1e3, 1e9))  # 限制在 1 kbps - 1 Gbps


def calculate_sgl_rate(tx_power: float) -> float:
    """计算 SGL 传输速率 - 论文公式 (10)
    
    R_S = B_S × log₂(1 + (p_S × h_S) / (N₀ × B_S))
    
    Args:
        tx_power: [W] 发射功率 p_S (Agent 可控)
        
    Returns:
        传输速率 [bps]，限制在 [1 kbps, 1 Gbps]
    """
    if tx_power <= 0:
        return 1e3  # 最低 1 kbps
    
    # SNR = (p_S × h_S) / (N₀ × B_S)
    snr = (tx_power * SGL_CHANNEL_GAIN) / (NOISE_POWER_DENSITY * SGL_BANDWIDTH)
    
    # R_S = B_S × log₂(1 + SNR)
    rate = SGL_BANDWIDTH * np.log2(1 + snr)
    
    return float(np.clip(rate, 1e3, 1e9))  # 限制在 1 kbps - 1 Gbps


__all__ = [
    # 物理常量
    "SPEED_OF_LIGHT",
    "KAPPA",
    # 通信链路 (固定速率)
    "DEFAULT_UPLINK_RATE",
    "DEFAULT_ISL_RATE",
    "DEFAULT_SGL_RATE",
    # 动态速率模型参数
    "ISL_BANDWIDTH",
    "ISL_CARRIER_FREQ",
    "ISL_TX_ANTENNA_GAIN",
    "ISL_RX_ANTENNA_GAIN",
    "ISL_MAX_DISTANCE",
    "SGL_BANDWIDTH",
    "SGL_CHANNEL_GAIN",
    "NOISE_POWER_DENSITY",
    # 动态速率计算函数
    "calculate_isl_path_loss",
    "calculate_isl_rate",
    "calculate_sgl_rate",
    # 云端路径
    "DEFAULT_FIBER_DISTANCE",
    "DEFAULT_CLOUD_CPU_FREQ",
    # 用户终端
    "DEFAULT_UD_CPU_FREQ",
    # 任务处理
    "DEFAULT_RESULT_RATIO",
    # 警告阈值
    "BATTERY_SAFETY_THRESHOLD",
    "BATTERY_CRITICAL_THRESHOLD",
]

