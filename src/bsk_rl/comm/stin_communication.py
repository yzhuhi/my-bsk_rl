"""STIN-specific communication implementations.

This module provides communication classes tailored for STIN (Satellite-Terrestrial
Integrated Network) scenarios where data synchronization is handled through ISL relay
rather than the base Data communication mechanism.
"""

from bsk_rl.comm.communication import LOSMultiCommunication


class STINLOSCommunication(LOSMultiCommunication):
    """Line-of-sight communication for STIN environments.
    
    与 LOSMultiCommunication 的区别：
    - **不同步 Data**：STIN 场景中，任务协作信息通过 ISL relay 机制传递，
      不需要通过 bsk_rl 的 Data 通信系统同步。
    - **保留可见性检测**：可用于判断卫星间 ISL 链路是否可用。
    
    使用场景：
        - 任务卸载协作（通过 ISL relay 实现，不走 Data）
        - 需要知道哪些卫星互相可见（用于决策）
    """

    def communicate(self) -> None:
        """Only update communication visibility, do not sync Data.
        
        STIN 场景中，每颗卫星的 processed_data/offloaded_data 是自己的统计，
        不需要与其他卫星同步。协作信息通过 ISL relay 机制传递。
        """
        # 只更新 LOS 日志（用于可见性判断），不执行数据同步
        # 父类的 communicate() 会同步 Data，这里跳过
        if hasattr(self, 'los_logs'):
            for sat_1, logs in self.los_logs.items():
                for sat_2, logger in logs.items():
                    # 清空日志以等待下一步的可见性检测
                    logger.clear()
        
        # 更新通信时间戳
        self.last_communication_time = self.satellites[0].simulator.sim_time


__all__ = ["STINLOSCommunication"]
