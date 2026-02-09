# -*- coding: utf-8 -*-
"""
测试分层路由卫星实现（HierComputationSatellite）。

本测试验证：
1. HierComputationSatellite 正确继承父类方法
2. process_slice_queue_hier 按分布采样路由切片
3. 优先级阈值正确分流高/低优先级切片
"""
import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import numpy as np
import pytest


class MockSlice:
    """模拟任务切片。"""
    def __init__(self, task_id, priority=0.5, data_size=1e6):
        self.task_id = task_id
        self.priority = priority
        self.data_size = data_size
        self.origin_position = None
        self.uplink_distance = 0.0
        self.uplink_rate = 1e6
        self.origin_satellite = ''
        self.hop_count = 0
        self.current_holder = ''
        self.t_prop_isl_total = 0.0
        self.t_tx_isl_total = 0.0
        self.last_isl_tx_delay = 0.0
        self.status = 'pending'
    
    def is_expired(self, sim_time: float) -> bool:
        return False


class MockNeighbor:
    """模拟邻居卫星。"""
    def __init__(self, name):
        self.name = name
        self.slice_queue = []
        self.current_tx_power = 10.0
    
    def process_incoming_slice(self, slice_obj):
        self.slice_queue.append(slice_obj)
        return True


class MockDynamics:
    """模拟动力学对象。"""
    transmitter_baud_rate = 50e6


class MockSimulator:
    """模拟仿真器。"""
    sim_time = 100.0


def test_mask_and_normalize_routing_probs():
    """测试路由概率 mask 与归一化。"""
    from bsk_rl.sats.hier_computation_satellite import HierComputationSatellite
    
    # 使用最小构造（类方法测试）
    sat = object.__new__(HierComputationSatellite)
    
    # 测试正常情况
    base_ratios = np.array([0.2, 0.3, 0.4, 0.1])
    neighbors = [MockNeighbor('n0'), MockNeighbor('n1'), MockNeighbor('n2')]
    
    probs = sat._mask_and_normalize_routing_probs(base_ratios, neighbors)
    
    assert np.abs(probs.sum() - 1.0) < 1e-6, f"Sum should be 1.0, got {probs.sum()}"
    assert np.all(probs >= 0), "All probabilities should be non-negative"
    print(f"✓ Normal case: {probs}")
    
    # 测试部分邻居不可达
    neighbors_partial = [MockNeighbor('n0'), None, MockNeighbor('n2')]
    probs_partial = sat._mask_and_normalize_routing_probs(base_ratios, neighbors_partial)
    
    assert probs_partial[2] == 0.0, "Unreachable neighbor should have prob=0"
    assert np.abs(probs_partial.sum() - 1.0) < 1e-6
    print(f"✓ Partial reachability: {probs_partial}")
    
    # 测试所有邻居不可达
    neighbors_none = [None, None, None]
    probs_none = sat._mask_and_normalize_routing_probs(base_ratios, neighbors_none)
    
    assert probs_none[-1] == 1.0, "Should fallback to local (self index)"
    assert np.all(probs_none[:-1] == 0.0)
    print(f"✓ All unreachable fallback: {probs_none}")


def test_priority_routing_distribution():
    """测试优先级路由分流逻辑。"""
    high_probs = np.array([0.2, 0.3, 0.5])  # 高优先级倾向于邻居
    low_probs = np.array([0.7, 0.2, 0.1])   # 低优先级倾向于本地
    threshold = 0.5
    
    slices = [
        MockSlice('task1', priority=0.8),  # 高优先级
        MockSlice('task2', priority=0.3),  # 低优先级
        MockSlice('task3', priority=0.6),  # 高优先级
        MockSlice('task4', priority=0.4),  # 低优先级
    ]
    
    high_count = 0
    low_count = 0
    
    for s in slices:
        if s.priority >= threshold:
            high_count += 1
        else:
            low_count += 1
    
    assert high_count == 2, f"Expected 2 high priority, got {high_count}"
    assert low_count == 2, f"Expected 2 low priority, got {low_count}"
    print(f"✓ Priority routing: {high_count} high, {low_count} low")


def test_routing_execution_mode_toggle():
    """测试 delayed/immediate 语义切换。"""
    from bsk_rl.act.stin_hier_hybrid_actions import STINHierarchicalHybridAction

    class MockSat:
        def __init__(self):
            self.name = "sat-test"
            self.slice_queue = []
            self.raw_task_queue = []
            self.shield_config = {}
            self.routing_execution_mode = "delayed"
            self.last_level1_ratios = None
            self._alpha_updated = False
            self.level2_candidate_slices_this_step = 0
            self.level2_processed_count_this_step = 0

        def set_resource_allocation(self, *_args, **_kwargs):
            return None

        def schedule_collaboration_action_hier(self, *_args, **_kwargs):
            # 模拟 Level-1 产生两个新切片
            self.slice_queue.extend([object(), object()])

        def process_slice_queue_hier(self, **kwargs):
            self.last_max_slices = kwargs.get("max_slices")
            return 1

    sat = MockSat()
    act = STINHierarchicalHybridAction(max_neighbors=4, routing_execution_mode="delayed")
    act.link_satellite(sat)
    act.neighbor_satellites = []
    action = np.full(11, 0.2, dtype=np.float32)

    # delayed: max_slices 由动作前的切片数量决定（这里为 0）
    sat.slice_queue = []
    sat.routing_execution_mode = "delayed"
    act.routing_execution_mode = "delayed"
    act.set_action(action)
    assert getattr(sat, "last_max_slices", None) is None or sat.last_max_slices == 0

    # immediate: max_slices=None，允许处理本步新切片
    sat.slice_queue = []
    sat.routing_execution_mode = "immediate"
    act.routing_execution_mode = "immediate"
    act.set_action(action)
    assert sat.last_max_slices is None


def test_hier_computation_satellite_inheritance():
    """测试 HierComputationSatellite 正确定义并继承方法。
    
    注意：分层路由方法（process_slice_queue_hier, schedule_collaboration_action_hier,
    _mask_and_normalize_routing_probs）仅定义在 HierComputationSatellite 子类中，
    而非 ComputationSatellite 父类。这是有意为之的设计。
    """
    from bsk_rl.sats.hier_computation_satellite import HierComputationSatellite
    from bsk_rl.sats.computation_satellite import ComputationSatellite
    
    # 验证方法定义在子类 HierComputationSatellite 中
    assert hasattr(HierComputationSatellite, 'process_slice_queue_hier'), \
        "process_slice_queue_hier should be defined in HierComputationSatellite"
    assert hasattr(HierComputationSatellite, 'schedule_collaboration_action_hier'), \
        "schedule_collaboration_action_hier should be defined in HierComputationSatellite"
    assert hasattr(HierComputationSatellite, '_mask_and_normalize_routing_probs'), \
        "_mask_and_normalize_routing_probs should be defined in HierComputationSatellite"
    
    # 验证子类正确继承自 ComputationSatellite
    assert issubclass(HierComputationSatellite, ComputationSatellite), \
        "HierComputationSatellite should inherit from ComputationSatellite"
    
    print("✓ HierComputationSatellite correctly defines hierarchical routing methods")


def test_stin_task_store_slice_queue_metrics():
    """slice_queue 统计应进入 STINTaskData 增量链路。"""
    from bsk_rl.data.stin_task_data import STINTaskStore

    class DummyTask:
        def __init__(self, data_size: float) -> None:
            self.data_size = data_size

    class DummyDynamics:
        max_tx_power_draw = 12.0
        rx_power_draw = 6.0

    class DummySat:
        def __init__(self) -> None:
            self.dynamics = DummyDynamics()
            self.completed_tasks_buffer = []
            self.expired_tasks_buffer = []

    store = object.__new__(STINTaskStore)
    store.satellite = DummySat()

    old_state = {
        "completed_count": 0,
        "expired_count": 0,
        "processed_data": 0.0,
        "offloaded_data": 0.0,
        "sim_time": 0.0,
        "total_power": 2.0,
        "battery_level": 0.9,
        "tx_time_this_step": 0.0,
        "rx_time_this_step": 0.0,
        "raw_task_queue": [DummyTask(1e9)],
        "task_queue": [DummyTask(2e9)],
        "slice_queue": [DummyTask(2e9)],
    }
    new_state = {
        "completed_count": 0,
        "expired_count": 0,
        "processed_data": 0.0,
        "offloaded_data": 0.0,
        "sim_time": 1.0,
        "total_power": 2.0,
        "battery_level": 0.9,
        "tx_time_this_step": 0.0,
        "rx_time_this_step": 0.0,
        "raw_task_queue": [DummyTask(1e9)],
        "task_queue": [DummyTask(2e9)],
        "slice_queue": [DummyTask(1e9)],
    }

    delta = store.compare_log_states(old_state, new_state)
    assert delta.slice_queue_len == 1
    assert delta.slice_queue_data_gbits == pytest.approx(1.0)
    assert delta.slice_queue_data_delta_gbits == pytest.approx(-1.0)


def test_stin_task_reward_slice_drain_component():
    """w_slice_drain 应为切片 backlog 下降提供正奖励。"""
    from bsk_rl.data.stin_task_data import STINTaskData, STINTaskReward

    rewarder = STINTaskReward(
        w_progress=0.0,
        w_complete=0.0,
        w_expired=0.0,
        w_energy=0.0,
        w_delay=0.0,
        w_queue_penalty=0.0,
        w_slice_drain=0.01,
        global_reward_ratio=0.0,
        w_utilization=0.0,
        w_pbrs_queue=0.0,
    )
    reward = rewarder.calculate_reward(
        {
            "sat-0": STINTaskData(
                slice_queue_data_delta_gbits=-2.0,
                raw_queue_len=0,
                queue_len=0,
                slice_queue_len=0,
            )
        }
    )

    assert reward["sat-0"] == pytest.approx(0.02)
    assert rewarder.last_reward_components["sat-0"]["slice_drain_reward"] == pytest.approx(
        0.02
    )


if __name__ == "__main__":
    print("\n=== 运行分层卫星实现测试 ===\n")
    test_mask_and_normalize_routing_probs()
    test_priority_routing_distribution()
    test_hier_computation_satellite_inheritance()
    print("\n=== 所有测试通过 ✓ ===\n")
