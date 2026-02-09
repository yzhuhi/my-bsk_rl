# -*- coding: utf-8 -*-
"""
测试新版双头逻辑的统计信息兼容性。

验证：
1. 动作空间为 Box 类型（兼容 BenchMARL/TorchRL）
2. 观测空间为 Box 类型（兼容 BenchMARL/TorchRL）
3. 维度计算正确
4. action_description 为 List[str] 格式
5. 扁平动作能正确解析
"""

import numpy as np
import pytest
from gymnasium import spaces


def test_action_space_compatibility():
    """测试动作空间兼容性。"""
    from bsk_rl.act.stin_hier_hybrid_actions import (
        STINHierarchicalHybridAction,
        HierarchicalHybridActionBuilder,
    )
    
    K = 4  # max_neighbors
    act = STINHierarchicalHybridAction(max_neighbors=K)
    
    # 1. flat_space 必须是 Box 类型（gym.py 和 BenchMARL 要求）
    assert isinstance(act.flat_space, spaces.Box), \
        f"flat_space should be Box, got {type(act.flat_space)}"
    
    # 2. 维度正确: K + 7
    expected_dim = K + 7
    assert act.flat_space.shape[0] == expected_dim, \
        f"Expected {expected_dim}, got {act.flat_space.shape[0]}"
    
    # 3. 分层空间是 Dict 类型（可选使用）
    assert isinstance(act.space, spaces.Dict), \
        f"hierarchical space should be Dict, got {type(act.space)}"
    
    # 4. REQUIRED_ACTION_DIMS_FORMULA 存在
    assert hasattr(act, 'REQUIRED_ACTION_DIMS_FORMULA')
    assert act.REQUIRED_ACTION_DIMS_FORMULA == "K + 7"
    
    print("✓ Action space compatibility test passed")


def test_observation_space_compatibility():
    """测试观测空间兼容性。
    
    注意：STINHierarchicalObservations 现在只包含邻居特征和掩码，
    不包含自身特征（由 SatProperties 单独处理）。
    
    维度公式：N * neighbor_features + N (mask) = 7N + N = 8N (默认 7 个特征)
    """
    from bsk_rl.obs.stin_hier_observations import STINHierarchicalObservations
    
    K = 4  # max_neighbors
    
    # 1. 测试默认配置（7 个邻居特征 + mask）
    obs_default = STINHierarchicalObservations(max_neighbors=K)
    
    # observation_space 必须是 Box 类型
    assert isinstance(obs_default.observation_space, spaces.Box), \
        f"observation_space should be Box, got {type(obs_default.observation_space)}"
    
    # 默认配置: K*7 (neighbor, 默认7个特征) + K (mask) = 8K = 32 (K=4)
    default_expected_dim = K * 7 + K  # 28 + 4 = 32
    assert obs_default.observation_space.shape[0] == default_expected_dim, \
        f"Default: Expected {default_expected_dim}, got {obs_default.observation_space.shape[0]}"
    
    # 2. 测试自定义邻居特征配置
    obs_custom = STINHierarchicalObservations(
        neighbor_properties=[
            dict(prop="get_task_queue_size"),
            dict(prop="get_battery_fraction"),
            dict(prop="get_isl_distance"),
        ],
        max_neighbors=K,
    )
    
    # 自定义配置: K*3 (neighbor) + K (mask) = 4K = 16 (K=4)
    custom_expected_dim = K * 3 + K  # 12 + 4 = 16
    assert obs_custom.observation_space.shape[0] == custom_expected_dim, \
        f"Custom: Expected {custom_expected_dim}, got {obs_custom.observation_space.shape[0]}"
    
    # 3. 内部维度检查（不再有 self_dim）
    assert obs_default.neighbor_features == 7, f"neighbor_features should be 7, got {obs_default.neighbor_features}"
    assert obs_default.mask_dim == K, f"mask_dim should be {K}, got {obs_default.mask_dim}"
    assert obs_default.total_dim == default_expected_dim, \
        f"total_dim should be {default_expected_dim}, got {obs_default.total_dim}"
    
    print("✓ Observation space compatibility test passed")


def test_action_description_format():
    """测试 action_description 格式（训练日志需要）。"""
    from bsk_rl.act.stin_hier_hybrid_actions import STINHierarchicalHybridAction
    
    K = 4
    act = STINHierarchicalHybridAction(max_neighbors=K)
    
    # 1. action_description 必须是 List[str]（旧版风格）
    desc = act.action_description
    assert isinstance(desc, list), f"Should be list, got {type(desc)}"
    assert all(isinstance(d, str) for d in desc), "All elements should be strings"
    
    # 2. 长度与动作维度匹配
    expected_len = K + 7
    assert len(desc) == expected_len, \
        f"Length mismatch: expected {expected_len}, got {len(desc)}"
    
    # 3. 也提供 Dict 格式（可选）
    dict_desc = act.action_description_dict
    assert isinstance(dict_desc, dict)
    assert 'level1' in dict_desc
    assert 'level2' in dict_desc
    
    print("✓ Action description format test passed")


def test_flat_action_parsing():
    """测试扁平动作解析。"""
    from bsk_rl.act.stin_hier_hybrid_actions import STINHierarchicalHybridAction
    
    K = 4
    act = STINHierarchicalHybridAction(max_neighbors=K)
    
    # 创建扁平动作
    total_dim = K + 7
    flat_action = np.random.rand(total_dim).astype(np.float32)
    
    # 解析
    parsed = act._parse_action(flat_action)
    
    # 验证结构
    assert 'level1' in parsed
    assert 'level2' in parsed
    assert 'continuous' in parsed['level2']
    
    # 验证维度
    level1_dim = 4
    continuous_dim = 2 + (K + 1)
    
    assert parsed['level1'].shape[0] == level1_dim, \
        f"Level1: expected {level1_dim}, got {parsed['level1'].shape[0]}"
    assert parsed['level2']['continuous'].shape[0] == continuous_dim, \
        f"Continuous: expected {continuous_dim}, got {parsed['level2']['continuous'].shape[0]}"
    
    print("✓ Flat action parsing test passed")


def test_configurable_observation():
    """测试可配置观测的向后兼容性。
    
    STINHierarchicalObservations 只包含邻居特征和掩码，
    自身特征由 SatProperties 单独配置。
    """
    from bsk_rl.obs.stin_hier_observations import (
        STINHierarchicalObservations,
        DEFAULT_NEIGHBOR_PROPERTIES,
    )
    
    # 1. 默认配置（只有邻居特征）
    obs_default = STINHierarchicalObservations()
    assert obs_default.neighbor_features == len(DEFAULT_NEIGHBOR_PROPERTIES)
    
    # 2. 自定义配置
    custom_neighbor = [dict(prop="get_isl_distance", norm=1e7)]
    obs_custom = STINHierarchicalObservations(
        neighbor_properties=custom_neighbor,
        max_neighbors=2,
    )
    assert obs_custom.neighbor_features == 1
    assert obs_custom.total_dim == 2*1 + 2  # neighbors + mask = 4
    
    print("✓ Configurable observation test passed")


def test_dimension_summary():
    """输出维度汇总，便于对照检查。
    
    STINHierarchicalObservations 只包含邻居特征和掩码。
    自身特征由 HierComputationSatellite.observation_spec 中的 SatProperties 处理。
    """
    from bsk_rl.act.stin_hier_hybrid_actions import STINHierarchicalHybridAction
    from bsk_rl.obs.stin_hier_observations import STINHierarchicalObservations
    
    print("\n" + "=" * 50)
    print("维度汇总 (K = max_neighbors = 4)")
    print("=" * 50)
    
    K = 4
    act = STINHierarchicalHybridAction(max_neighbors=K)
    obs = STINHierarchicalObservations(max_neighbors=K)
    
    print(f"\n动作空间 (K + 7 = {K + 7}):")
    print("  - Level-1 任务切分: 4 维")
    print(f"  - Level-2 资源+路由: {2 + (K + 1)} 维")
    print(f"  - 总计: {act.flat_space.shape[0]} 维")
    
    print(f"\nSTINHierarchicalObservations (邻居 + 掩码):")
    print(f"  - 邻居特征: {K} × {obs.neighbor_features} = {K * obs.neighbor_features} 维")
    print(f"  - 邻居掩码: {obs.mask_dim} 维")
    print(f"  - 总计: {obs.total_dim} 维")
    
    print(f"\n完整观测空间 (SatProperties + STINHierarchicalObservations):")
    print(f"  - 自身状态: ~22 维 (由 SatProperties 定义)")
    print(f"  - 邻居状态: {obs.total_dim} 维")
    print(f"  - Time + Eclipse: 2 维")
    
    print("\n" + "=" * 50)


def test_hier_computation_satellite_observation_spec():
    """测试 HierComputationSatellite 是否使用正确的观测空间风格。
    
    设计说明：HierComputationSatellite 使用分层观测风格：
    - SatProperties: 自身状态（电池、队列、CPU 等）
    - STINHierarchicalObservations: 邻居状态 + 可达性掩码
    - Time + Eclipse: 时间信息
    
    STINHierarchicalObservations 特点：
    - 只包含邻居特征和掩码（不含自身特征，由 SatProperties 处理）
    - 维度: N * neighbor_features + N (mask) = 7N (默认 K=4 → 28D)
    """
    from bsk_rl.sats.hier_computation_satellite import HierComputationSatellite
    from bsk_rl.obs import SatProperties, Time, Eclipse
    from bsk_rl.obs.stin_hier_observations import STINHierarchicalObservations
    
    # 检查类属性中的观测空间类型
    obs_spec = HierComputationSatellite.observation_spec
    
    # 1. 检查观测组件数量和类型
    assert len(obs_spec) == 6, f"Expected 6 observation components, got {len(obs_spec)}"
    
    # 2. 前三个应该是 SatProperties（自身状态分组）
    for i in range(3):
        assert isinstance(obs_spec[i], SatProperties), \
            f"obs_spec[{i}] should be SatProperties, got {type(obs_spec[i]).__name__}"
    
    # 3. 第四个应该是 STINHierarchicalObservations（邻居状态 + 掩码）
    assert isinstance(obs_spec[3], STINHierarchicalObservations), \
        f"obs_spec[3] should be STINHierarchicalObservations, got {type(obs_spec[3]).__name__}"
    
    # 4. 最后两个应该是 Time 和 Eclipse
    assert isinstance(obs_spec[4], Time), \
        f"obs_spec[4] should be Time, got {type(obs_spec[4]).__name__}"
    assert isinstance(obs_spec[5], Eclipse), \
        f"obs_spec[5] should be Eclipse, got {type(obs_spec[5]).__name__}"
    
    # 5. 检查 STINHierarchicalObservations 的配置
    hier_obs = obs_spec[3]
    assert hier_obs.max_neighbors == 4, f"max_neighbors should be 4, got {hier_obs.max_neighbors}"
    assert hier_obs.neighbor_features == 7, f"neighbor_features should be 7, got {hier_obs.neighbor_features}"
    # 总维度: 4*7 (neighbor) + 4 (mask) = 32
    expected_total = 4 * 7 + 4  # = 32
    assert hier_obs.total_dim == expected_total, \
        f"total_dim should be {expected_total}, got {hier_obs.total_dim}"
    
    # 6. 检查 SatProperties 的命名组
    assert obs_spec[0].name == "sat_state", f"First SatProperties should be 'sat_state'"
    assert obs_spec[1].name == "resource_state", f"Second SatProperties should be 'resource_state'"
    assert obs_spec[2].name == "queue_stats", f"Third SatProperties should be 'queue_stats'"
    
    print("✓ HierComputationSatellite observation_spec test passed")
    print("  - 使用分层观测风格（SatProperties + STINHierarchicalObservations）")
    print("  - SatProperties: sat_state, resource_state, queue_stats")
    print(f"  - STINHierarchicalObservations: {hier_obs.neighbor_features} 特征 × {hier_obs.max_neighbors} 邻居 + {hier_obs.max_neighbors} 掩码")
    print("  - Time + Eclipse: 时间信息")


if __name__ == "__main__":
    print("\n=== 新版双头逻辑统计信息兼容性测试 ===\n")
    test_action_space_compatibility()
    test_observation_space_compatibility()
    test_action_description_format()
    test_flat_action_parsing()
    test_configurable_observation()
    test_hier_computation_satellite_observation_spec()
    test_dimension_summary()
    print("\n=== 所有测试通过 ✓ ===\n")
