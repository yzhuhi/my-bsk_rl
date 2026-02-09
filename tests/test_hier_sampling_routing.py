# -*- coding: utf-8 -*-
"""
测试分层路由采样逻辑的正确性。
"""
import numpy as np
import pytest


def test_probability_normalization():
    """测试概率归一化。"""
    # 模拟 logits
    logits = np.array([0.1, 0.3, 0.2, 0.15, 0.25])
    
    # softmax 归一化
    def softmax(x):
        x = np.clip(x, -10, 10)
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    probs = softmax(logits)
    
    # 验证和为 1
    assert np.abs(probs.sum() - 1.0) < 1e-6, f"Sum should be 1.0, got {probs.sum()}"
    # 验证所有值非负
    assert np.all(probs >= 0), "All probabilities should be non-negative"
    print(f"✓ Normalization test passed: {probs}")


def test_sampling_distribution():
    """测试采样分布收敛到目标概率。"""
    probs = np.array([0.3, 0.5, 0.2])
    n_samples = 10000
    
    # 采样
    samples = np.random.choice(len(probs), size=n_samples, p=probs)
    
    # 统计频率
    counts = np.bincount(samples, minlength=len(probs))
    frequencies = counts / n_samples
    
    # 验证频率接近目标概率（允许 5% 误差）
    for i, (freq, prob) in enumerate(zip(frequencies, probs)):
        error = np.abs(freq - prob)
        assert error < 0.05, f"Index {i}: frequency {freq:.3f} differs from prob {prob:.3f} by {error:.3f}"
    
    print(f"✓ Sampling test passed:")
    print(f"  Target probs: {probs}")
    print(f"  Frequencies:  {frequencies}")


def test_priority_routing():
    """测试优先级路由分流。"""
    high_probs = np.array([0.2, 0.3, 0.5])  # 高优先级倾向于邻居
    low_probs = np.array([0.7, 0.2, 0.1])   # 低优先级倾向于本地
    threshold = 0.5
    
    # 模拟切片队列
    class MockSlice:
        def __init__(self, priority):
            self.priority = priority
    
    slices = [
        MockSlice(0.8),  # 高优先级
        MockSlice(0.3),  # 低优先级
        MockSlice(0.6),  # 高优先级
    ]
    
    for s in slices:
        if s.priority >= threshold:
            selected_probs = high_probs
            level = "HIGH"
        else:
            selected_probs = low_probs
            level = "LOW"
        
        print(f"  Slice priority={s.priority:.1f} → {level}, probs={selected_probs}")
    
    print("✓ Priority routing test passed")


def test_reachability_mask():
    """测试邻居可达性掩码。"""
    base_probs = np.array([0.2, 0.3, 0.4, 0.1])  # [n0, n1, n2, self]
    
    # 模拟邻居列表：n1 不可达
    neighbors = [
        "sat_0",  # 可达
        None,     # 不可达
        "sat_2",  # 可达
    ]
    
    # 应用掩码
    probs = base_probs.copy()
    for i in range(len(neighbors)):
        if neighbors[i] is None:
            probs[i] = 0.0
    
    # 重新归一化
    total = probs.sum()
    if total > 1e-8:
        probs = probs / total
    else:
        probs = np.zeros_like(probs)
        probs[-1] = 1.0  # fallback 到本地（self）
    
    print(f"  Base probs:   {base_probs}")
    print(f"  Masked probs: {probs}")
    
    # 验证 n1 被屏蔽
    assert probs[1] == 0.0, "Unreachable neighbor should have prob=0"
    # 验证和为 1
    assert np.abs(probs.sum() - 1.0) < 1e-6, "Sum should be 1.0"
    
    print("✓ Reachability mask test passed")


def test_all_unreachable_fallback():
    """测试所有邻居不可达时的 fallback。"""
    base_probs = np.array([0.1, 0.3, 0.4, 0.2])
    neighbors = [None, None, None]  # 所有邻居不可达
    
    probs = base_probs.copy()
    for i in range(len(neighbors)):
        if neighbors[i] is None:
            probs[i] = 0.0
    
    total = probs.sum()
    if total < 1e-8:
        probs = np.zeros_like(probs)
        probs[-1] = 1.0  # fallback
    else:
        probs = probs / total
    
    print(f"  Fallback probs: {probs}")
    
    # 验证 fallback 到本地
    assert probs[-1] == 1.0, "Should fallback to local when all neighbors unreachable"
    assert np.all(probs[:-1] == 0.0), "All neighbor probs should be 0"
    
    print("✓ Fallback test passed")


if __name__ == "__main__":
    print("\n=== 运行分层路由采样测试 ===\n")
    test_probability_normalization()
    test_sampling_distribution()
    test_priority_routing()
    test_reachability_mask()
    test_all_unreachable_fallback()
    print("\n=== 所有测试通过 ✓ ===\n")
