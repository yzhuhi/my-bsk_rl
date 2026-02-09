# BSK-RL 新版模块优化实施计划

## 概述

对 `bsk_rl` 文件夹下的新版层级化模块（观测、动作、卫星）进行代码风格统一，使其与旧版写法对齐，同时保留各模块的独有功能。

---

## 核心问题分析

| 模块 | 新版文件                        | 旧版参照                        | 主要差异                           |
| ---- | ------------------------------- | ------------------------------- | ---------------------------------- |
| 观测 | `stin_hier_observations.py`     | `stin_relative_observations.py` | 新版硬编码特征列表，旧版支持可配置 |
| 动作 | `stin_hier_hybrid_actions.py`   | `stin_continuous_actions.py`    | 新版分层结构良好，文档风格不一致   |
| 卫星 | `hier_computation_satellite.py` | `computation_satellite.py`      | 新版继承正确，部分方法可优化       |

---

## 提议的修改

### 1. 观测模块 (`stin_hier_observations.py`)

#### [MODIFY] [stin_hier_observations.py](file:///e:/py_project/bsk_rl/src/bsk_rl/obs/stin_hier_observations.py)

**问题**：当前使用硬编码特征列表：
```python
self.self_feature_names = [
    "raw_queue_size",
    "task_queue_size",
    ...
]
```

**方案**：改为可配置风格，与 `STINRelativeObservations` 对齐：
```python
def __init__(
    self, 
    *self_properties: dict[str, Any],
    *neighbor_properties: dict[str, Any],
    max_neighbors: int = 4,
    ...
)
```

**具体修改**：
1. 重构 `__init__` 支持 `*self_properties` 和 `*neighbor_properties` 可变参数
2. 添加 `_process_property_spec` 方法处理属性规范
3. 统一辅助函数签名为 `get_xxx(sat, self_sat)` 双参数形式
4. 提供默认配置以保持向后兼容

**保留独有功能**：
- `neighbor_mask` 邻居可达性掩码
- 自身状态特征（8D）

---

### 2. 动作模块 (`stin_hier_hybrid_actions.py`)

#### [MODIFY] [stin_hier_hybrid_actions.py](file:///e:/py_project/bsk_rl/src/bsk_rl/act/stin_hier_hybrid_actions.py)

**问题**：文档风格与旧版不一致（旧版使用 `REQUIRED_ACTION_DIMS` 类属性）

**方案**：
1. 添加 `REQUIRED_ACTION_DIMS` 类属性计算公式注释
2. 统一 `action_description` 返回 `List[str]` 格式（旧版风格）
3. 完善方法文档字符串的中英文一致性

**具体修改**：
```python
class STINHierarchicalHybridAction(Action):
    # 动作维度: 2K + 10（K=4 时为 18D，L2 含 self）
    REQUIRED_ACTION_DIMS_FORMULA = "2K + 10"  # K = max_neighbors
    
    @property
    def action_description(self) -> List[str]:  # 改为 List[str]
        ...
```

**保留独有功能**：
- 分层 `space` / `flat_space` 结构
- `_parse_action` 双格式解析

---

### 3. 卫星模块 (`hier_computation_satellite.py`)

#### [MODIFY] [hier_computation_satellite.py](file:///e:/py_project/bsk_rl/src/bsk_rl/sats/hier_computation_satellite.py)

**当前状态**：分层方法 `schedule_collaboration_action_hier` 和 `process_slice_queue_hier` 已正确定义在子类中。

**优化方向**：
1. 确认父类 `computation_satellite.py` 也有这些方法的占位定义（测试文件检查失败说明需要修复）
2. 统一日志格式
3. 补充完整的方法文档字符串

**发现的问题**：
测试文件 `test_hier_computation_satellite_v2.py` 第 130-135 行断言这些方法在 `ComputationSatellite` 中存在，但它们实际只定义在 `HierComputationSatellite` 中。这可能导致测试失败。

**建议修改**：
- 在 `ComputationSatellite` 中添加抽象/占位方法，或
- 修复测试文件中的断言

---

## 验证计划

### 自动化测试

```powershell
# 运行现有分层卫星测试
cd e:\py_project\bsk_rl
python -m pytest tests/test_hier_computation_satellite_v2.py -v

# 运行采样路由测试
python -m pytest tests/test_hier_sampling_routing.py -v

# 运行观测模块单元测试
python -m pytest tests/unittest/obs/test_observations.py -v
```

### 导入兼容性验证

```powershell
cd e:\py_project\bsk_rl
python -c "from bsk_rl.obs import STINHierarchicalObservations; print('OK')"
python -c "from bsk_rl.act import STINHierarchicalHybridAction; print('OK')"
python -c "from bsk_rl.sats import HierComputationSatellite; print('OK')"
```

### 功能验证（向后兼容）

修改后的模块应支持以下用法：

```python
# 观测模块 - 新的可配置风格
obs.STINHierarchicalObservations(
    dict(prop="task_queue_size", norm=50.0),
    dict(prop="battery_fraction"),
    max_neighbors=4,
)

# 观测模块 - 向后兼容的默认配置
obs.STINHierarchicalObservations()  # 使用默认特征
```

---

## 风险评估

> [!IMPORTANT]
> 观测模块的重构涉及接口变更，需确保向后兼容。建议保留默认参数使现有代码无需修改。

> [!NOTE]
> 测试文件中的断言可能需要同步更新，特别是关于方法位置的测试。

---

## 待确认事项

1. **观测模块兼容性**：新的可配置风格是否需要完全向后兼容？还是可以接受轻微 API 变更？

2. **辅助函数签名**：统一为双参数 `get_xxx(sat, self_sat)` 是否会影响现有使用？

3. **测试修复**：`test_hier_computation_satellite_v2.py` 中的断言是否需要修改？

---

## 预计工作量

| 任务         | 预计时间 |
| ------------ | -------- |
| 观测模块重构 | 中等     |
| 动作模块优化 | 轻度     |
| 卫星模块优化 | 轻度     |
| 测试验证     | 轻度     |
