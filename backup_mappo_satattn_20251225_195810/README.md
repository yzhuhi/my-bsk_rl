# 备份说明 - mappo_task_1_satattn__2165b1e3_25_12_25-19_58_10

## 备份时间
2025-12-26 17:30

## 备份内容
此备份包含 2025-12-25 19:58 训练成功后的代码版本。
训练运行名称: `mappo_task_1_satattn__2165b1e3_25_12_25-19_58_10`

## 文件列表

### bsk_rl 核心文件
- `stin_task_data.py` - 任务数据和奖励计算
- `gym.py` - 环境主逻辑
- `computation_satellite.py` - 计算卫星类
- `stin_scenario.py` - 任务场景
- `stin_relative_observations.py` - 相对观察模块

### BenchMARL 配置文件
- `benchmarl/common.py` - 环境封装
- `benchmarl/task_1.yaml` - 任务配置
- `benchmarl/task_1.py` - 任务数据类

## 2025-12-26 修改内容 (今天的优化)

以下是今天相对于此备份所做的修改：

### 1. stin_task_data.py
- **83-88行**: 新增 `__add__` 方法中的 sum_energy NaN/Inf 安全检查
- **187-196行**: 新增 `compare_log_states()` 中的 total_power 和 energy_consumed 安全检查
- **248行**: 注释更新 (优先级已实现)
- **494行**: 注释修正 (MAX_BATTERY_PENALTY 15.0)
- **568行**: 电池截断阈值保持 0.00 (用户决策)

### 2. gym.py
- **454-475行**: 新增 `_get_info()` 中的 energy_val NaN/Inf 跳过逻辑
- **471-475行**: 新增 total_energy 最终安全检查

### 3. common.py (BenchMARL)
- **56-60行**: 新增 priority_fn 构建逻辑
- **72行**: 新增 priority_distribution=priority_fn 参数
- **199-212行**: 简化 log_info() 返回空字典 (避免冗余)

### 4. task_1.yaml (BenchMARL)
- n_tasks: 3000 → 1500
- n_select_from: 6000 → 3000
- 新增: data_size_mean, data_size_std (启用截断正态分布)
- 新增: priority_min, priority_max

### 5. task_1.py (BenchMARL dataclass)
- 新增: data_size_mean, data_size_std 字段
- 新增: priority_min, priority_max 字段

## 回滚方法

如需回滚到此备份版本：

```powershell
# bsk_rl
copy "backup_mappo_satattn_20251225_195810\stin_task_data.py" "src\bsk_rl\data\"
copy "backup_mappo_satattn_20251225_195810\gym.py" "src\bsk_rl\"
copy "backup_mappo_satattn_20251225_195810\computation_satellite.py" "src\bsk_rl\sats\"
copy "backup_mappo_satattn_20251225_195810\stin_scenario.py" "src\bsk_rl\scene\"
copy "backup_mappo_satattn_20251225_195810\stin_relative_observations.py" "src\bsk_rl\obs\"

# BenchMARL
copy "backup_mappo_satattn_20251225_195810\benchmarl\common.py" "e:\py_project\BenchMARL\benchmarl\environments\mylossatenv\"
copy "backup_mappo_satattn_20251225_195810\benchmarl\task_1.yaml" "e:\py_project\BenchMARL\benchmarl\conf\task\mylossatenv\"
copy "backup_mappo_satattn_20251225_195810\benchmarl\task_1.py" "e:\py_project\BenchMARL\benchmarl\environments\mylossatenv\"
```

## 注意
此备份中的文件是**当前最新版本**（包含今天的优化）。
如需获取昨晚训练时的精确版本，需要手动撤销上述修改列表中的变更。
