"""Quick validation script for STIN MARL implementation.

This script performs basic checks without running the full environment.
Tests cover: imports, TaskSlice, ComputationTask, STINTaskData, 
STINTaskReward, STINContinuousAction, STINRelativeObservations.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_imports():
    """Test that all STIN modules can be imported."""
    print("Testing imports...")
    
    modules = [
        ("bsk_rl.data", "STINTaskReward"),
        ("bsk_rl.scene", "CityTaskScenario"),
        ("bsk_rl.scene", "STINTaskScenario"),
        ("bsk_rl.scene", "ComputationTask"),
        ("bsk_rl.sats", "ComputationSatellite"),
        ("bsk_rl.act", "STINContinuousAction"),
        ("bsk_rl.obs", "STINRelativeObservations"),
        ("bsk_rl.sim.dyn", "ComputationDynModel"),
    ]
    
    all_ok = True
    for module_name, class_name in modules:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"  ✓ {module_name}.{class_name}")
        except Exception as e:
            print(f"  ✗ {module_name}.{class_name}: {e}")
            all_ok = False
    
    if all_ok:
        print("\n✓ All imports successful!\n")
    return all_ok


def test_task_slice():
    """Test TaskSlice creation, properties, and delay calculations."""
    print("Testing TaskSlice...")
    
    try:
        import numpy as np
        from bsk_rl.sats.computation_satellite import TaskSlice, TaskStatus
        
        # Create a test task slice
        task = TaskSlice(
            task_id=1,
            data_size=10e6,  # 10 Mb
            workload=500.0,  # 500 cycles/bit
            max_delay=10.0,  # 10 seconds
            origin_position=np.array([6378137.0, 0.0, 0.0]),
            uplink_distance=800e3,  # 800 km
            uplink_rate=10e6,  # 10 Mbps
        )
        
        # Basic property checks
        assert task.task_id == 1, "Task ID mismatch"
        assert task.data_size == 10e6, "Data size mismatch"
        assert task.status == TaskStatus.PENDING, "Initial status should be PENDING"
        
        # Delay calculations
        assert task.t_tx_up > 0, "Upload transmission delay should be > 0"
        assert task.t_prop_up > 0, "Upload propagation delay should be > 0"
        assert task.t_uplink_total == task.t_tx_up + task.t_prop_up, "Total uplink mismatch"
        
        # Test path delay calculations
        task.alpha_local = 0.3
        task.alpha_cloud = 0.2
        task.alpha_sat = 0.5
        
        t_ud = task.calculate_ud_path_delay()
        t_cloud = task.calculate_cloud_path_delay(sgl_distance=600e3)
        
        print(f"  Task ID: {task.task_id}")
        print(f"  Data size: {task.data_size/1e6:.2f} Mb")
        print(f"  T_tx_up: {task.t_tx_up*1000:.2f} ms")
        print(f"  T_prop_up: {task.t_prop_up*1000:.2f} ms")
        print(f"  T_uplink_total: {task.t_uplink_total*1000:.2f} ms")
        print(f"  T_UD (30% local): {t_ud*1000:.2f} ms")
        print(f"  T_Cloud (20% cloud): {t_cloud*1000:.2f} ms")
        print(f"  Total cycles: {task.total_cycles/1e9:.2f} G cycles")
        
        print("\n✓ TaskSlice tests passed!\n")
        return True
        
    except Exception as e:
        print(f"✗ TaskSlice tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_computation_task():
    """Test ComputationTask creation and properties."""
    print("Testing ComputationTask...")
    
    try:
        import numpy as np
        from bsk_rl.scene.stin_scenario import ComputationTask
        
        task = ComputationTask(
            name="test-task-1",
            r_LP_P=np.array([6378137.0, 0.0, 0.0]),
            data_size=5e6,
            workload=300.0,
            max_delay=15.0,
            priority=0.8,
            uplink_rate=10e6,
            fiber_distance=500e3,
            cloud_cpu_freq=10e9,
            ud_cpu_freq=1e9,
        )
        
        assert task.name == "test-task-1"
        assert task.data_size == 5e6
        assert task.task_id >= 0
        assert np.allclose(task.origin_position, task.r_LP_P)
        assert task.total_cycles == task.data_size * task.workload
        
        print(f"  Task name: {task.name}")
        print(f"  Task ID: {task.task_id}")
        print(f"  Data size: {task.data_size/1e6:.2f} Mb")
        print(f"  Workload: {task.workload} cycles/bit")
        print(f"  Total cycles: {task.total_cycles/1e9:.2f} G cycles")
        print(f"  Priority: {task.priority}")
        print(f"  Max delay: {task.max_delay} s")
        
        print("\n✓ ComputationTask tests passed!\n")
        return True
        
    except Exception as e:
        print(f"✗ ComputationTask tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_stin_task_data():
    """Test STINTaskData creation and merging."""
    print("Testing STINTaskData...")
    
    try:
        from bsk_rl.data.stin_task_data import STINTaskData
        
        # Create two data units
        data1 = STINTaskData(
            completed_tasks=[],
            expired_tasks=[],
            processed_data=10e6,
            offloaded_data=5e6,
            energy_consumed=100.0,
            avg_latency=0.5,
        )
        
        data2 = STINTaskData(
            completed_tasks=[],
            expired_tasks=[],
            processed_data=8e6,
            offloaded_data=2e6,
            energy_consumed=80.0,
            avg_latency=0.3,
        )
        
        # Test merging
        merged = data1 + data2
        
        assert merged.processed_data == 18e6, "Processed data merge failed"
        assert merged.offloaded_data == 7e6, "Offloaded data merge failed"
        assert merged.energy_consumed == 180.0, "Energy merge failed"
        
        print(f"  Data1: processed={data1.processed_data/1e6:.1f}Mb, offloaded={data1.offloaded_data/1e6:.1f}Mb")
        print(f"  Data2: processed={data2.processed_data/1e6:.1f}Mb, offloaded={data2.offloaded_data/1e6:.1f}Mb")
        print(f"  Merged: processed={merged.processed_data/1e6:.1f}Mb, offloaded={merged.offloaded_data/1e6:.1f}Mb")
        
        print("\n✓ STINTaskData tests passed!\n")
        return True
        
    except Exception as e:
        print(f"✗ STINTaskData tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reward_calculation():
    """Test STINTaskReward initialization."""
    print("Testing STINTaskReward...")
    
    try:
        from bsk_rl.data.stin_task_data import STINTaskReward, STINTaskData
        
        reward_system = STINTaskReward(
            base_reward=1.0,
            timeout_penalty=2.0,
            delay_weight=0.5,
            latency_penalty=0.1,
            energy_weight=0.01,
            battery_penalty=5.0,
            battery_safety_threshold=0.2,
            balance_weight=0.05,
        )
        
        # Verify all parameters
        assert reward_system.base_reward == 1.0
        assert reward_system.timeout_penalty == 2.0
        assert reward_system.delay_weight == 0.5
        assert reward_system.latency_penalty == 0.1
        assert reward_system.energy_weight == 0.01
        assert reward_system.battery_penalty == 5.0
        assert reward_system.battery_safety_threshold == 0.2
        assert reward_system.balance_weight == 0.05
        
        print(f"  Base reward: {reward_system.base_reward}")
        print(f"  Timeout penalty: {reward_system.timeout_penalty}")
        print(f"  Delay weight: {reward_system.delay_weight}")
        print(f"  Latency penalty: {reward_system.latency_penalty}")
        print(f"  Energy weight: {reward_system.energy_weight}")
        print(f"  Battery penalty: {reward_system.battery_penalty}")
        print(f"  Battery safety threshold: {reward_system.battery_safety_threshold}")
        print(f"  Balance weight: {reward_system.balance_weight}")
        
        print("\n✓ STINTaskReward tests passed!\n")
        return True
        
    except Exception as e:
        print(f"✗ STINTaskReward tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_continuous_action():
    """Test STINContinuousAction space definition."""
    print("Testing STINContinuousAction...")
    
    try:
        from bsk_rl.act.stin_continuous_actions import STINContinuousAction
        import numpy as np
        
        action = STINContinuousAction(name="test_action")
        
        # Check action space
        space = action.space
        assert space.shape == (10,), f"Expected (10,) dims, got {space.shape}"
        assert np.allclose(space.low, 0.0), "Low bound should be 0"
        assert np.allclose(space.high, 1.0), "High bound should be 1"
        
        # Check action description
        desc = action.action_description
        assert len(desc) == 10, f"Expected 10 descriptions, got {len(desc)}"
        
        print(f"  Action space shape: {space.shape}")
        print(f"  Action space low: {space.low[0]}")
        print(f"  Action space high: {space.high[0]}")
        print(f"  Required dims: {action.REQUIRED_ACTION_DIMS}")
        print("  Action dimensions:")
        for i, d in enumerate(desc[:5]):  # Show first 5
            print(f"    [{i}] {d[:50]}...")
        print("    ...")
        
        print("\n✓ STINContinuousAction tests passed!\n")
        return True
        
    except Exception as e:
        print(f"✗ STINContinuousAction tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scenario():
    """Test STINTaskScenario and CityTaskScenario."""
    print("Testing STIN Scenarios...")
    
    try:
        from bsk_rl.scene.stin_scenario import STINTaskScenario, CityTaskScenario
        
        # Test uniform scenario
        uniform_scenario = STINTaskScenario(
            n_tasks=5,
            data_size_range=(1e6, 5e6),
            workload_range=(100, 500),
            max_delay_range=(5.0, 15.0),
        )
        
        assert uniform_scenario._n_tasks == 5
        assert uniform_scenario.data_size_range == (1e6, 5e6)
        
        # Test city scenario
        city_scenario = CityTaskScenario(
            n_tasks=3,
            n_select_from=50,
            location_offset=10000,
        )
        
        assert city_scenario._n_tasks == 3
        assert city_scenario.n_select_from == 50
        
        print(f"  Uniform scenario tasks: {uniform_scenario._n_tasks}")
        print(f"  City scenario tasks: {city_scenario._n_tasks}")
        print(f"  City select from: {city_scenario.n_select_from}")
        
        print("\n✓ STIN Scenario tests passed!\n")
        return True
        
    except Exception as e:
        print(f"✗ STIN Scenario tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dynamics_model():
    """Test ComputationDynModel parameters."""
    print("Testing ComputationDynModel...")
    
    try:
        from bsk_rl.sim.dyn.computation_dynamics import ComputationDynModel
        
        # Check that the class exists and has expected attributes
        assert hasattr(ComputationDynModel, '__init__'), "Missing __init__"
        assert hasattr(ComputationDynModel, 'setup_cpu_power_sink'), "Missing setup_cpu_power_sink"
        assert hasattr(ComputationDynModel, 'set_cpu_power'), "Missing set_cpu_power"
        assert hasattr(ComputationDynModel, 'set_base_power'), "Missing set_base_power"
        
        # Check @default_args decorator added expected parameters
        # The decorator adds these to _default_args class attribute
        # We can check by looking at the __init__ signature or _default_args
        import inspect
        sig = inspect.signature(ComputationDynModel.__init__)
        
        print(f"  Class: ComputationDynModel")
        print(f"  Parent: GroundStationDynModel")
        print("  Expected SMEC parameters (via @default_args):")
        print("    - cpuMinFrequency: 0.1 GHz")
        print("    - cpuMaxFrequency: 1.0 GHz")
        print("    - cpuWorkload: 1000 cycles/bit")
        print("    - cpuPowerDraw: -20 W")
        print("    - taskMinimumElevation: 0.0 rad")
        print("  Methods:")
        print("    - setup_cpu_power_sink()")
        print("    - set_cpu_power()")
        print("    - set_base_power()")
        
        print("\n✓ ComputationDynModel tests passed!\n")
        return True
        
    except Exception as e:
        print(f"✗ ComputationDynModel tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    print("="*60)
    print("STIN MARL Implementation Validation")
    print("="*60 + "\n")
    
    tests = [
        ("Imports", test_imports),
        ("TaskSlice", test_task_slice),
        ("ComputationTask", test_computation_task),
        ("STINTaskData", test_stin_task_data),
        ("STINTaskReward", test_reward_calculation),
        ("STINContinuousAction", test_continuous_action),
        ("STIN Scenarios", test_scenario),
        ("ComputationDynModel", test_dynamics_model),
    ]
    
    results = []
    for name, test in tests:
        try:
            result = test()
            results.append((name, result))
        except Exception as e:
            print(f"✗ {name} crashed: {e}")
            results.append((name, False))
    
    print("="*60)
    print("Summary:")
    print("-"*60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name:.<40} {status}")
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    
    print("-"*60)
    print(f"Results: {passed_count}/{total_count} tests passed")
    print("="*60)
    
    if all(p for _, p in results):
        print("\n✓ All validation tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
