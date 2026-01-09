"""
Vizard Visualization Demo for Lite Computation Satellite
=========================================================

This script demonstrates the 3D visualization of a LiteComputationSatellite 
constellation using Basilisk's Vizard tool.

Requirements:
- Vizard must be downloaded and installed
- Set VIZARD_PATH environment variable or modify the path below

Usage:
    python lite_demo_vizard.py
    
After running, open the generated .bin file in Vizard to view the simulation.
"""

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)

from bsk_rl import ConstellationTasking
from bsk_rl.sats import LiteComputationSatellite
from bsk_rl.comm import LOSMultiCommunication
from bsk_rl.data import NoReward  # Use built-in NoReward instead of custom class
from bsk_rl.scene import CityTaskScenario
from bsk_rl.utils.orbital import walker_delta_args

# Import TaskVizController if available
try:
    from Basilisk.ExternalModules import taskVizController
    TASK_VIZ_AVAILABLE = True
except ImportError:
    TASK_VIZ_AVAILABLE = False
    print("Warning: TaskVizController not available, visualization will be limited")

# Import vizSupport for ISL visualization
try:
    from Basilisk.utilities import vizSupport
    VIZ_SUPPORT_AVAILABLE = True
except ImportError:
    VIZ_SUPPORT_AVAILABLE = False
    print("Warning: vizSupport not available, ISL visualization disabled")


class ISLVisualizer:
    """
    Helper class for visualizing Inter-Satellite Links (ISL) in Vizard.
    
    Uses createTargetLine to draw dynamic lines between communicating satellites.
    """
    
    # Color definitions for different link types
    # Note: Uplink is NOT included - Vizard's showLocationCommLines already shows sat-to-task visibility
    COLORS = {
        'isl': 'yellow',           # ISL offload link (satellite to satellite)
        'downlink': 'green',       # Downlink to ground (result transmission)
    }

    
    def __init__(self, viz_instance):
        """Initialize with a vizInterface instance."""
        self.viz = viz_instance
        self.active_links = {}  # Track active links: (from, to) -> color
        self._link_counter = 0
    
    def add_isl_link(self, from_sat_name: str, to_sat_name: str, link_type: str = 'isl'):
        """
        Draw an ISL link between two satellites.
        
        Args:
            from_sat_name: Name of the source satellite
            to_sat_name: Name of the target satellite  
            link_type: Type of link ('isl', 'downlink', 'uplink')
        """
        if not VIZ_SUPPORT_AVAILABLE:
            return
        
        link_key = (from_sat_name, to_sat_name)
        if link_key in self.active_links:
            return  # Link already exists
        
        color = self.COLORS.get(link_type, 'white')
        
        try:
            vizSupport.createTargetLine(
                self.viz,
                toBodyName=to_sat_name,
                lineColor=color,
                fromBodyName=from_sat_name
            )
            self.active_links[link_key] = color
            self._link_counter += 1
            
            if self._link_counter <= 5:  # Only print first 5
                print(f"[ISL_VIZ] Drew {link_type} link: {from_sat_name} -> {to_sat_name}")
        except Exception as e:
            print(f"[ISL_VIZ] Error creating link: {e}")
    
    def clear_all_links(self):
        """Clear all visualization links."""
        if not VIZ_SUPPORT_AVAILABLE:
            return
        
        try:
            vizSupport.targetLineList.clear()
            vizSupport.updateTargetLineList(self.viz)
            self.active_links.clear()
        except Exception as e:
            print(f"[ISL_VIZ] Error clearing links: {e}")
    
    def get_link_count(self) -> int:
        """Return number of active links."""
        return len(self.active_links)


# ============================================================
# Configuration
# ============================================================

# Vizard output directory - change this to your preferred location
VIZARD_OUTPUT_DIR = Path("./vizard_output")

# Satellite parameters (same as lite mode)
sat_args = {
    "cpuMaxFrequency": 4.8e9,
    "cpuMinFrequency": 2.0e9,
    "cpuWorkload": 1000.0,
    "cpuPowerDraw": -40.0,
    "transmitterBaudRate": -100e6,
    "transmitterPowerDraw": -20.0,
    "basePowerDraw": -20.0,
    "batteryStorageCapacity": 2.7e6,
    "storedCharge_Init": 2.16e6,
    "panelArea": 2.0,
    "taskMinimumElevation": 0.26,
    "dataStorageCapacity": 4e12,
    "mass": 680.0,
    "width": 1.8,
    "depth": 1.2,
    "height": 1.2,
}

# Walker Delta constellation configuration  
# Note: Number of satellites is determined by the satellites list, not this config
constellation_args = walker_delta_args(
    n_planes=4,            # 4 orbital planes
    altitude=781,          # km
    inc=86.4,              # deg (near-polar)
    rel_phasing=2.0,       # Relative phasing between planes
)


def create_vizard_env():
    """Create environment with Vizard enabled."""
    
    # Task scenario
    task_scenario = CityTaskScenario(
        n_tasks=100,
        n_select_from=500,
        data_size_range=(1e6, 50e6),
        workload_range=(500, 1500),
        max_delay_range=(60.0, 600.0),
        task_arrival_rate=0.1,
    )
    
    # Create environment with Vizard enabled
    env = ConstellationTasking(
        satellites=[
            LiteComputationSatellite(
                name=f"sat-{i}",
                sat_args=sat_args,
            )
            for i in range(24)
        ],
        sat_arg_randomizer=constellation_args,
        scenario=task_scenario,
        rewarder=NoReward(),
        communicator=LOSMultiCommunication(),
        time_limit=6000.0,  # 1 hour simulation
        max_step_duration=60.0,  # 60s per step
        # Enable Vizard visualization
        vizard_dir=str(VIZARD_OUTPUT_DIR),
        vizard_settings={
            "openBrowserOnRun": False,  # Don't auto-open browser
            "showSpaceCraftLabels": 1,  # Show satellite labels
            "showBodyLabels": 0,
            "orbitLineOn": 1,  # Show orbit lines
        },
    )
    
    return env


def run_demo(env, steps=100):
    """Run the simulation for visualization."""
    print("Resetting environment...")
    obs, info = env.reset()
    
    # DEBUG: Diagnostic dump
    print(f"DEBUG: TASK_VIZ_AVAILABLE = {TASK_VIZ_AVAILABLE}")
    print(f"DEBUG: env.simulator has vizInstance? {hasattr(env.simulator, 'vizInstance')}")
    
    # Force enable satellite-to-task communication lines in Vizard
    if hasattr(env.simulator, 'vizInstance'):
        print(f"DEBUG: vizInstance.settings.showLocationCommLines before = {env.simulator.vizInstance.settings.showLocationCommLines}")
        env.simulator.vizInstance.settings.showLocationCommLines = 1
        print(f"DEBUG: vizInstance.settings.showLocationCommLines after = {env.simulator.vizInstance.settings.showLocationCommLines}")
        print("✓ Enabled showLocationCommLines for task visualization")
    
    # Add TaskVizController after simulator is created
    if TASK_VIZ_AVAILABLE:
        try:
            task_viz = taskVizController.TaskVizController()
            task_viz.ModelTag = "taskVizController"
            task_viz.enableLogging = True
            
            # Connect to all satellites
            for sat in env.satellites:
                if hasattr(sat, 'task_event_writer'):
                    task_viz.taskEventInMsg.subscribeTo(sat.task_event_writer)
            
            # Add to simulation
            # Use task name from the first satellite's dynamics model
            task_name = env.satellites[0].dynamics.task_name
            env.simulator.AddModelToTask(task_name, task_viz)
            
            # Connect to vizInterface.liveSettings for dynamic line visualization
            if hasattr(env.simulator, 'vizInstance'):
                task_viz.vizLiveSettings = env.simulator.vizInstance.liveSettings
                task_viz.enableDynamicLines = True
                print("✓ TaskVizController connected to vizInterface.liveSettings")
            
            # Count connected satellites
            connected_count = sum(1 for sat in env.satellites if hasattr(sat, 'task_event_writer'))
            print(f"✓ TaskVizController enabled with {connected_count}/{len(env.satellites)} satellites connected")

        except Exception as e:
            print(f"ERROR initializing TaskVizController: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("✗ TaskVizController not available")
    
    print(f"\nRunning simulation for {steps} steps...")
    # Initialize ISL Visualizer
    isl_visualizer = None
    if VIZ_SUPPORT_AVAILABLE and hasattr(env.simulator, 'vizInstance'):
        isl_visualizer = ISLVisualizer(env.simulator.vizInstance)
        
        # Attach visualizer callback to each satellite
        for sat in env.satellites:
            sat.isl_visualizer = isl_visualizer
        
        print(f"ISLVisualizer enabled for {len(env.satellites)} satellites")
    else:
        print("ISLVisualizer not available (missing vizInstance or vizSupport)")
    
    for step in range(steps):
        # Random actions for demo
        actions = {sat.id: sat.action_space.sample() for sat in env.satellites}
        obs, reward, terminated, truncated, info = env.step(actions)
        
        if step % 10 == 0:
            time_mins = env.simulator.sim_time / 60.0
            print(f"  Step {step}, Sim time: {time_mins:.1f} mins")
        
        if any(terminated.values()) or any(truncated.values()):
            print("Episode ended early")
            break
    
    print(f"\nSimulation complete!")
    print(f"Vizard output saved to: {VIZARD_OUTPUT_DIR}")
    print("Open the .bin file in Vizard to visualize the constellation.")


if __name__ == "__main__":
    print("=" * 60)
    print("Vizard Demo for LiteComputationSatellite Constellation")
    print("=" * 60)
    
    # Create output directory
    VIZARD_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create environment
    print(f"\nCreating environment with 24 satellites...")
    env = create_vizard_env()
    print(f"  ✓ Environment created")
    print(f"  ✓ Vizard output: {VIZARD_OUTPUT_DIR}")
    
    # Run demo
    print("\n" + "-" * 60)
    run_demo(env, steps=180)  # 60 steps = 1 hour at 60s/step
    
    env.close()
    print("\nDone!")
