"""
STIN Vizard Visualization Script
================================

This script sets up the full STIN environment (CityTaskScenario + ComputationSatellite)
and runs a simulation to generate a Vizard visualization file.

Usage:
    python stin_viz.py
    
Output:
    ./vizard_output/  (contains .bin file for Vizard)
"""

import logging
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO)

from bsk_rl import ConstellationTasking
from bsk_rl.sats import ComputationSatellite
from bsk_rl.comm import LOSMultiCommunication
from bsk_rl.data import STINTaskReward
from bsk_rl.scene import CityTaskScenario
from bsk_rl.utils.orbital import walker_delta_args

# --- Configuration ---
VIZARD_OUTPUT_DIR = Path("./vizard_output")
NUM_SATELLITES = 30  # 5 planes * 6 sats
SIM_DURATION = 6000.0  # 100 minutes (approx one orbit)
STEP_DURATION = 60.0

# Walker Delta parameters (Polar constellation)
constellation_args = walker_delta_args(
    n_planes=5,
    altitude=800,  # km
    inc=86.4,      # deg
    rel_phasing=0.0
)

# Satellite arguments (matching training config)
sat_args = {
    # Power params
    "batteryStorageCapacity": 80.0 * 3600.0, # 80 Wh
    "storedCharge_Init": 80.0 * 3600.0,
    "basePowerDraw": -5.0,
    "cpuPowerDraw": -20.0,       # Max CPU power
    "transmitterPowerDraw": -15.0, # Max TX power
    # Computation params
    "cpuMaxFrequency": 1e9,      # 1 GHz
    "cpuMinFrequency": 0.1e9,
    "cpuWorkload": 0.0,          # Placeholder
}


def create_viz_env():
    """Create the STIN environment with Vizard enabled."""
    
    # 1. Scenario: Real city tasks
    scenario = CityTaskScenario(
        n_tasks=800,               # Large enough to see tasks
        n_select_from=2000,
        task_arrival_rate=1.0,     # High rate for visibility
        data_size_range=(4e7, 1.6e8),
        # Ensure Vizard uses blue lines for visibility
    )
    
    # 2. Environment
    env = ConstellationTasking(
        satellites=[
            ComputationSatellite(
                name=f"Sat-{i}",
                sat_args=sat_args,
                training_mode=False,  # IMPORTANT: Enable Vizard events
            )
            for i in range(NUM_SATELLITES)
        ],
        sat_arg_randomizer=constellation_args,
        scenario=scenario,
        rewarder=STINTaskReward(
            w_complete=5.0,
            w_expired=5.0
        ),
        communicator=LOSMultiCommunication(),
        time_limit=SIM_DURATION,
        max_step_duration=STEP_DURATION,
        
        # 3. Vizard Viz Settings
        vizard_dir=str(VIZARD_OUTPUT_DIR),
        vizard_settings={
            "openBrowserOnRun": False,  # Manual open
            "showSpaceCraftLabels": 1,
            "showBodyLabels": 0,
            "orbitLineOn": 1,
            "showLocationCommLines": 1, # Show blue lines to visible tasks
            "showLocationCones": 1,     # Show sensing cones
            "showLocationLabels": 0,    # Show task names
            "celestialBodyColor": "earth: 0.2 0.2 1.0 1.0", # Make earth blue-er
        }
    )
    
    return env

def run_viz_simulation():
    """Run the simulation loop."""
    print(f"Creating environment...")
    VIZARD_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    env = create_viz_env()
    print("Resetting environment...")
    obs, info = env.reset()
    
    print(f"Running simulation for {SIM_DURATION}s...")
    
    steps = int(SIM_DURATION / STEP_DURATION)
    
    for i in range(steps):
        # Sample random actions (just to keep simulation running)
        actions = {sat.id: sat.action_space.sample() for sat in env.satellites}
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(actions)
        
        # Print progress
        if i % 10 == 0:
            time_percent = (i / steps) * 100
            print(f"Progress: {time_percent:.1f}%")
            
        if any(terminated.values()):
            print("Terminated early!")
            break
            
    env.close()
    print("Simulation complete.")
    print(f"✅ Vizard data saved to: {VIZARD_OUTPUT_DIR.resolve()}")
    print("Use 'Vizard.exe' to open the generated .bin file.")

if __name__ == "__main__":
    run_viz_simulation()
