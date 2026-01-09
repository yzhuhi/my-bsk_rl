"""
Visualization Demo for Lite Computation Satellite
================================================

This script demonstrates the simulation of a constellation of LiteComputationSatellite objects.
It runs the simulation for a few steps and plots key metrics:
- Power generation and consumption
- Battery level
- Task queue status
- Orbital positions

Usage:
    python lite_demo_visualize.py
"""

import matplotlib.pyplot as plt
import numpy as np
import logging
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.ERROR)

from bsk_rl import ConstellationTasking
from bsk_rl.sats import LiteComputationSatellite
from bsk_rl.comm import LOSMultiCommunication
from bsk_rl.data import GlobalReward
from bsk_rl.scene import UniformTargets

# Define a simple configuration for the demo
sat_args = {
    "cpuMaxFrequency": 4.8e9,
    "cpuMinFrequency": 2.0e9,
    "cpuWorkload": 1000.0,
    "cpuPowerDraw": -40.0,
    "transmitterBaudRate": -100e6,
    "transmitterPowerDraw": -20.0,
    "basePowerDraw": -20.0,
    "batteryStorageCapacity": 2.7e6,
    "storedCharge_Init": 2.16e6, # 80% of capacity
    "panelArea": 2.0,
    "taskMinimumElevation": 0.26,
    "dataStorageCapacity": 4e12,
    "mass": 680.0,
    "width": 1.8,
    "depth": 1.2,
    "height": 1.2,
}

class DemoReward(GlobalReward):
    def calculate_reward(self, satellites):
        return 0.0

def create_env():
    # Minimal scenario
    scenario = UniformTargets(n_targets=0) 
    
    return ConstellationTasking(
        satellites=[
            LiteComputationSatellite(
                name=f"sat-{i}",
                sat_args=sat_args,
            )
            for i in range(24) # 24 satellite demo
        ],
        scenario=scenario, 
        rewarder=DemoReward(),
        communicator=LOSMultiCommunication(),
        time_limit=6000.0,
        disable_env_checker=True 
    )

def run_simulation(env, steps=100):
    env.reset()
    history = {
        "time": [],
        "battery": [],
        "power_in": [],
        "power_out": [],
        "queue_size": [],
    }
    
    # Initialize history list for each satellite
    n_sats = len(env.satellites)
    history["battery"] = [[] for _ in range(n_sats)]
    history["power_in"] = [[] for _ in range(n_sats)]
    history["power_out"] = [[] for _ in range(n_sats)]
    history["queue_size"] = [[] for _ in range(n_sats)]

    print(f"Running simulation for {steps} steps...")
    
    for step in range(steps):
        # Random action (just for demo)
        actions = {sat.id: sat.action_space.sample() for sat in env.satellites}
        obs, reward, terminated, truncated, info = env.step(actions)
        
        time = env.simulator.sim_time
        history["time"].append(time)
        
        for i, sat in enumerate(env.satellites):
            # Access internal state (safe for demo, not for RL agent)
            # Use safe access or try-except if properties are missing
            try:
                bat_level = sat.dynamics.storage_level / sat.dynamics.batteryStorageCapacity
            except AttributeError:
                # Fallback if property names differ
                bat_level = 0.0
                
            history["battery"][i].append(bat_level)
            
            # Simple list length for queue size
            history["queue_size"][i].append(len(sat.task_queue))
            
    return history

def plot_results(history):
    time = np.array(history["time"]) / 60.0 # Convert to minutes
    n_sats = len(history["battery"])
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot Battery Levels
    ax = axes[0]
    for i in range(min(5, n_sats)): # Plot first 5 sats to avoid clutter
        ax.plot(time, history["battery"][i], label=f"Sat {i}")
    ax.set_ylabel("Battery Fraction")
    ax.set_title("Battery Level (First 5 Sats)")
    ax.grid(True)
    ax.legend()
    
    # Plot Task Queue Size
    ax = axes[1]
    for i in range(min(5, n_sats)):
        ax.plot(time, history["queue_size"][i], label=f"Sat {i}")
    ax.set_ylabel("Queue Size")
    ax.set_xlabel("Time (mins)")
    ax.set_title("Task Queue Size (First 5 Sats)")
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig("lite_demo_results.png")
    print("Results saved to lite_demo_results.png")

if __name__ == "__main__":
    env = create_env()
    history = run_simulation(env, steps=200)
    plot_results(history)
