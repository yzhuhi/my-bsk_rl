"""
Vizard Visualization Demo for PPT Presentation (Enhanced)
=========================================================

Features:
1. **Task Perception**: Green lines indicating task access (Satellite <-> Task).
2. **Task Offloading**: Yellow lines indicating inter-satellite routing (Satellite <-> Satellite).
3. **High Density**: 40 Satellites, 500 Tasks for a busy, impressive visual.

Usage:
    python computingSatellite_demo_vizard_ppt.py
"""

import logging
import re
import random
import numpy as np
from typing import Any, Optional, List
from pathlib import Path
from weakref import proxy

# Configure Logging
logging.basicConfig(level=logging.INFO)
bsk_logger = logging.getLogger("bsk_rl")

from bsk_rl import ConstellationTasking
from bsk_rl.sats import LiteComputationSatellite, ComputationSatellite
from bsk_rl.comm import LOSMultiCommunication
from bsk_rl.data import NoReward
from bsk_rl.scene import CityTaskScenario
from bsk_rl.utils.orbital import walker_delta_args

# --- Visualization Support ---
try:
    from Basilisk.utilities import vizSupport
    VIZ_SUPPORT_AVAILABLE = True
except ImportError:
    VIZ_SUPPORT_AVAILABLE = False
    print("Warning: vizSupport not available via Basilisk.utilities")

try:
    from Basilisk.ExternalModules import taskVizController
    TASK_VIZ_AVAILABLE = True
except ImportError:
    TASK_VIZ_AVAILABLE = False


class ISLVisualizer:
    """Helper class for visualizing Inter-Satellite Links (ISL) using Python-side calls."""
    
    COLORS = {
        'isl': 'yellow',      # Offload link
        'downlink': 'green',  # Result return
    }
    
    def __init__(self, viz_instance):
        self.viz = viz_instance
        self.active_links = {}
        self.link_ttl = {}  # Time To Live for links (in steps)
        self.ttl_duration = 10 # Keep lines valid for X steps to make them visible
        
    def add_isl_link(self, from_sat: str, to_sat: str, link_type: str = 'isl'):
        if not VIZ_SUPPORT_AVAILABLE: return
        
        # Unique key for the link (undirected for visualization simplicty)
        key = tuple(sorted((from_sat, to_sat))) 
        
        # Reset TTL if exists
        self.link_ttl[key] = self.ttl_duration
        
        if key not in self.active_links:
            try:
                # Draw new line
                vizSupport.createTargetLine(
                    self.viz,
                    toBodyName=to_sat,
                    lineColor=self.COLORS.get(link_type, 'white'),
                    fromBodyName=from_sat
                )
                self.active_links[key] = True
                print(f"[VIZ] ⚡ Draw Link: {from_sat} <--> {to_sat}")
            except Exception as e:
                # print(f"[VIZ] Error drawing link: {e}")
                pass

    def update(self):
        """Decrease TTL and clear expired links."""
        if not VIZ_SUPPORT_AVAILABLE: return
        
        expired = []
        for key in list(self.link_ttl.keys()):
            self.link_ttl[key] -= 1
            if self.link_ttl[key] <= 0:
                expired.append(key)
        
        if expired:
            # Re-drawing is expensive/complex in vizSupport (clearing specific lines is hard).
            # Strategy: We unfortunately have to clear ALL target lines and redraw active ones.
            # This might flicker, but ensures correctness.
            
            try:
                # Clear all lines managed by vizSupport (targetLineList)
                vizSupport.targetLineList.clear()
                
                # Remove expired from tracking
                for key in expired:
                    if key in self.active_links:
                        del self.active_links[key]
                    if key in self.link_ttl:
                        del self.link_ttl[key]

                # Redraw all currently active valid links
                # Note: vizSupport.createTargetLine adds to the global list
                # We need to re-call it for every active link
                
                # Copy active keys to iterate
                current_active = list(self.active_links.keys()) 
                self.active_links.clear() # Clear tracking to re-add
                
                for (sat1, sat2) in current_active:
                     vizSupport.createTargetLine(
                        self.viz,
                        toBodyName=sat2,
                        lineColor='yellow', # Default to yellow for now
                        fromBodyName=sat1
                    )
                     self.active_links[(sat1, sat2)] = True
                
                # Push updates to Vizard
                vizSupport.updateTargetLineList(self.viz)
                
            except Exception as e:
                print(f"[VIZ] Error refreshing links: {e}")


class ISLLogHandler(logging.Handler):
    """Intercepts [RELAY] logs and calls the visualizer."""
    def __init__(self, visualizer):
        super().__init__()
        self.visualizer = visualizer
        self.relay_pattern = re.compile(r"Forwarding slice .* to (sat-\d+)")
        
    def emit(self, record):
        try:
            msg = record.getMessage()
            if "[RELAY]" in msg:
                match = self.relay_pattern.search(msg)
                if match:
                    to_sat = match.group(1)
                    if "sat-" in record.name:
                        from_sat = record.name.split(".")[-1]
                        self.visualizer.add_isl_link(from_sat, to_sat)
        except Exception:
            self.handleError(record)

# Monkey-patch routing
def _mask_and_normalize_routing_probs(self, base_ratios: np.ndarray, neighbor_satellites: List[Any]) -> np.ndarray:
    # Ensure inputs are treated as non-negative ratios
    # If the network outputs logits, we should ideally softmax, but for this linear normalization usage:
    valid_logits = np.maximum(base_ratios.copy(), 0.0)
    
    for i in range(1, len(valid_logits)):
        neighbor_idx = i - 1
        # Check if neighbor exists
        if neighbor_idx >= len(neighbor_satellites) or neighbor_satellites[neighbor_idx] is None:
            valid_logits[i] = 0.0
    
    total = np.sum(valid_logits)
    if total > 1e-6:
        return valid_logits / total
    else:
        probs = np.zeros_like(valid_logits)
        probs[0] = 1.0
        return probs

ComputationSatellite._mask_and_normalize_routing_probs = _mask_and_normalize_routing_probs
# LiteComputationSatellite._mask_and_normalize_routing_probs = _mask_and_normalize_routing_probs # Inherits now? Explicit is safer
LiteComputationSatellite._mask_and_normalize_routing_probs = _mask_and_normalize_routing_probs

# Configuration
VIZARD_OUTPUT_DIR = Path("./vizard_output_ppt")
VIZARD_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

start_time_str = str(random.randint(1000, 9999)) # Unique ID for this run files

sat_args = {
    "cpuMaxFrequency": 4.8e9,
    "cpuMinFrequency": 2.0e9,
    "cpuWorkload": 500.0,
    "transmitterBaudRate": -1.0e9,
    "dataStorageCapacity": 4e12,
}

# HIGH DENSITY SCENE
constellation_args = walker_delta_args(
    n_planes=10, altitude=781, inc=86.4, rel_phasing=2.0
)

def create_env():
    task_scenario = CityTaskScenario(
        n_tasks=800,
        n_select_from=2000,
        data_size_range=(1e7, 5e7),
        task_arrival_rate=1.0,
    )
    
    env = ConstellationTasking(
        # Use simple LiteComputationSatellite (training_mode=True default is fine for Lite, 
        # but we need to ensure FSW doesn't crash if we change it. 
        # Let's keep defaults but externally manage lines.)
        satellites=[LiteComputationSatellite(name=f"sat-{i}", sat_args=sat_args) for i in range(60)],
        sat_arg_randomizer=constellation_args,
        scenario=task_scenario,
        rewarder=NoReward(),
        communicator=LOSMultiCommunication(),
        time_limit=3600.0,
        max_step_duration=30.0,
        vizard_dir=str(VIZARD_OUTPUT_DIR),
        vizard_settings={
            "showSpaceCraftLabels": 1,
            "orbitLineOn": 1,
            # "showLocationCommLines": 1, # We will force this manually later
        },
    )
    return env

def make_forcing_action(env):
    """Generate actions that force offloading."""
    actions = {}
    for sat in env.satellites:
        action = np.zeros(sat.action_space.shape, dtype=np.float32)
        # Full Power
        action[0:3] = 1.0
        # Force Offload (alpha_local=0, alpha_cloud=0)
        action[3] = 0.0
        action[4] = 0.0
        action[5] = 0.5
        
        # High ISL Weights
        weights = np.random.rand(4) * 10.0
        # High Priority
        action[6] = 0.0 # Min Self (0.0 is safe for linear norm)
        action[7:11] = weights
        # Low Priority
        action[11] = 0.0 # Min Self
        action[12:16] = weights
        
        actions[sat.id] = action
    return actions

def run_ppt_demo(env, steps=200):
    print("Initializing PPT Demo (Python-Viz Mode)...")
    env.reset()
    
    # 1. Force Enable Task Lines (Green)
    if hasattr(env.simulator, 'vizInstance'):
        # Explicitly set the setting on the live instance
        env.simulator.vizInstance.settings.showLocationCommLines = 1
        print("✓ Task Lines Enabled (Green) [showLocationCommLines=1]")

    # 2. Setup ISL Viz (Python)
    isl_viz = None
    if VIZ_SUPPORT_AVAILABLE and hasattr(env.simulator, 'vizInstance'):
        isl_viz = ISLVisualizer(env.simulator.vizInstance)
        handler = ISLLogHandler(isl_viz)
        bsk_logger.addHandler(handler)
        print("✓ ISL Routing Lines Enabled (Yellow) [ISLVisualizer]")
        
    print(f"\nRunning simulation for {steps} steps...")
    
    try:
        for s in range(steps):
            actions = make_forcing_action(env)
            env.step(actions)
            
            # Update visualizer (handle TTL)
            if isl_viz:
                isl_viz.update()
                
            if s % 10 == 0:
                print(f"Step {s}/{steps} | Time: {env.simulator.sim_time/60:.1f}m")
                
    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        print(f"\nSaved visualization to {VIZARD_OUTPUT_DIR}")

if __name__ == "__main__":
    env = create_env()
    try:
        run_ppt_demo(env)
    finally:
        env.close()
