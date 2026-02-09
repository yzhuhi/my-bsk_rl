"""
Interactive 3D Constellation Visualization with Plotly.

This script visualizes satellite constellations, task locations, and communication links
in an interactive 3D viewer. It parses simulation logs to reconstruct:
- Satellite trajectories (blue)
- Task locations (cyan)
- ISL communication links (yellow)
- Downlink/SGL links (green)

Usage:
    python plot_constellation_links.py --log sim_log.json
"""

import json
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import argparse
from typing import List, Dict, Tuple


def create_earth_sphere(radius_km: float = 6371.0, resolution: int = 50):
    """Create a 3D sphere mesh representing Earth."""
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = radius_km * np.outer(np.cos(u), np.sin(v))
    y = radius_km * np.outer(np.sin(u), np.sin(v))
    z = radius_km * np.outer(np.ones(np.size(u)), np.cos(v))
    
    return go.Surface(
        x=x, y=y, z=z,
        colorscale=[[0, 'rgb(30,60,100)'], [1, 'rgb(70,130,180)']],
        showscale=False,
        opacity=0.8,
        name='Earth',
        hoverinfo='skip'
    )


def parse_simulation_log(log_file: Path) -> Dict:
    """Parse simulation log file to extract trajectory and event data."""
    with open(log_file, 'r') as f:
        data = json.load(f)
    return data


def create_constellation_figure(
    sat_positions: Dict[str, np.ndarray],  # {sat_name: Nx3 array}
    task_positions: np.ndarray,  # Mx3 array
    isl_events: List[Tuple[str, str, float]],  # [(sat1, sat2, time), ...]
    downlink_events: List[Tuple[str, np.ndarray, float]],  # [(sat, task_pos, time), ...]
    time_step: float = 0,  # Current time for filtering events
) -> go.Figure:
    """Create interactive 3D Plotly figure."""
    
    fig = go.Figure()
    
    # Add Earth
    fig.add_trace(create_earth_sphere())
    
    # Add task locations (static cyan dots)
    task_pos_km = task_positions / 1000.0  # Convert m to km
    fig.add_trace(go.Scatter3d(
        x=task_pos_km[:, 0],
        y=task_pos_km[:, 1],
        z=task_pos_km[:, 2],
        mode='markers',
        marker=dict(size=4, color='cyan', symbol='diamond'),
        name='Tasks',
        text=[f'Task-{i}' for i in range(len(task_positions))],
        hoverinfo='text'
    ))
    
    # Add satellite trajectories
    for sat_name, positions in sat_positions.items():
        pos_km = positions / 1000.0  # Convert m to km
        
        # Trajectory line
        fig.add_trace(go.Scatter3d(
            x=pos_km[:, 0],
            y=pos_km[:, 1],
            z=pos_km[:, 2],
            mode='lines',
            line=dict(color='rgba(150,150,150,0.3)', width=1),
            name=f'{sat_name} trajectory',
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Current position (larger marker)
        current_idx = min(int(time_step), len(positions) - 1)
        fig.add_trace(go.Scatter3d(
            x=[pos_km[current_idx, 0]],
            y=[pos_km[current_idx, 1]],
            z=[pos_km[current_idx, 2]],
            mode='markers',
            marker=dict(size=8, color='blue', symbol='circle'),
            name=sat_name,
            text=sat_name,
            hoverinfo='text'
        ))
    
    # Add ISL links (yellow) - only those active at current time
    for sat1, sat2, event_time in isl_events:
        if abs(event_time - time_step) < 1.0:  # Within 1 second
            # Calculate proper index based on trajectory length
            idx1 = min(int((time_step / 5700) * len(sat_positions[sat1])), len(sat_positions[sat1]) - 1)
            idx2 = min(int((time_step / 5700) * len(sat_positions[sat2])), len(sat_positions[sat2]) - 1)
            
            pos1 = sat_positions[sat1][idx1] / 1000.0
            pos2 = sat_positions[sat2][idx2] / 1000.0
            
            fig.add_trace(go.Scatter3d(
                x=[pos1[0], pos2[0]],
                y=[pos1[1], pos2[1]],
                z=[pos1[2], pos2[2]],
                mode='lines',
                line=dict(color='yellow', width=4),
                name=f'ISL: {sat1}↔{sat2}',
                showlegend=False,
                hoverinfo='text',
                text=f'ISL: {sat1} → {sat2}'
            ))
    
    # Add downlink/SGL (green)
    for sat_name, task_pos, event_time in downlink_events:
        if abs(event_time - time_step) < 1.0:
            idx = min(int((time_step / 5700) * len(sat_positions[sat_name])), len(sat_positions[sat_name]) - 1)
            sat_pos = sat_positions[sat_name][idx] / 1000.0
            task_pos_km = task_pos / 1000.0
            
            fig.add_trace(go.Scatter3d(
                x=[sat_pos[0], task_pos_km[0]],
                y=[sat_pos[1], task_pos_km[1]],
                z=[sat_pos[2], task_pos_km[2]],
                mode='lines',
                line=dict(color='green', width=3),
                name=f'Downlink: {sat_name}',
                showlegend=False,
                hoverinfo='text',
                text=f'Downlink: {sat_name} → UD'
            ))
    
    # Layout
    fig.update_layout(
        title=f'Satellite Constellation - t={time_step:.1f}s',
        scene=dict(
            xaxis_title='X (km)',
            yaxis_title='Y (km)',
            zaxis_title='Z (km)',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2),
                center=dict(x=0, y=0, z=0)
            ),
            bgcolor='rgb(10, 10, 20)'
        ),
        height=800,
        showlegend=True,
        paper_bgcolor='rgb(20, 20, 30)',
        font=dict(color='white')
    )
    
    return fig


def create_animation_frames(
    sat_positions: Dict[str, np.ndarray],
    task_positions: np.ndarray,
    isl_events: List[Tuple[str, str, float]],
    downlink_events: List[Tuple[str, np.ndarray, float]],
    num_frames: int = 100
) -> List[go.Frame]:
    """Create animation frames for time evolution."""
    frames = []
    max_time = max(
        max([t for _, _, t in isl_events], default=0),
        max([t for _, _, t in downlink_events], default=0),
        max([len(p) for p in sat_positions.values()], default=1)
    )
    
    for i in range(num_frames):
        time_step = (i / num_frames) * max_time
        frame_data = []
        
        # Update satellite positions
        for sat_name, positions in sat_positions.items():
            idx = min(int(time_step), len(positions) - 1)
            pos_km = positions[idx] / 1000.0
            frame_data.append(go.Scatter3d(
                x=[pos_km[0]],
                y=[pos_km[1]],
                z=[pos_km[2]]
            ))
        
        frames.append(go.Frame(data=frame_data, name=f'frame_{i}'))
    
    return frames


def main():
    parser = argparse.ArgumentParser(description='Visualize constellation with Plotly')
    parser.add_argument('--log', type=str, help='Path to simulation log JSON file')
    parser.add_argument('--output', type=str, default='constellation_viz.html',
                        help='Output HTML file')
    parser.add_argument('--animate', action='store_true',
                        help='Create animation')
    args = parser.parse_args()
    
    # For demo, create synthetic data
    print("Generating synthetic constellation data...")
    
    # Synthetic satellite positions (circular orbits)
    n_sats = 6
    altitude_km = 781
    radius_km = 6371 + altitude_km
    t_points = 100
    
    sat_positions = {}
    for i in range(n_sats):
        phase = (2 * np.pi * i) / n_sats
        t = np.linspace(0, 5700, t_points)  # One orbit ~95 min
        
        positions = np.zeros((t_points, 3))
        for j, time in enumerate(t):
            angle = phase + (2 * np.pi * time / 5700)
            positions[j, 0] = radius_km * np.cos(angle) * 1000  # Convert to m
            positions[j, 1] = radius_km * np.sin(angle) * 1000
            positions[j, 2] = radius_km * 0.1 * np.sin(time / 500) * 1000  # Slight Z variation
        
        sat_positions[f'sat-{i}'] = positions
    
    # Synthetic task positions (on Earth surface)
    n_tasks = 20
    task_positions = np.zeros((n_tasks, 3))
    for i in range(n_tasks):
        lat = np.random.uniform(-60, 60) * np.pi / 180
        lon = np.random.uniform(-180, 180) * np.pi / 180
        R = 6371e3
        task_positions[i, 0] = R * np.cos(lat) * np.cos(lon)
        task_positions[i, 1] = R * np.cos(lat) * np.sin(lon)
        task_positions[i, 2] = R * np.sin(lat)
    
    # Synthetic ISL events
    isl_events = [
        ('sat-0', 'sat-1', 1200.0),
        ('sat-1', 'sat-2', 1500.0),
        ('sat-3', 'sat-4', 2000.0),
    ]
    
    # Synthetic downlink events
    downlink_events = [
        ('sat-0', task_positions[0], 1000.0),
        ('sat-2', task_positions[5], 1800.0),
    ]
    
    # Create figure
    fig = create_constellation_figure(
        sat_positions=sat_positions,
        task_positions=task_positions,
        isl_events=isl_events,
        downlink_events=downlink_events,
        time_step=1500.0  # Show state at t=1500s
    )
    
    # Save to HTML
    output_path = Path(args.output if args.log else 'constellation_viz.html')
    fig.write_html(str(output_path))
    print(f"✓ Visualization saved to {output_path}")
    print(f"  Open it in a browser to interact with the 3D view!")


if __name__ == '__main__':
    main()
