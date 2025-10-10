"""
Debug: Check what the actual observation looks like
"""
import numpy as np
import sys
from pathlib import Path

# Add gym-pybullet-drones to path
gym_path = Path(__file__).parent.parent / "gym-pybullet-drones"
sys.path.append(str(gym_path))

from gym_pybullet_drones.envs.SpiralAviary import SpiralAviary
from gym_pybullet_drones.utils.enums import DroneModel, ActionType, Physics

# Create environment
env = SpiralAviary(
    drone_model=DroneModel.CF2X,
    gui=False,
    record=False,
    mode="spiral",
    physics=Physics.PYB,
    pyb_freq=240,
    ctrl_freq=30,
    act=ActionType.RPM
)

print("=" * 80)
print("Observation Space Debug")
print("=" * 80)

# Reset and get observation
obs, info = env.reset()

print(f"\nObservation shape: {obs.shape}")
print(f"Observation space shape: {env.observation_space.shape}")
print(f"\nFirst observation:\n{obs}")
print(f"\nObservation breakdown:")
print(f"  Total length: {len(obs)}")

# Try to match with expected structure
if len(obs) >= 16:
    print(f"  state[0:16]: {obs[0:16].shape} - {obs[0:16]}")
if len(obs) >= 20:
    print(f"  last_action[16:20]: {obs[16:20].shape} - {obs[16:20]}")
if len(obs) >= 23:
    print(f"  ref_pos[20:23]: {obs[20:23].shape} - {obs[20:23]}")
if len(obs) >= 26:
    print(f"  target_vel[23:26]: {obs[23:26].shape} - {obs[23:26]}")
if len(obs) >= 29:
    print(f"  ref_att[26:29]: {obs[26:29].shape} - {obs[26:29]}")
if len(obs) >= 32:
    print(f"  delta_pos[29:32]: {obs[29:32].shape} - {obs[29:32]}")
if len(obs) >= 35:
    print(f"  delta_vel[32:35]: {obs[32:35].shape} - {obs[32:35]}")
if len(obs) >= 38:
    print(f"  delta_att[35:38]: {obs[35:38].shape} - {obs[35:38]}")

print("\n" + "=" * 80)

env.close()
