"""
Check the observation space of saved SAC models
"""
from stable_baselines3 import SAC
import zipfile
import json
from pathlib import Path

models_to_check = [
    "logs/sac_spiral/final_model.zip",
    "logs/sac_spiral/best_model/best_model.zip",
    "logs/SAC/Drone/best_model/best_model.zip",
    "logs/final/sac_spiral.zip",
]

print("=" * 80)
print("SAC Model Observation Space Check")
print("=" * 80)

for model_path in models_to_check:
    path = Path(model_path)
    if not path.exists():
        print(f"\n❌ {model_path} - NOT FOUND")
        continue
    
    try:
        # Open the zip file and read the data
        with zipfile.ZipFile(model_path, 'r') as archive:
            # Read the data.json which contains model metadata
            with archive.open('data') as f:
                data = json.load(f)
                obs_space = data.get('observation_space', {})
                
                print(f"\n✓ {model_path}")
                print(f"  Observation space type: {obs_space.get('_gym_type', 'Unknown')}")
                if 'shape' in obs_space:
                    print(f"  Observation shape: {obs_space['shape']}")
                elif 'Box' in str(obs_space):
                    print(f"  Observation space: {obs_space}")
                
                # Also check policy info
                policy_class = data.get('policy_class', {})
                print(f"  Policy class: {policy_class.get('_gym_type', 'Unknown')}")
                
    except Exception as e:
        print(f"\n⚠️  {model_path} - Error reading: {e}")

print("\n" + "=" * 80)
