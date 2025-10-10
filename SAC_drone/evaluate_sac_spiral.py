from stable_baselines3 import SAC

# Load trained model
model = SAC.load("./sac_spiral_models/sac_spiral_20250110_120000_best")

# Test on environment with GUI
env = create_spiral_env(gui=True, mode="spiral")
obs, info = env.reset()

for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()