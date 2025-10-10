# SAC Training for Spiral Trajectory Tracking

This directory contains scripts for training and evaluating Soft Actor-Critic (SAC) on the SpiralAviary environment.

## Files

### Training
- **`train_sac_spiral.py`**: Main training script with TensorBoard logging and checkpointing
- **`SpiralEnvTest.py`**: PID baseline test for environment validation

### Evaluation
- **`evaluate_sac_spiral.py`**: Evaluate trained model and generate diagnostic plots

## Quick Start

### 1. Environment Validation (Optional but Recommended)

First, test the environment with PID controller to establish a baseline:

```bash
python SpiralEnvTest.py
```

This will generate PID baseline plots:
- `spiral_test_position_tracking.png`
- `spiral_test_reward_analysis.png`
- `spiral_test_attitude_control.png`
- `spiral_test_3d_trajectory.png`

**Expected PID Performance:**
- Mean position error: ~0.048m
- Mean reward: ~0.99
- Attitude stability: ±4°

### 2. Train SAC Model

```bash
python train_sac_spiral.py
```

**Training Configuration:**
- Total timesteps: 1,000,000
- Parallel environments: 4
- Learning rate: 3e-4
- Buffer size: 1M
- Batch size: 256
- Episode length: 25s

**During Training:**
- Logs saved to: `logs/sac_spiral/`
- Checkpoints saved every 50k timesteps
- Best model saved based on mean reward
- Evaluation runs every 10k timesteps

### 3. Monitor Training

Open TensorBoard in a separate terminal:

```bash
cd C:\Projects\masters-thesis\SAC_drone
tensorboard --logdir=logs/sac_spiral/tensorboard
```

Then open browser to: `http://localhost:6006`

**Metrics to Monitor:**
- `rollout/ep_rew_mean`: Mean episode reward (target: >0.95)
- `rollout/ep_len_mean`: Mean episode length (~25s)
- `train/actor_loss`: Actor network loss
- `train/critic_loss`: Critic network loss

### 4. Evaluate Trained Model

After training completes (or after enough timesteps):

```bash
python evaluate_sac_spiral.py
```

This will:
- Load the best model
- Run evaluation episode with visualization (GUI=True)
- Generate diagnostic plots matching PID format:
  - `sac_position_tracking.png`
  - `sac_reward_analysis.png`
  - `sac_attitude_control.png`
  - `sac_3d_trajectory.png`

### 5. Compare SAC vs PID

Compare the generated plots:

**Position Tracking:**
- Compare `sac_position_tracking.png` with `spiral_test_position_tracking.png`
- Check if SAC matches or improves upon PID's ~0.048m error

**Reward Performance:**
- PID baseline: ~0.99 mean reward
- SAC target: >0.95 mean reward

**Attitude Control:**
- PID baseline: ±4° stability
- SAC target: Similar or better stability

## Directory Structure

```
SAC_drone/
├── train_sac_spiral.py          # Training script
├── evaluate_sac_spiral.py       # Evaluation script
├── SpiralEnvTest.py             # PID baseline test
├── README_SAC_TRAINING.md       # This file
└── logs/
    └── sac_spiral/
        ├── tensorboard/         # TensorBoard logs
        ├── checkpoints/         # Model checkpoints (every 50k steps)
        ├── best_model/          # Best model based on reward
        │   ├── best_model.zip
        │   └── eval/            # Evaluation callback results
        ├── final_model.zip      # Final model after training
        └── vec_normalize.pkl    # Observation normalization stats
```

## Training Tips

### 1. Initial Training Phase (0-100k steps)
- Expect low rewards initially (~0.3-0.5)
- Agent is exploring and filling replay buffer
- Position errors may be large (>0.5m)

### 2. Learning Phase (100k-500k steps)
- Rewards should steadily increase
- Position error should decrease below 0.2m
- Attitude control improves

### 3. Convergence Phase (500k-1M steps)
- Rewards should plateau near 0.95-0.99
- Position error stable around 0.05-0.10m
- Performance should match or exceed PID

### 4. If Training Fails
- Check TensorBoard for:
  - Actor/Critic losses exploding
  - Reward not increasing after 200k steps
  - Episode length dropping (early termination)
  
- Solutions:
  - Reduce learning rate: `3e-4 → 1e-4`
  - Increase `learning_starts`: `10k → 20k`
  - Check environment termination logic
  - Verify observation normalization

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'gym_pybullet_drones'"
**Solution:** The script automatically adds the path. Ensure gym-pybullet-drones folder exists at `C:\Projects\masters-thesis\gym-pybullet-drones`

### Issue: Training very slow
**Solution:**
- Reduce GUI rendering during training (already done: `gui=False`)
- Use fewer parallel environments: `NUM_ENVS = 4 → 2`
- Reduce evaluation frequency: `EVAL_FREQ = 10k → 50k`

### Issue: GPU not detected
**Solution:**
- Check PyTorch installation: `python -c "import torch; print(torch.cuda.is_available())"`
- Training will automatically use CPU if GPU unavailable (slower but works)

### Issue: Model evaluation fails
**Solution:**
- Ensure `best_model.zip` exists in `logs/sac_spiral/best_model/`
- If not, use a checkpoint: modify `MODEL_PATH` in `evaluate_sac_spiral.py`
- Check `vec_normalize.pkl` exists for observation normalization

## Expected Results

After successful training (1M timesteps):

**SAC Performance Target:**
- Mean position error: 0.03-0.08m (comparable to PID's 0.048m)
- Mean reward: 0.95-0.99 (comparable to PID's 0.99)
- Attitude stability: ±5° (comparable to PID's ±4°)
- Smooth control: Less oscillation than PID

**Training Time (estimated):**
- With GPU: ~4-6 hours
- With CPU: ~12-16 hours

## Next Steps

1. **Hyperparameter Tuning:**
   - Adjust learning rate
   - Try different network architectures
   - Modify reward weights

2. **Compare with DHP:**
   - Run DHP training on same environment
   - Generate comparison plots
   - Analyze convergence speed

3. **Extended Evaluation:**
   - Test on multiple random seeds
   - Evaluate robustness to disturbances
   - Test generalization to different trajectories

## Citation

If you use this code, please cite:
```
@mastersthesis{your_thesis,
  author = {Your Name},
  title = {Comparison of SAC and DHP for Quadrotor Control},
  school = {Your University},
  year = {2025}
}
```
