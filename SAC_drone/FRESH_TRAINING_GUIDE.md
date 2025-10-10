# Fresh SAC Training - Quick Start Guide

## Step 1: Clear Old Data
Run the cleanup script to delete all previous training data:
```bash
clear_and_restart.bat
```

This will delete:
- All old model checkpoints
- Training logs
- TensorBoard logs
- VecNormalize statistics

## Step 2: Verify Environment
Your current setup:
- ✅ SpiralAviary.py with **38-dim observation space**
  - state[16] + action[4] + ref_pos[3] + target_vel[3] + ref_att[3] + delta_pos[3] + delta_vel[3] + delta_att[3]
- ✅ Reward function with 20× scale (suitable for SAC)
- ✅ 25-second episodes (750 steps at 30Hz)

## Step 3: Start Training
```bash
python SAC_gym_pybullet.py
```

Training configuration:
- Total timesteps: 4,000,000
- Parallel environments: 4
- Learning rate: 3e-4
- Buffer size: 2,000,000
- Network: 256×256 (policy & Q-function)

Expected training time: ~6 hours (depends on CPU)

## Step 4: Monitor Progress
### TensorBoard (recommended)
```bash
tensorboard --logdir=logs/tensorboard/SAC_fresh
```
Then open: http://localhost:6006

### Log Files
- `logs/training_log.txt` - Episode rewards and lengths
- `logs/eval_log.txt` - Evaluation metrics every 100k steps

## Step 5: Evaluate Trained Model
After training completes:
```bash
python evaluate_final_model.py
```

This will:
- Load the best model from `logs/SAC/Drone/best_model/`
- Run 5 evaluation episodes with GUI
- Record videos to `results/` directory
- Display performance statistics

## Troubleshooting

### Issue: "Observation space mismatch"
**Solution**: Make sure you ran `clear_and_restart.bat` to remove old models with different observation spaces.

### Issue: Training very slow
**Solution**: Check CPU usage. If low, increase `NUM_ENVS` in `SAC_gym_pybullet.py` (line 133).

### Issue: Rewards not improving
**Solution**: 
1. Check TensorBoard for learning curves
2. Verify episodes are completing full 25 seconds (750 steps)
3. Check `logs/training_log.txt` for mean rewards over time

## Expected Results

After successful training:
- **Episode length**: 750 steps (25 seconds) consistently
- **Mean reward**: 12,000 - 15,000 (with 20× scale factor)
- **Tracking error**: < 0.1m average
- **Completion rate**: 100% (no premature truncations)

## Saved Models

After training, you'll have:
1. `logs/final/sac_spiral.zip` - Final model at 4M steps
2. `logs/SAC/Drone/best_model/best_model.zip` - Best evaluation model
3. `logs/best_training_model/model_XXXXX.zip` - Best training checkpoint
4. `logs/final/vec_normalize.pkl` - Observation normalization stats

All models are compatible with 38-dim observation space.
