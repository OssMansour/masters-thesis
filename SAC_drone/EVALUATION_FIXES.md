# SAC Spiral Training - Evaluation Fixes

## Issues Fixed

### 1. **Reset Bug in SpiralAviary.py**
**Problem**: The `reset()` method was using `PYB_FREQ` (240Hz) instead of `CTRL_FREQ` (30Hz) for randomizing the starting position on the spiral.

**Location**: Line 283 in `SpiralAviary.py`

**Fix**:
```python
# Before (WRONG):
t0 = self.np_random.integers(0, int(0.1*self.EPISODE_LEN_SEC * self.PYB_FREQ))

# After (CORRECT):
t0 = self.np_random.integers(0, int(0.1*self.EPISODE_LEN_SEC * self.CTRL_FREQ))
```

**Impact**: This bug caused the starting position to be randomized with wrong scale (8x too large), potentially starting drones too far along the spiral trajectory or at invalid positions.

---

### 2. **Excessive Logging in Training**
**Problem**: `SaveBestModelCallback` was printing every time a new best model was saved, causing console spam during training.

**Location**: `train_sac_spiral.py` - `SaveBestModelCallback` class

**Fix**:
- Added `print_threshold=1.0` parameter
- Only prints when improvement is ≥1.0 reward units
- Still saves best model internally but reduces console output

**Impact**: Cleaner training logs while still tracking best model.

---

### 3. **Evaluation Speed Too Fast**
**Problem**: Evaluation ran as fast as possible, making it impossible to observe drone performance visually.

**Solution**: Created new evaluation script `evaluate_sac_spiral_realtime.py` with:
- **Real-time synchronization**: Sleeps between steps to match 30Hz control frequency (0.033s per step)
- Configurable via `REALTIME_SYNC` flag
- Progress indicators every 2 seconds

**Usage**:
```python
REALTIME_SYNC = True  # Enable real-time visualization
REALTIME_SYNC = False  # Run as fast as possible
```

---

### 4. **No Video Recording**
**Problem**: No way to record and save evaluation videos.

**Solution**: Added video recording capability in `evaluate_sac_spiral_realtime.py`:
- Uses PyBullet's `STATE_LOGGING_VIDEO_MP4`
- Saves to `logs/sac_spiral/evaluation_video.mp4`
- Configurable via `RECORD_VIDEO` flag

**Usage**:
```python
RECORD_VIDEO = True   # Record video during evaluation
RECORD_VIDEO = False  # No recording
```

---

## New Evaluation Script Features

### `evaluate_sac_spiral_realtime.py`

#### Key Features:
1. **Real-Time Synchronization** 
   - Sleeps between control steps to maintain 30Hz frequency
   - Makes drone motion observable by human eye
   - Actual step duration = control calculation + sleep time

2. **Video Recording**
   - Records evaluation as MP4 video
   - Saves to `logs/sac_spiral/evaluation_video.mp4`
   - Uses PyBullet's native video recording

3. **Progress Monitoring**
   - Prints status every 2 seconds
   - Shows: step count, time, position error, reward
   - Helps monitor evaluation progress

4. **Comprehensive Diagnostics**
   - 4-panel plot (position, reward, attitude, 3D trajectory)
   - Performance statistics (position error, attitude error, rewards)
   - Comparison table with PID baseline

5. **Proper Reward Logging**
   - Sets `norm_reward = False` during evaluation
   - Logs raw rewards (not normalized)
   - Matches PID test format for fair comparison

---

## How to Use

### 1. Run Training (if not done already)
```bash
cd c:\Projects\masters-thesis\SAC_drone
python train_sac_spiral.py
```

### 2. Run Evaluation with Real-Time Visualization
```bash
python evaluate_sac_spiral_realtime.py
```

**Configuration Options** (in script):
```python
REALTIME_SYNC = True   # Enable real-time visualization (slower)
RECORD_VIDEO = True    # Enable video recording
GUI = True             # Show PyBullet GUI
NUM_EVAL_EPISODES = 1  # Number of evaluation episodes
```

### 3. Check Outputs
After evaluation completes:
- **Video**: `logs/sac_spiral/evaluation_video.mp4`
- **Plots**: `logs/sac_spiral/sac_spiral_evaluation_plots.png`
- **Console**: Performance statistics and PID comparison

---

## Expected Behavior

### Real-Time Sync Enabled (`REALTIME_SYNC = True`)
- Each control step takes exactly 0.033 seconds (30Hz)
- 25-second episode takes ~25 seconds of real time
- Drone motion is smooth and observable
- Progress printed every 2 seconds:
  ```
  Step 0/750 | Time: 0.0s | Pos Error: 0.123m | Reward: 0.987
  Step 60/750 | Time: 2.0s | Pos Error: 0.045m | Reward: 0.995
  ...
  ```

### Video Recording Enabled (`RECORD_VIDEO = True`)
- PyBullet records entire episode
- Video saved as MP4 file
- Can be viewed with any video player
- Includes GUI rendering (camera view, trajectory lines)

### Evaluation Outputs
```
📊 Performance Metrics:
  Position Error: 0.048m ± 0.023m (max: 0.156m)
  Roll Error: 3.5° (max: 8.2°)
  Pitch Error: 4.1° (max: 9.7°)
  Mean Reward: 0.990 ± 0.015
  Episode Reward: 742.50

Comparison with PID Baseline:
------------------------------------------------------------
Metric                    | PID Baseline  | SAC Result
------------------------------------------------------------
Mean Reward               | 0.990         | 0.990
Position Error (m)        | 0.048         | 0.048
Roll Error (deg)          | ±4.0          | ±3.5
Pitch Error (deg)         | ±4.0          | ±4.1
Episode Completion        | 750/750       | 750/750
------------------------------------------------------------
```

---

## Troubleshooting

### Issue: Training shows rewards 10-19 instead of 0-1
**Possible Causes**:
1. VecNormalize accumulating episode rewards instead of per-step
2. Reward function returning unbounded values
3. Early termination causing fewer steps but high accumulated reward

**Debug Steps**:
1. Check episode length in training logs - should be 750 steps (25s × 30Hz)
2. Verify reward function returns [0, 1] per step
3. Check truncation conditions (too_far, too_tilted)

### Issue: Episodes ending early (9-13s instead of 25s)
**Cause**: Truncation conditions triggering:
- `too_far`: Drone >2.5m from origin
- `too_tilted`: Roll/pitch >±1.2 rad (±69°)

**Solutions**:
1. Check initial policy performance - may need better initialization
2. Consider more forgiving truncation thresholds during early training
3. Reduce starting noise to help initial learning

### Issue: Video file not found
**Cause**: PyBullet video recording path issue

**Fix**: Ensure path is absolute and writable:
```python
VIDEO_PATH = Path("logs/sac_spiral/evaluation_video.mp4").absolute()
```

### Issue: Real-time sync too slow/fast
**Adjust**: Change control frequency or disable sync:
```python
CTRL_FREQ = 30  # Hz (default)
REALTIME_SYNC = True  # Set False to run as fast as possible
```

---

## Summary of Changes

| File | Change | Purpose |
|------|--------|---------|
| `SpiralAviary.py` (line 283) | `PYB_FREQ` → `CTRL_FREQ` | Fix reset randomization scale |
| `train_sac_spiral.py` | Add print threshold | Reduce console spam |
| `evaluate_sac_spiral_realtime.py` | New script | Real-time viz + video recording |

---

## Next Steps

1. **Run evaluation with real-time visualization** to observe trained policy
2. **Check video recording** to verify performance
3. **Compare SAC vs PID** using the comparison table
4. **Debug training issues** if SAC performance is poor:
   - Check episode lengths in logs
   - Verify reward scaling
   - Inspect truncation triggers

---

## Performance Expectations

Based on PID baseline:
- **Position tracking**: <0.05m average error
- **Attitude stability**: ±4° roll/pitch
- **Reward**: >0.99 mean per step
- **Episode completion**: Full 750 steps (25 seconds)

SAC should match or exceed these metrics after sufficient training (~500k-1M steps).
