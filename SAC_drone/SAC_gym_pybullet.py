import matplotlib
matplotlib.use('Agg')
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import psutil
import matplotlib.pyplot as plt

# ensure gym-pybullet-drones is on the path
sys.path.append("C:\\Projects\\masters-thesis\\gym-pybullet-drones")

from gym_pybullet_drones.envs.SpiralAviary import SpiralAviary
from gym_pybullet_drones.utils.enums import ActionType, Physics

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.monitor import Monitor
# -----------------------------------------------------------------------------
# 1) Custom policy with larger network and LayerNorm
# -----------------------------------------------------------------------------
class CustomSACPolicy(SACPolicy):
    def _build_mlp_extractor(self) -> None:
        super()._build_mlp_extractor()
        pol = self.mlp_extractor.policy_net
        self.mlp_extractor.policy_net = nn.Sequential(
            nn.LayerNorm(pol[0].in_features), *list(pol)
        )
        val = self.mlp_extractor.value_net
        self.mlp_extractor.value_net = nn.Sequential(
            nn.LayerNorm(val[0].in_features), *list(val)
        )
# -----------------------------------------------------------------------------
# 2) TqdmCallback: live progress bar
# -----------------------------------------------------------------------------

class TqdmCallback(BaseCallback):
    def __init__(self, total_timesteps: int, n_envs: int = 1, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.total_timesteps = total_timesteps
        self.n_envs = n_envs
        self.pbar = None

    def _on_training_start(self) -> None:
        self.pbar = tqdm(total=self.total_timesteps, desc="Train", unit="step")

    def _on_step(self) -> bool:
        self.pbar.update(self.n_envs)
        return True

    def _on_training_end(self) -> None:
        self.pbar.close()

# -----------------------------------------------------------------------------
# 3) Simple console/file logger (no TensorBoard)
# -----------------------------------------------------------------------------
class TrainingLoggerCallback(BaseCallback):
    def __init__(self, log_file="training_log.txt", verbose=0):
        super().__init__(verbose=verbose)
        self.episode_count = 0  # Track total episodes
        self.reward_buffer = []  # Store rewards for averaging
        self.length_buffer = []  # Store lengths for averaging
        self.tracking_error_buffer = []  # For tracking errors
        self.log_file = log_file
        
        # Create logs directory and log file
        os.makedirs("logs", exist_ok=True)
        self.log_path = os.path.join("logs", log_file)
        
        # Write header to log file
        with open(self.log_path, "w") as f:
            f.write("Timestep,Episodes,Mean_Reward,Mean_Length,Mean_Error\n")
        
    def _on_step(self) -> bool:
        return True  # Required by BaseCallback

    def _on_rollout_end(self) -> None:
        infos = self.locals.get("infos", [])
        for info in infos:
            ep = info.get("episode")
            if ep is not None:
                self.reward_buffer.append(ep["r"])
                self.length_buffer.append(ep["l"])
                
                # Also track average tracking error if available
                if "avg_tracking_error" in info:
                    self.tracking_error_buffer.append(info["avg_tracking_error"])
                    
                self.episode_count += 1

        # Log every 1000 episodes (reduced frequency)
        if self.episode_count >= 1000:
            mean_reward = np.mean(self.reward_buffer[-100:])
            mean_length = np.mean(self.length_buffer[-100:])
            mean_error = np.mean(self.tracking_error_buffer[-100:]) if self.tracking_error_buffer else 0.0
            
            # File output
            try:
                with open(self.log_path, "a") as f:
                    f.write(f"{self.num_timesteps},{len(self.reward_buffer)},{mean_reward:.4f},"
                           f"{mean_length:.2f},{mean_error:.6f}\n")
            except Exception as e:
                print(f"Warning: Could not write to log file: {e}")
            
            # Reset counter (keep buffer for partial logging)
            self.episode_count = 0

# -----------------------------------------------------------------------------
# 4) Save Best Models based on Training Performance
# -----------------------------------------------------------------------------
class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq=1000, log_dir="./logs/", verbose=1):
        super().__init__(verbose=verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_training_model")
        self.best_mean_reward = -np.inf
        self.best_reward_window = []  # Store recent rewards
        
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Get episode rewards from infos if available
            infos = self.locals.get("infos", [])
            for info in infos:
                ep = info.get("episode")
                if ep is not None:
                    self.best_reward_window.append(ep["r"])
                    # Keep only the last 100 rewards
                    if len(self.best_reward_window) > 100:
                        self.best_reward_window.pop(0)
            
            # Calculate mean reward if we have enough data
            if len(self.best_reward_window) >= 10:  # At least 10 episodes
                mean_reward = np.mean(self.best_reward_window)
                
                # Only print when we find a new best model
                if mean_reward > self.best_mean_reward:
                    if self.verbose > 0:
                        print(f"🏆 New best training model! Reward: {mean_reward:.2f} (was {self.best_mean_reward:.2f})")
                        print(f"   Saving to {self.save_path}")
                    self.model.save(f"{self.save_path}/model_{self.num_timesteps}")
                    self.best_mean_reward = mean_reward
        
        return True

# -----------------------------------------------------------------------------
# 5) Setup parameters
# -----------------------------------------------------------------------------
TOTAL_TIMESTEPS = 4_000_000  # Double the training time for better learning
NUM_ENVS = 8  # More parallel environments for stability
EVAL_FREQ = 100_000
N_EVALUATIONS = 1

# reduce physics fidelity for speed
# wrap env creation to set CPU affinity per process
def make_env(worker_id: int):
    def _init():
        # Pin this worker to a single core (core == worker_id here)
        try:
            os.sched_setaffinity(0, {worker_id})
        except AttributeError:
            pass  # Windows / macOS don't support sched_setaffinity
        # Prevent OpenMP from spawning threads
        os.environ["OMP_NUM_THREADS"] = "1"
        # Only print occasionally to avoid spam
        if worker_id == 0:  # Only print from first worker
            pid = os.getpid()
            affinity = psutil.Process(pid).cpu_affinity()
            print(f"[ENV] Training environments initialized with {len(affinity)} cores")
        raw= SpiralAviary(
            gui=False,
            record=False,
            act=ActionType.RPM,
            mode="spiral",  # Use spiral mode for training
            pyb_freq=240,
            ctrl_freq=30,
            physics=Physics.PYB
        )
        return Monitor(raw)
    return _init

# evaluation env wrapper - now with configurable GUI
def make_eval_env(worker_id, gui=False):
    def _init():
        try:
            os.sched_setaffinity(0, {worker_id})
        except AttributeError:
            pass
        os.environ["OMP_NUM_THREADS"] = "1"
        return SpiralAviary(
            gui=gui,
            record=gui,  # Only record when GUI is enabled
            act=ActionType.RPM,
            mode="spiral",  # Use spiral mode for evaluation
            pyb_freq=240,
            ctrl_freq=30,
            physics=Physics.PYB
        )
    return _init

if __name__ == '__main__':
    # Create required directories
    os.makedirs("logs/SAC/Drone/best_model", exist_ok=True)
    os.makedirs("logs/tensorboard/SAC/Drone", exist_ok=True)
    os.makedirs("logs/best_training_model", exist_ok=True)
    os.makedirs("logs/final", exist_ok=True)

    # vectorized training env with subprocesses
    train_env = VecNormalize(
        SubprocVecEnv([make_env(i) for i in range(NUM_ENVS)]),
        norm_obs=True,
        norm_reward=False, # Disable reward normalization for SAC
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
        epsilon=1e-8
    )

    # evaluation callback environment - GUI disabled during training
    eval_env_cb = VecNormalize(
        SubprocVecEnv([make_eval_env(i, gui=False) for i in range(NUM_ENVS)]),
        norm_obs=True,
        norm_reward=False,  # Disable reward normalization for SAC
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
        epsilon=1e-8
    )
    eval_env_cb.training = False
    eval_env_cb.norm_reward = False

    # -------------------------------------------------------------------------
    # 6) Simple eval callback without TensorBoard
    # -------------------------------------------------------------------------
    class CustomEvalCallback(EvalCallback):
        def __init__(self, log_file="eval_log.txt"):
            super().__init__(
                eval_env=eval_env_cb,
                best_model_save_path="logs/SAC/Drone/best_model",
                log_path="logs/tensorboard/SAC_fresh",  # Same path as model tensorboard_log
                eval_freq=EVAL_FREQ,
                n_eval_episodes=N_EVALUATIONS,
                deterministic=True,
                render=False
            )
            self.eval_log_file = os.path.join("logs", log_file)
            
            # Create eval log file
            with open(self.eval_log_file, "w") as f:
                f.write("Timestep,Mean_Reward,Std_Reward,Mean_Tracking_Error\n")

        def _on_step(self) -> bool:
            if self.n_calls % self.eval_freq == 0:
                # Run evaluation
                mean_reward, std_reward = evaluate_policy(
                    self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes,
                    deterministic=self.deterministic
                )
                
                # Calculate tracking errors
                errs = []
                for _ in range(min(3, self.eval_env.num_envs)):  # Only test a few episodes
                    obs = self.eval_env.reset()
                    dones = [False]
                    step_count = 0
                    while not dones[0] and step_count < 1000:  # Limit episode length
                        action, _ = self.model.predict(obs, deterministic=True)
                        obs, _, dones, infos = self.eval_env.step(action)
                        info = infos[0]
                        step_count += 1
                        if dones[0]:
                            if "avg_tracking_error" in info:
                                errs.append(info["avg_tracking_error"])
                            break
                
                mean_error = np.mean(errs) if errs else 0.0
                
                # Console output - only for evaluation milestones
                print(f"\n📊 EVAL @ {self.num_timesteps} steps: Reward {mean_reward:.2f}±{std_reward:.2f} | Error {mean_error:.4f}")
                
                # File output
                try:
                    with open(self.eval_log_file, "a") as f:
                        f.write(f"{self.num_timesteps},{mean_reward:.4f},{std_reward:.4f},{mean_error:.6f}\n")
                except Exception as e:
                    print(f"Warning: Could not write eval log: {e}")
                
                # Check if this is the best model
                if mean_reward > getattr(self, '_best_mean_reward', -np.inf):
                    self._best_mean_reward = mean_reward
                    print(f"🎯 New best eval model! Saving to {self.best_model_save_path}")
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                
            return True

    # -------------------------------------------------------------------------
    # 7) Build & train (without TensorBoard)
    # -------------------------------------------------------------------------
    print(f"\n▶ Training for {TOTAL_TIMESTEPS} steps with {NUM_ENVS} envs…")
    
    # Get device - use CUDA if available for faster training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directories for logs WITH TensorBoard
    os.makedirs("logs/SAC/Drone/best_model", exist_ok=True)
    os.makedirs("logs/best_training_model", exist_ok=True)
    os.makedirs("logs/final", exist_ok=True)
    os.makedirs("logs/tensorboard/SAC_fresh", exist_ok=True)  # Fresh TensorBoard directory
    print("Logging to console, files, AND TensorBoard (issue resolved!)")
    
    model = SAC(
        CustomSACPolicy,
        train_env,
        verbose=0,  # Reduced verbosity
        tensorboard_log="logs/tensorboard/SAC_fresh",  # RE-ENABLE TensorBoard!
        learning_rate=3e-4,  # Slightly lower learning rate for stability
        buffer_size=2_000_000,  # Larger buffer
        learning_starts=20_000,  # Start learning earlier
        batch_size=128,  # Larger batch size
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef=0.2,  # Higher entropy coefficient for exploration
        target_update_interval=1,
        target_entropy='auto',
        policy_kwargs=dict(
            net_arch=dict(
                pi=[256, 256],  # Larger policy network
                qf=[256, 256]   # Larger Q-function network
            ),
            activation_fn=nn.Softsign # Changed to Softsign for stabilized learning
        ),
        device=device
    )

    callbacks = CallbackList([
        TqdmCallback(TOTAL_TIMESTEPS, NUM_ENVS),
        TrainingLoggerCallback(log_file="training_log.txt"),
        CustomEvalCallback(log_file="eval_log.txt"),
        SaveOnBestTrainingRewardCallback(check_freq=5000, log_dir="logs")
    ])
    
    # Start fresh training (no loading of old models)
    print("Starting fresh training from scratch...")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        log_interval=100,  # Reduced log frequency
        callback=callbacks,
        tb_log_name="SAC_Drone_Training"  # Explicit TensorBoard run name
    )

    print("✔ Training finished — saving model & stats.")
    model.save("logs/final/sac_spiral")
    train_env.save("logs/final/vec_normalize.pkl")

    # -------------------------------------------------------------------------
    # Generate Training Plots
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("GENERATING TRAINING PLOTS")
    print("=" * 80)
    
    try:
        from plot_sac_results import plot_sac_training_results
        plot_sac_training_results(log_dir="logs", output_dir="plots")
        print("✓ Training plots generated successfully!")
    except Exception as e:
        print(f"⚠ Warning: Could not generate training plots: {e}")
    
    # -------------------------------------------------------------------------
    # Generate 3D Trajectory Plots
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("GENERATING 3D TRAJECTORY PLOTS")
    print("=" * 80)
    
    try:
        from plot_sac_trajectories import evaluate_and_plot_trajectories
        evaluate_and_plot_trajectories(
            model_path="logs/SAC/Drone/best_model/best_model.zip",
            vec_normalize_path="logs/final/vec_normalize.pkl",
            output_dir="plots/trajectories",
            n_episodes=1
        )
        print("✓ Trajectory plots generated successfully!")
    except Exception as e:
        print(f"⚠ Warning: Could not generate trajectory plots: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Models saved to:")
    print(f"  - logs/final/sac_spiral.zip")
    print(f"  - logs/SAC/Drone/best_model/best_model.zip")
    print(f"  - logs/best_training_model/")
    print(f"\nTraining plots saved to:")
    print(f"  - plots/sac_training_progress.png")
    print(f"  - plots/sac_evaluation_results.png")
    print(f"  - plots/sac_training_statistics.png")
    print(f"\nTrajectory plots saved to:")
    print(f"  - plots/trajectories/episode_*_3d_trajectory.png")
    print(f"  - plots/trajectories/episode_*_position_tracking.png")
    print(f"  - plots/trajectories/episode_*_attitude_control.png")
    print(f"  - plots/trajectories/episode_*_actions_velocities.png")
    print(f"  - plots/trajectories/combined_trajectory_summary.png")
    print(f"\nTo evaluate, run:")
    print(f"  python evaluate_final_model.py")
    print(f"\nTo regenerate trajectory plots, run:")
    print(f"  python plot_sac_trajectories.py")
    print("=" * 80)

    # Uncomment below for post-training evaluation with GUI
    # -------------------------------------------------------------------------
    # # Final evaluation
    # -------------------------------------------------------------------------
    print("\n▶ Final evaluation (GUI + record)…")
    # Use only 1 environment for visualization to avoid multiple windows
    eval_env = VecNormalize.load(
        "logs/final/vec_normalize.pkl",
        SubprocVecEnv([make_eval_env(0, gui=True)])  # Only 1 env with GUI enabled
    )
    eval_env.training = False
    eval_env.norm_reward = False

    # load the best‐model checkpoint

    best_model = SAC.load("logs/SAC/Drone/best_model/best_model.zip", env=eval_env)

    mean_r, std_r = evaluate_policy(
        best_model,
        eval_env,
        n_eval_episodes=1,
        render=True,
        deterministic=True
    )
    print(f"🏁 Mean reward: {mean_r:.2f} ± {std_r:.2f}")
    
    # Also evaluate the training best model
    print("\n▶ Evaluating best training model…")
    # Find latest best training model
    best_train_models = [f for f in os.listdir("logs/best_training_model") if f.startswith("model_")]
    best_train_models.sort(
    key=lambda x: int(x.split("_")[1].split(".")[0]), reverse=True
    )
    
    if best_train_models:
        best_train_model_path = os.path.join("logs/best_training_model", best_train_models[0])
        best_train_model = SAC.load(best_train_model_path, env=eval_env)
        
        mean_r, std_r = evaluate_policy(
            best_train_model,
            eval_env,
            n_eval_episodes=1,
            render=True,
            deterministic=True
        )
        print(f"🏁 Best training model reward: {mean_r:.2f} ± {std_r:.2f}")


    # input("Press ENTER to finish…")
    eval_env.close()