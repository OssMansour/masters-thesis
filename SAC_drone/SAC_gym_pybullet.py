import matplotlib
matplotlib.use('Agg')
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import psutil 

# ensure gym-pybullet-drones is on the path
sys.path.append("/home/osos/Mohamed_Masters_Thesis/gym-pybullet-drones")

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
# 3) Runtime rollout logger (with dump for TensorBoard)
# -----------------------------------------------------------------------------
class TrainingLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose=verbose)
        self.episode_count = 0  # Track total episodes
        self.reward_buffer = []  # Store rewards for averaging
        self.length_buffer = []  # Store lengths for averaging
        self.tracking_error_buffer = []  # For tracking errors
        
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
                
        # Log every 100 episodes (more frequent than before)
        if self.episode_count >= 100:
            mean_reward = np.mean(self.reward_buffer[-100:])
            mean_length = np.mean(self.length_buffer[-100:])
            
            self.logger.record("train/100_ep_rew_mean", mean_reward)
            self.logger.record("train/100_ep_len_mean", mean_length)
            
            if self.tracking_error_buffer:
                mean_error = np.mean(self.tracking_error_buffer[-100:])
                self.logger.record("train/100_ep_error_mean", mean_error)
                
            self.logger.dump(self.num_timesteps)
            
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
                
                # Make sure the folder exists
                if self.verbose > 0:
                    print(f"Step: {self.num_timesteps}")
                    print(f"Best mean training reward: {self.best_mean_reward:.2f}")
                    print(f"Current mean training reward: {mean_reward:.2f}")
                
                # New best model, save it
                if mean_reward > self.best_mean_reward:
                    if self.verbose > 0:
                        print(f"Saving new best training model to {self.save_path}")
                    self.model.save(f"{self.save_path}/model_{self.num_timesteps}")
                    self.best_mean_reward = mean_reward
        
        return True

# -----------------------------------------------------------------------------
# 5) Setup parameters
# -----------------------------------------------------------------------------
TOTAL_TIMESTEPS = 100_000  # Double the training time for better learning
NUM_ENVS = 4
EVAL_FREQ = 10_000
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
        pid = os.getpid()
        affinity = psutil.Process(pid).cpu_affinity()
        print(f"[ENV INIT] PID={pid}  pinned to cores={affinity}")
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
    # 6) Eval callback
    # -------------------------------------------------------------------------
    class CustomEvalCallback(EvalCallback):
        def __init__(self):
            super().__init__(
                eval_env=eval_env_cb,
                best_model_save_path="logs/SAC/Drone/best_model",
                log_path="logs/tensorboard/SAC/Drone",
                eval_freq=EVAL_FREQ,
                n_eval_episodes=N_EVALUATIONS,
                deterministic=True,
                render=False
            )

        def _on_step(self) -> bool:
            res = super()._on_step()
            if self.n_calls % self.eval_freq == 0:
                errs = []
                for _ in range(self.eval_env.num_envs):
                    obs = self.eval_env.reset()
                    dones = [False]
                    while not dones[0]:
                        action, _ = self.model.predict(obs, deterministic=True)
                        obs, _, dones, infos = self.eval_env.step(action)
                        info = infos[0]
                        if dones[0]:
                            if "avg_tracking_error" in info:
                                errs.append(info["avg_tracking_error"])
                            break
                if errs:
                    self.logger.record("eval/mean_tracking_error", np.mean(errs))
            return res

    # -------------------------------------------------------------------------
    # 7) Build & train
    # -------------------------------------------------------------------------
    print(f"\n▶ Training for {TOTAL_TIMESTEPS} steps with {NUM_ENVS} envs…")
    
    # Get device - use CUDA if available for faster training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = SAC(
        CustomSACPolicy,
        train_env,
        verbose=1,
        tensorboard_log="logs/tensorboard/SAC/Drone",
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
        TrainingLoggerCallback(),
        CustomEvalCallback(),
        SaveOnBestTrainingRewardCallback(check_freq=5000, log_dir="logs")
    ])
    try:
        # Load existing model if available
        model = SAC.load("logs/SAC/Drone/best_model/best_model.zip", env=train_env)
        print("✔ Loaded existing model.")
    except FileNotFoundError:
        print("No existing model found, starting training from scratch.")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        log_interval=100,
        callback=callbacks
    )

    print("✔ Training finished — saving model & stats.")
    model.save("logs/final/sac_spiral")
    train_env.save("logs/final/vec_normalize.pkl")

    # -------------------------------------------------------------------------
    # 8) Final evaluation
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
        n_eval_episodes=5,
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
            n_eval_episodes=5,
            render=True,
            deterministic=True
        )
        print(f"🏁 Best training model reward: {mean_r:.2f} ± {std_r:.2f}")


    # input("Press ENTER to finish…")
    eval_env.close()