import os
from datetime import datetime
import numpy as np
from stable_baselines3 import PPO, DQN, A2C, SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import torch

from windy.env import SailingEnv
from windy.wind import WindDataProcessor

# Fix OpenMP runtime error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def make_env(wind_data, render_mode=None):
    """Create a sailing environment."""

    def _init():
        env = SailingEnv(
            wind_data=wind_data,
            render_mode=render_mode,
            initial_lat=45.0,
            initial_lon=-10.0,
            initial_heading=90.0,
        )
        return env

    return _init


def train_model(
    algo_class,
    algo_kwargs: dict,
    algo_name: str,
    env,
    eval_env,
    total_timesteps=100_000,
    save_freq=1000,
    eval_freq=1000,
    n_eval_episodes=5,
):
    """Train a model using Stable Baselines 3.

    Args:
        algo_class: Algorithm class (PPO, DQN, A2C, or SAC)
        algo_kwargs: Dictionary of algorithm-specific keyword arguments
        algo_name: Name of the algorithm for logging
        env: Training environment
        eval_env: Evaluation environment
        total_timesteps: Total timesteps to train for
        save_freq: How often to save the model
        eval_freq: How often to evaluate the model
        n_eval_episodes: Number of episodes for evaluation
    """
    # Create log directory
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("logs", algo_name, current_time)
    os.makedirs(log_dir, exist_ok=True)

    # Create callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_dir, "best_model"),
        log_path=log_dir,
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        n_eval_episodes=n_eval_episodes,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=os.path.join(log_dir, "checkpoints"),
        name_prefix="model",
    )

    # Create and train the model
    model = algo_class(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        device="cuda" if torch.cuda.is_available() else "cpu",
        **algo_kwargs,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    # Save the final model
    model.save(os.path.join(log_dir, "final_model"))

    return model, log_dir


def main():
    # Load wind data
    data_path = "wind_data/2024-01-01_2024-01-31_50.0_40.0_-5.0_-15.0_ERA5_data.grib"
    wind_data = WindDataProcessor(data_path)

    # Create environments
    env = DummyVecEnv([make_env(wind_data)])
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
        epsilon=1e-8,  # Prevent division by zero
    )

    # Create separate environment for evaluation
    eval_env = DummyVecEnv([make_env(wind_data, render_mode="folium")])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
        epsilon=1e-8,  # Prevent division by zero
        training=False,  # Don't update normalization statistics during evaluation
    )

    # Dictionary of algorithms to try
    algorithms = {
        "PPO": (
            PPO,
            {
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.01,
                "max_grad_norm": 0.5,  # Add gradient clipping
                "policy_kwargs": dict(
                    activation_fn=torch.nn.Tanh,  # Change to Tanh for bounded outputs
                    net_arch=dict(pi=[64, 64], vf=[64, 64]),
                    normalize_images=False,
                    log_std_init=-2.0,  # Add initial log std
                ),
                "normalize_advantage": True,  # Normalize advantages
            },
        ),
        # "A2C": (A2C, {"learning_rate": 7e-4}),
        # "SAC": (SAC, {"learning_rate": 3e-4, "batch_size": 256, "buffer_size": 1000000}),
    }

    # Train each algorithm
    for algo_name, (algo_class, algo_kwargs) in algorithms.items():
        print(f"\nTraining {algo_name}...")
        model, log_dir = train_model(
            algo_class=algo_class,
            algo_kwargs=algo_kwargs,
            algo_name=algo_name,
            env=env,
            eval_env=eval_env,
            total_timesteps=1_000_000,
            save_freq=10000,  # Save more frequently
            eval_freq=5000,  # Evaluate more frequently
            n_eval_episodes=3,
        )
        print(f"Training complete. Model saved in {log_dir}")

        # Save the normalization statistics
        env.save(os.path.join(log_dir, "vec_normalize.pkl"))


if __name__ == "__main__":
    main()
