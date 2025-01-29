import numpy as np
from windy.env import SailingEnv
from windy.wind import WindDataProcessor
from windy.rl_algs.simple_dqn import DQNAgent
import torch
import os
from datetime import datetime

# Fix OpenMP runtime error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def discretize_action(action: int, num_actions: int = 36) -> np.ndarray:
    """Convert discrete action to continuous heading."""
    heading = (action * (360.0 / num_actions)) % 360
    return np.array([heading], dtype=np.float32)


def train_dqn(
    env: SailingEnv,
    num_episodes: int = 1000,
    max_steps: int = 200,
    eval_freq: int = 10,
    save_freq: int = 100,
    render_eval: bool = True,
):
    """Train DQN agent on the sailing environment.

    Args:
        env: Sailing environment
        num_episodes: Number of episodes to train
        max_steps: Maximum steps per episode
        eval_freq: Frequency of evaluation episodes
        save_freq: Frequency of saving the model
        render_eval: Whether to render evaluation episodes
    """
    # Create agent
    num_actions = 36  # Discretize heading into 36 bins (10 degrees each)
    state_dim = env.observation_space.shape[0]
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=num_actions,
        hidden_dims=[256, 256],
        learning_rate=1e-4,  # Reduced learning rate for stability
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,  # Increased minimum exploration
        epsilon_decay=0.999,  # Slower decay rate
        buffer_size=100000,
        batch_size=128,  # Increased batch size
        target_update_freq=500,  # Less frequent target updates
    )

    # Create directories for saving
    save_dir = os.path.join("models", datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(save_dir, exist_ok=True)

    # Training loop
    best_reward = float("-inf")
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = 0

        for step in range(max_steps):
            # Select and perform action
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(
                discretize_action(action, num_actions)
            )
            done = terminated or truncated

            # Store transition and optimize
            agent.replay_buffer.push(state, action, reward, next_state, done)
            loss = agent.update()

            state = next_state
            episode_reward += reward
            if loss is not None:
                episode_loss += loss

            if done:
                break

        # Log training metrics
        avg_loss = episode_loss / (step + 1)
        agent.writer.add_scalar("train/reward", episode_reward, episode)
        agent.writer.add_scalar("train/avg_loss", avg_loss, episode)
        agent.writer.add_scalar("train/epsilon", agent.epsilon, episode)

        print(
            f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Avg Loss = {avg_loss:.4f}, Epsilon = {agent.epsilon:.2f}"
        )

        # Evaluate
        if (episode + 1) % eval_freq == 0:
            eval_reward = evaluate_agent(env, agent, num_actions, render=render_eval)
            agent.writer.add_scalar("eval/reward", eval_reward, episode)
            print(f"Evaluation Reward = {eval_reward:.2f}")

            # Save best model
            if eval_reward > best_reward:
                best_reward = eval_reward
                agent.save(os.path.join(save_dir, "best_model.pth"))

        # Save periodic checkpoints
        if (episode + 1) % save_freq == 0:
            agent.save(os.path.join(save_dir, f"model_ep{episode+1}.pth"))

    # Save final model
    agent.save(os.path.join(save_dir, "final_model.pth"))
    return agent


def evaluate_agent(
    env: SailingEnv,
    agent: DQNAgent,
    num_actions: int,
    num_episodes: int = 5,
    render: bool = True,
) -> float:
    """Evaluate the agent's performance."""
    eval_rewards = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            with torch.no_grad():
                action = agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(
                discretize_action(action, num_actions)
            )
            done = terminated or truncated
            episode_reward += reward

        if render:
            env.render()
        eval_rewards.append(episode_reward)

    return np.mean(eval_rewards)


if __name__ == "__main__":
    # Load wind data
    data_path = "wind_data/2024-01-01_2024-01-31_50.0_40.0_-5.0_-15.0_ERA5_data.grib"
    wind_data = WindDataProcessor(data_path)

    # Create environment
    env = SailingEnv(
        wind_data=wind_data,
        render_mode="folium",
        initial_lat=45.0,
        initial_lon=-10.0,
        initial_heading=90.0,
    )

    # Train agent
    agent = train_dqn(env, num_episodes=10_000, render_eval=True)

    # Close environment
    env.close()
