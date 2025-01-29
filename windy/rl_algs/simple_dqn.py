import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Tuple, List
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime


class DQN(nn.Module):
    """Deep Q-Network architecture."""

    def __init__(
        self, input_dim: int, output_dim: int, hidden_dims: List[int] = [128, 128]
    ):
        """Initialize the DQN.

        Args:
            input_dim: Dimension of input (observation space)
            output_dim: Dimension of output (action space)
            hidden_dims: List of hidden layer dimensions
        """
        super(DQN, self).__init__()

        # Build layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            linear = nn.Linear(prev_dim, hidden_dim)
            # Initialize weights using Xavier initialization
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.extend([linear, nn.ReLU()])
            prev_dim = hidden_dim

        # Output layer with smaller initialization
        output_layer = nn.Linear(prev_dim, output_dim)
        nn.init.xavier_uniform_(output_layer.weight, gain=0.1)
        nn.init.zeros_(output_layer.bias)
        layers.append(output_layer)

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)


class ReplayBuffer:
    """Experience replay buffer."""

    def __init__(self, capacity: int):
        """Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Add a transition to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        """Sample a batch of transitions."""
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return (
            np.array(state),
            np.array(action),
            np.array(reward),
            np.array(next_state),
            np.array(done),
        )

    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)


class DQNAgent:
    """DQN Agent implementation."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [128, 128],
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize the DQN agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: List of hidden layer dimensions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Starting value of epsilon for ε-greedy exploration
            epsilon_end: Minimum value of epsilon
            epsilon_decay: Decay rate of epsilon
            buffer_size: Size of replay buffer
            batch_size: Size of batch for training
            target_update_freq: Frequency of target network update
            device: Device to run the model on
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device
        self.step_count = 0

        # Initialize networks
        self.policy_net = DQN(state_dim, action_dim, hidden_dims).to(device)
        self.target_net = DQN(state_dim, action_dim, hidden_dims).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Setup tensorboard
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.writer = SummaryWriter(f"runs/DQN_{current_time}")

    def select_action(self, state: np.ndarray) -> int:
        """Select action using ε-greedy policy."""
        if random.random() > self.epsilon:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
        else:
            return random.randrange(self.action_dim)

    def update(self) -> float:
        """Update the policy network."""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        # Sample batch from replay buffer
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = (
            self.replay_buffer.sample(self.batch_size)
        )

        # Convert to tensors and ensure proper scaling
        state_tensor = torch.FloatTensor(state_batch).to(self.device)
        action_tensor = torch.LongTensor(action_batch).to(self.device)
        reward_tensor = (
            torch.FloatTensor(reward_batch).clamp(-10, 10).to(self.device)
        )  # Clip rewards
        next_state_tensor = torch.FloatTensor(next_state_batch).to(self.device)
        done_tensor = torch.FloatTensor(done_batch).to(self.device)

        # Compute current Q values
        current_q_values = self.policy_net(state_tensor).gather(
            1, action_tensor.unsqueeze(1)
        )

        # Compute next Q values with target network
        with torch.no_grad():
            next_q_values = self.target_net(next_state_tensor).max(1)[0]
            target_q_values = (
                reward_tensor + (1 - done_tensor) * self.gamma * next_q_values
            )

        # Compute loss with gradient clipping
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)

        self.optimizer.step()

        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return loss.item()

    def save(self, path: str):
        """Save the model."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "policy_net_state_dict": self.policy_net.state_dict(),
                "target_net_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "step_count": self.step_count,
            },
            path,
        )

    def load(self, path: str):
        """Load the model."""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]
        self.step_count = checkpoint["step_count"]
