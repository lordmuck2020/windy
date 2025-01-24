import gymnasium as gym
from sailing_gym_env import SailingEnv
from wind import WindDataProcessor

# Load wind data
data_path = "wind_data/2024-01-01_2024-01-31_50.0_40.0_-5.0_-15.0_ERA5_data.grib"
wind_data = WindDataProcessor(data_path)

# Create environment
env = SailingEnv(wind_data=wind_data, render_mode="human")

# Training loop
obs, info = env.reset()
for _ in range(48):
    action = env.action_space.sample()  # Your agent's action here
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.render()

env.close()
