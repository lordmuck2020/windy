# Windy: A Sailing Vessel Simulation and RL Environment

This package provides a sailing vessel simulation environment that can be used for reinforcement learning experiments. The environment simulates a sailing vessel's movement through a wind field, taking into account wind speed and direction.

## Features

- Realistic sailing vessel physics simulation
- Real wind data from ERA5 reanalysis
- Gymnasium-compatible environment for reinforcement learning
- Multiple visualization options (matplotlib animation, folium interactive maps)
- Land collision detection using Natural Earth data

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/windy.git
cd windy
```

2. Install the package:
```bash
pip install -e .
```

## Project Structure

```
windy/
├── wind/               # Wind data processing and management
├── vessel/             # Vessel simulation and physics
├── plotting/           # Visualization tools
├── env/                # Gymnasium environment
└── utils/              # Utility functions
```

## Usage

Basic usage example:

```python
import gymnasium as gym
from windy.env import SailingEnv
from windy.wind import WindDataProcessor

# Load wind data
data_path = "wind_data/2024-01-01_2024-01-31_50.0_40.0_-5.0_-15.0_ERA5_data.grib"
wind_data = WindDataProcessor(data_path)

# Create environment
env = SailingEnv(wind_data=wind_data, render_mode="matplotlib")

# Training loop
obs, info = env.reset()
for _ in range(48):
    action = env.action_space.sample()  # Your agent's action here
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.render()
env.close()
```

## Requirements

- Python >= 3.8
- See `setup.py` for full list of dependencies

## License

MIT License
