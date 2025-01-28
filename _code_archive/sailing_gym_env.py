import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any
from wind import WindDataProcessor
from vessel import VesselSimulator
from animation_plotter import AnimationPlotter
from folium_plotter import plot_vessel_track


class SailingEnv(gym.Env):
    """Sailing environment that follows gym interface

    Args:
        wind_data: WindDataProcessor instance containing wind field data
        initial_lat: Initial latitude in degrees
        initial_lon: Initial longitude in degrees
        initial_heading: Initial vessel heading in degrees (0=North, 90=East)
        render_mode: One of 'human', 'folium', 'matplotlib', or 'rgb_array'
        time_step: Time step in hours
    """

    metadata = {
        "render_modes": ["human", "folium", "matplotlib", "rgb_array"],
        "render_fps": 4,
    }

    def __init__(
        self,
        wind_data: WindDataProcessor,
        initial_lat: float = 45.0,
        initial_lon: float = -10.0,
        initial_heading: float = 90.0,
        render_mode: Optional[str] = None,
        time_step: float = 1.0,
    ):
        """Initialize the sailing environment.

        Args:
            wind_data: WindDataProcessor instance containing wind field data
            initial_lat: Initial latitude in degrees
            initial_lon: Initial longitude in degrees
            initial_heading: Initial vessel heading in degrees (0=North, 90=East)
            render_mode: One of 'human', 'folium', 'matplotlib', or 'rgb_array'
            time_step: Time step in hours
        """
        self.wind_data = wind_data
        self.initial_lat = initial_lat
        self.initial_lon = initial_lon
        self.initial_heading = initial_heading
        self.time_step = time_step

        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=0, high=360, shape=(1,), dtype=np.float32
        )  # Heading in degrees

        # Observation space: [lat, lon, heading, speed, wind_speed, wind_direction]
        # TODO: Change the observation space to match the actual data
        self.observation_space = spaces.Box(
            low=np.array([-90.0, -180.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([90.0, 180.0, 360.0, 30.0, 50.0, 360.0], dtype=np.float32),
            dtype=np.float32,
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # For rendering
        self.window = None
        self.clock = None
        self.states = []
        self.simulation_times = []

    # TODO: Change the observation space to match the actual data
    def _get_obs(self) -> np.ndarray:
        """Get current observation."""
        lat, lon = self.vessel.position
        wind_speed, wind_direction = self.wind_data.get_wind_at_position(
            lat, lon, self.current_time
        )
        return np.array(
            [
                lat,
                lon,
                self.vessel.heading,
                self.vessel.speed,
                wind_speed,
                wind_direction,
            ],
            dtype=np.float32,
        )

    def _get_info(self) -> Dict:
        """Get additional information."""
        lat, lon = self.vessel.position
        return {
            "position": (lat, lon),
            "heading": self.vessel.heading,
            "speed": self.vessel.speed,
            "time": self.current_time,
        }

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        super().reset(seed=seed)

        # Initialize vessel
        self.vessel = VesselSimulator(
            initial_lat=self.initial_lat,
            initial_lon=self.initial_lon,
            initial_heading=self.initial_heading,
        )

        # Reset time and history
        self.current_time = self.wind_data.times[0]
        self.states = []
        self.simulation_times = []

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Step the environment."""
        # Get desired heading from action
        desired_heading = float(action[0])

        # Update vessel heading
        self.vessel.set_heading(desired_heading)

        # Get wind conditions and update vessel state
        lat, lon = self.vessel.position
        wind_speed, wind_direction = self.wind_data.get_wind_at_position(
            lat, lon, self.current_time
        )

        # Update vessel state
        self.vessel.update_state_from_wind(
            wind_speed=wind_speed,
            wind_direction=wind_direction,
            time_step=self.time_step,
        )

        # Store state for rendering
        state = {
            "position": self.vessel.position,
            "heading": desired_heading,
            "speed": self.vessel.speed,
            "wind_speed": wind_speed,
            "wind_direction": wind_direction,
            "time": self.current_time,
        }
        self.states.append(state)
        self.simulation_times.append(self.current_time)

        # Update time
        self.current_time = self.current_time + np.timedelta64(
            int(self.time_step * 3600), "s"
        )

        # Get observation and info
        observation = self._get_obs()
        info = self._get_info()

        # TODO: Update reward function to be more complex
        reward = self._calculate_reward()

        # Check if episode is done (example conditions)
        terminated = False
        truncated = False

        return observation, reward, terminated, truncated, info

    def render(self):
        """Render the environment."""
        if self.render_mode == "folium":
            plot_vessel_track(self.states, wind_data=self.wind_data)
        elif self.render_mode in ["human", "matplotlib"]:
            # Create a copy of the wind data to avoid modifying the original
            wind_data_copy = WindDataProcessor(self.wind_data.file_path)

            # Find the closest available times in the wind dataset
            valid_times = []
            for sim_time in self.simulation_times:
                time_diff = np.abs(wind_data_copy.times - sim_time)
                closest_time_idx = np.argmin(time_diff)
                valid_times.append(wind_data_copy.times[closest_time_idx])

            # Select only the valid times
            wind_data_copy.ds = wind_data_copy.ds.sel(time=valid_times)

            plotter = AnimationPlotter(
                wind_data=wind_data_copy,
                stride=3,
                arrow_scale=200,
                vessel_states=self.states,
            )
            plotter.animate(interval=200)
        elif self.render_mode == "rgb_array":
            # Implement if needed for training visualization
            raise NotImplementedError()

    def _calculate_reward(self):
        """Calculate the reward

        TODO: Update reward to maximise speed while minimising end distance from the start point.
        """
        return self.vessel.speed

    def close(self):
        """Clean up environment resources."""
        if self.window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.window = None
