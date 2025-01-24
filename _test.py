from wind import WindDataProcessor, LoadWindData
from vessel import VesselSimulator
from typing import Tuple, Dict, Callable, List
from datetime import datetime
import numpy as np
import random
from animation_plotter import AnimationPlotter


class VesselSimulation:
    """Simulates a vessel's journey over a period of time using wind data.

    This class handles the simulation of a vessel's movement through a wind field,
    tracking its position, heading, and other state variables over time.
    """

    def __init__(
        self,
        wind_data: WindDataProcessor,
        initial_lat: float,
        initial_lon: float,
        initial_heading: float = 0.0,
    ):
        """Initialize the vessel simulation.

        Args:
            wind_data: WindDataProcessor instance containing wind field data
            initial_lat: Initial latitude in degrees
            initial_lon: Initial longitude in degrees
            initial_heading: Initial vessel heading in degrees (0=North, 90=East)
        """
        self.wind_data = wind_data
        self.vessel = VesselSimulator(
            initial_lat=initial_lat,
            initial_lon=initial_lon,
            initial_heading=initial_heading,
        )

        # Store simulation history
        self.position_history: List[Tuple[float, float]] = [self.vessel.position]
        self.heading_history: List[float] = [initial_heading]
        self.speed_history: List[float] = [0.0]
        self.time_history: List[datetime] = [self.wind_data.times[0]]

    def step(
        self,
        vessel_heading: float,
        time_step: float = 1.0,
        current_time: datetime = None,
    ) -> Dict:
        """Update vessel position based on wind conditions and user-controlled heading.

        Args:
            vessel_heading: User-controlled vessel heading in degrees (0=North, 90=East)
            time_step: Time step in hours
            current_time: Current simulation time (defaults to last time if None)

        Returns:
            Dict containing current simulation state
        """
        # Get current vessel position
        lat, lon = self.vessel.position

        # Get wind conditions at current position and time
        wind_speed, wind_direction = self.wind_data.get_wind_at_position(
            lat, lon, current_time
        )

        # Set the new vessel heading
        self.vessel.set_heading(vessel_heading)

        # Update vessel state based on wind conditions
        self.vessel.update_state_from_wind(
            wind_speed=wind_speed, wind_direction=wind_direction, time_step=time_step
        )

        # Update history
        self.position_history.append(self.vessel.position)
        self.heading_history.append(vessel_heading)
        self.speed_history.append(self.vessel.speed)

        # Update time history
        if current_time is not None:
            self.time_history.append(current_time)

        # Return current state
        return {
            "position": self.vessel.position,
            "heading": vessel_heading,
            "speed": self.vessel.speed,
            "wind_speed": wind_speed,
            "wind_direction": wind_direction,
            "time": self.time_history[-1],
        }

    def run_simulation(
        self,
        duration_hours: float,
        get_heading_func: Callable[[Dict], float],
        time_step: float = 1.0,
        start_time: datetime = None,
    ) -> List[Dict]:
        """Run a simulation for a specified duration.

        Args:
            duration_hours: Duration to simulate in hours
            get_heading_func: Function that takes current state and returns desired heading
            time_step: Time step in hours
            start_time: Start time for simulation (defaults to first time in dataset)

        Returns:
            List of state dictionaries for each time step
        """
        if start_time is None:
            current_time = self.wind_data.times[0]
        else:
            current_time = start_time

        states = []
        steps = int(duration_hours / time_step)

        for _ in range(steps):
            # Get desired heading from control function
            current_state = {
                "position": self.vessel.position,
                "heading": self.vessel.heading,
                "speed": self.vessel.speed,
                "time": current_time,
            }
            desired_heading = get_heading_func(current_state)

            # Step simulation
            state = self.step(
                vessel_heading=desired_heading,
                time_step=time_step,
                current_time=current_time,
            )
            states.append(state)

            # Update time
            current_time = current_time + np.timedelta64(int(time_step * 3600), "s")

        return states

    @property
    def current_state(self) -> Dict:
        """Get the current state of the simulation."""
        return {
            "position": self.vessel.position,
            "heading": self.vessel.heading,
            "speed": self.vessel.speed,
            "time": self.time_history[-1],
        }

    @staticmethod
    def get_user_heading() -> float:
        """Prompt user for heading or generate random heading."""
        choice = (
            input(
                "Enter 'r' for random heading or a number between 0-360 for specific heading: "
            )
            .strip()
            .lower()
        )
        if choice == "r":
            return random.uniform(0, 360)
        try:
            heading = float(choice)
            if 0 <= heading <= 360:
                return heading
            else:
                print("Invalid heading. Using random heading instead.")
                return random.uniform(0, 360)
        except ValueError:
            print("Invalid input. Using random heading instead.")
            return random.uniform(0, 360)

    def run_simulation_with_animation(
        self,
        duration_hours: int = 48,
        time_step: float = 1.0,
        use_random_heading: bool = True,
    ):
        """Run simulation with interactive heading selection and animation.

        Args:
            duration_hours: Duration of simulation in hours
            time_step: Time step in hours
            use_random_heading: Whether to use random headings or get user input
        """
        states = []
        current_time = self.wind_data.times[0]
        steps = int(duration_hours / time_step)
        simulation_times = []

        for step in range(steps):
            # Get current state
            current_state = {
                "position": self.vessel.position,
                "heading": self.vessel.heading,
                "speed": self.vessel.speed,
                "time": current_time,
                "step": step,
                "total_steps": steps,
            }

            # Get heading from user or random
            print(f"\nStep {step + 1}/{steps}")
            print(f"Current position: {current_state['position']}")
            print(f"Current speed: {current_state['speed']:.2f} knots")

            desired_heading = (
                random.uniform(0, 360)
                if use_random_heading
                else self.get_user_heading()
            )

            # Step simulation
            state = self.step(
                vessel_heading=desired_heading,
                time_step=time_step,
                current_time=current_time,
            )
            states.append(state)
            simulation_times.append(current_time)

            # Update time
            current_time = current_time + np.timedelta64(int(time_step * 3600), "s")

        # Create animation with vessel track
        wind_data = LoadWindData(self.wind_data.file_path)
        # Filter wind data to match simulation duration
        wind_data.ds = wind_data.ds.sel(time=simulation_times)

        plotter = AnimationPlotter(
            wind_data=wind_data,
            stride=3,
            arrow_scale=200,
            vessel_states=states,
        )

        # Run animation
        plotter.animate(interval=200)


def example_usage():
    """Example usage of the VesselSimulation class."""
    # Load wind data
    file_path = "wind_data/2024-01-01_2024-01-31_50.0_40.0_-5.0_-15.0_ERA5_data.grib"
    wind_data = WindDataProcessor(file_path)

    # Create simulation
    sim = VesselSimulation(
        wind_data=wind_data,
        initial_lat=45.0,
        initial_lon=-10.0,
        initial_heading=90.0,  # Start heading east
    )

    # Run simulation with animation
    sim.run_simulation_with_animation(duration_hours=48, use_random_heading=True)


if __name__ == "__main__":
    example_usage()
