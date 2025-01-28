import numpy as np
from dataclasses import dataclass
from typing import Tuple
import cartopy.feature as cfeature
from shapely.geometry import Point
import os
import geopandas as gpd
import cartopy.io.shapereader as shpreader


@dataclass
class VesselState:
    """Represents the current state of a vessel"""

    latitude: float
    longitude: float
    heading: float  # in degrees, 0 = North, 90 = East
    speed: float  # in knots


class VesselSimulator:
    """Simulates movement of a sailing vessel based on wind conditions

    Parameters:
        initial_lat: Initial latitude in degrees
        initial_lon: Initial longitude in degrees
        initial_heading: Initial vessel heading in degrees (0 = North, 90 = East) (default 0.0)
        initial_speed: Initial vessel speed in knots (default 0.0)
        hull_efficiency: Efficiency of the hull (0-1) (default 0.4)
    """

    def __init__(
        self,
        initial_lat: float,
        initial_lon: float,
        initial_heading: float = 0.0,
        initial_speed: float = 0.0,
        hull_efficiency: float = 0.4,
    ):
        """
        Initialize vessel simulator

        Args:
            initial_lat: Initial latitude in degrees
            initial_lon: Initial longitude in degrees
            initial_heading: Initial vessel heading in degrees (0 = North, 90 = East) (default 0.0)
            initial_speed: Initial vessel speed in knots (default 0.0)
            hull_efficiency: Efficiency of the hull (0-1) (default 0.4)
        """
        self.state = VesselState(
            latitude=initial_lat,
            longitude=initial_lon,
            heading=initial_heading,
            speed=initial_speed,
        )
        self.hull_efficiency = hull_efficiency

    def calculate_new_position(self, time_step: float) -> Tuple[float, float]:
        """
        Calculate new position based on current heading and speed

        Args:
            time_step: Time step in hours

        Returns:
            Tuple of (new_latitude, new_longitude)
        """
        # Convert speed from knots to degrees per hour
        # 1 degree of latitude = 60 nautical miles
        speed_deg = self.state.speed / 60.0

        # Calculate distance traveled in degrees
        distance = speed_deg * time_step

        # Calculate movement components
        heading_rad = np.radians(self.state.heading)
        dlat = distance * np.cos(heading_rad)
        # Adjust longitude movement based on latitude
        dlon = distance * np.sin(heading_rad) / np.cos(np.radians(self.state.latitude))

        # Update position
        new_lat = self.state.latitude + dlat
        new_lon = self.state.longitude + dlon

        return new_lat, new_lon

    def update_state_from_wind(
        self,
        wind_speed: float,
        wind_direction: float,
        time_step: float = 1.0,
    ) -> None:
        """
        Update vessel state based on wind conditions

        Args:
            wind_speed: Wind speed in knots
            wind_direction: Wind direction in degrees (0 = North, 90 = East)
            time_step: Time step in hours
        """
        # Simple model: vessel can sail at 45° to the wind
        # Speed is proportional to wind speed with maximum at beam reach (90° to wind)
        relative_angle = abs((wind_direction - self.state.heading + 180) % 360 - 180)

        if relative_angle < 45:  # Into the wind - no movement
            self.state.speed = 0
        else:
            # Maximum speed at beam reach (90 degrees to wind)
            # Simplified speed calculation - can be made more sophisticated
            efficiency = np.sin(np.radians(relative_angle))
            self.state.speed = wind_speed * self.hull_efficiency * efficiency

        if self.state.speed > 0:
            new_lat, new_lon = self.calculate_new_position(time_step)
            self.state.latitude = new_lat
            self.state.longitude = new_lon

    def set_heading(self, new_heading: float) -> None:
        """
        Set new vessel heading

        Args:
            new_heading: New heading in degrees (0 = North, 90 = East)
        """
        self.state.heading = new_heading % 360

    @property
    def position(self) -> Tuple[float, float]:
        """Get current position"""
        return self.state.latitude, self.state.longitude

    @property
    def heading(self) -> float:
        """Get current heading"""
        return self.state.heading

    @property
    def speed(self) -> float:
        """Get current speed"""
        return self.state.speed


class LandChecker:
    """Class to check if a point is on land.

    Attributes:
        land_geometries (list): List of land geometries from Natural Earth
        scale (str): The scale of the land feature. Options: '10m', '50m', '110m'
        data_dir (str): Directory to store/load the Natural Earth data
    """

    def __init__(self, scale: str = "110m", data_dir: str = "geo_data"):
        """Initialize the LandChecker with a specific scale.

        Args:
            scale (str): The scale of the land feature. Options: '10m', '50m', '110m'
            data_dir (str): Directory to store/load the Natural Earth data
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        # Construct the filename
        self.shapefile_path = os.path.join(data_dir, f"ne_{scale}_land.shp")

        # Download if not exists
        if not os.path.exists(self.shapefile_path):
            self._download_data(scale)

        # Load the geometries from local file
        land = gpd.read_file(self.shapefile_path)
        self.land_geometries = list(land.geometry)

    def _download_data(self, scale):
        """Download the Natural Earth data and save locally."""
        land = cfeature.NaturalEarthFeature(
            category="physical", name="land", scale=scale
        )
        # Save to shapefile
        gpd.GeoDataFrame(crs="WGS84", geometry=list(land.geometries())).to_file(
            self.shapefile_path
        )

    def is_land(self, lon: float, lat: float) -> bool:
        """Check if a point is on land.

        Args:
            lon (float): Longitude
            lat (float): Latitude

        Returns:
            bool: True if the point is on land, False otherwise
        """
        point = Point(lon, lat)
        return any(point.within(geom) for geom in self.land_geometries)


if __name__ == "__main__":
    vessel = VesselSimulator(
        initial_lat=50.0,
        initial_lon=-5.0,
        initial_heading=0.0,
        initial_speed=0.0,
    )
    print(vessel.position)

    vessel.update_state_from_wind(
        wind_speed=10.0,
        wind_direction=90.0,
    )
    print(vessel.position)

    vessel.set_heading(45.0)
    print(vessel.heading)

    vessel.update_state_from_wind(
        wind_speed=20.0,
        wind_direction=90.0,
    )
    print(vessel.position)

    # # Example usage
    # checker = LandChecker()  # Initialize once, downloads data if needed
    # print(checker.is_land(-0.127, 51.507))  # Check London
    # print(checker.is_land(0, 0))  # Check point in Atlantic Ocean
    # print(checker.is_land(-180, 0))  # Check point in Atlantic Ocean
    # print(checker.is_land(-86.158, 39.768))  # Check point in Tennessee
    # print(checker.is_land(-180, 90))  # Check point in Antarctic Ocean
    # print(checker.is_land(180, 90))  # Check point in Antarctic Ocean
