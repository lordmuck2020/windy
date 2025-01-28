import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.animation as animation
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from vessel import VesselSimulator
from wind import WindDataProcessor


class AnimationPlotter:
    def __init__(
        self,
        wind_data: WindDataProcessor,
        stride: int = 3,
        arrow_scale: int = 200,
        vessel_states: List[Dict] = None,
    ):
        """Initialize the animation plotter with wind data and parameters.

        Args:
            wind_data (WindDataProcessor): Wind data processor instance
            stride (int): Number of grid points to skip for arrow plotting
            arrow_scale (int): Scale factor for arrow length
            vessel_states (List[Dict]): List of vessel state dictionaries from simulation
        """
        self.wind_data = wind_data
        self.stride = stride
        self.arrow_scale = arrow_scale
        self.vessel_states = vessel_states
        self.vessel_line = None
        self.vessel_point = None
        self.speed_plot = None
        self.quiver_plot = None
        self.fig = None
        self.ax = None
        self.anim = None

        # WindDataProcessor already has wind_speed calculated
        self.setup_plot()

    def setup_plot(self) -> None:
        """Set up the plot figure and axes."""
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = plt.axes(projection=ccrs.PlateCarree())

        # Add map features
        self.ax.add_feature(cfeature.COASTLINE)
        self.ax.add_feature(cfeature.BORDERS)
        self.ax.gridlines(draw_labels=True)

        # Create initial speed plot
        self.speed_plot = self.ax.pcolormesh(
            self.wind_data.ds.longitude,
            self.wind_data.ds.latitude,
            self.wind_data.wind_speed.isel(time=0),
            transform=ccrs.PlateCarree(),
            cmap="viridis",
            shading="auto",
        )

        # Add colorbar
        plt.colorbar(self.speed_plot, ax=self.ax, label="Wind Speed (m/s)")

    def init_animation(self) -> None:
        """Initialize the animation frame."""
        # Remove old quiver if it exists
        if self.quiver_plot is not None:
            self.quiver_plot.remove()

        # Get strided coordinates and data
        lons = self.wind_data.ds.longitude.values[:: self.stride]
        lats = self.wind_data.ds.latitude.values[:: self.stride]
        u_data = (
            self.wind_data.ds[self.wind_data.u10]
            .isel(time=0)
            .values[:: self.stride, :: self.stride]
        )
        v_data = (
            self.wind_data.ds[self.wind_data.v10]
            .isel(time=0)
            .values[:: self.stride, :: self.stride]
        )

        # Create meshgrid for quiver
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        # Create initial quiver plot
        self.quiver_plot = self.ax.quiver(
            lon_grid,
            lat_grid,
            u_data,
            v_data,
            transform=ccrs.PlateCarree(),
            scale=self.arrow_scale,
            color="white",
            width=0.002,
            headwidth=4,
        )

        # Reset speed plot array
        self.speed_plot.set_array(self.wind_data.wind_speed.isel(time=0).values.ravel())
        self._update_title(0)

        return [self.speed_plot, self.quiver_plot]

    def update_frame(self, frame: int) -> None:
        """Update the animation frame."""
        # Update wind visualization
        self.speed_plot.set_array(
            self.wind_data.wind_speed.isel(time=frame).values.ravel()
        )

        u_data = (
            self.wind_data.ds[self.wind_data.u10]
            .isel(time=frame)
            .values[:: self.stride, :: self.stride]
        )
        v_data = (
            self.wind_data.ds[self.wind_data.v10]
            .isel(time=frame)
            .values[:: self.stride, :: self.stride]
        )

        self.quiver_plot.set_UVC(u_data, v_data)

        # Update vessel track if available
        if self.vessel_states is not None:
            # Get positions up to current frame
            positions = [state["position"] for state in self.vessel_states[: frame + 1]]
            if positions:
                lats, lons = zip(*positions)

                # Update or create the track line
                if self.vessel_line:
                    self.vessel_line.set_data(lons, lats)
                else:
                    (self.vessel_line,) = self.ax.plot(
                        lons, lats, "r-", linewidth=2, transform=ccrs.PlateCarree()
                    )

                # Update current position marker
                if self.vessel_point:
                    self.vessel_point.set_data([lons[-1]], [lats[-1]])
                else:
                    (self.vessel_point,) = self.ax.plot(
                        [lons[-1]],
                        [lats[-1]],
                        "ro",
                        markersize=8,
                        transform=ccrs.PlateCarree(),
                    )

        self._update_title(frame)

        if self.vessel_line and self.vessel_point:
            return [
                self.speed_plot,
                self.quiver_plot,
                self.vessel_line,
                self.vessel_point,
            ]
        return [self.speed_plot, self.quiver_plot]

    def _update_title(self, frame: int) -> None:
        """Update the plot title with timestamp."""
        timestamp = pd.to_datetime(
            self.wind_data.ds.time.isel(time=frame).values
        ).strftime("%Y-%m-%d %H:%M UTC")
        plt.title(f"Wind Speed and Direction - {timestamp}")

    def animate(self, interval: int = 200) -> None:
        """Create and display the animation.

        Args:
            interval (int): Animation interval in milliseconds
        """
        num_frames = len(self.wind_data.ds.time)
        self.anim = animation.FuncAnimation(
            self.fig,
            self.update_frame,
            init_func=self.init_animation,
            frames=num_frames,
            interval=interval,
            repeat=True,
        )
        plt.show()


if __name__ == "__main__":
    # Example usage
    file_path = "wind_data/2024-01-01_2024-01-31_50.0_40.0_-5.0_-15.0_ERA5_data.grib"
    wind_data = WindDataProcessor(file_path)
    plotter = AnimationPlotter(wind_data, stride=3, arrow_scale=200)
    plotter.animate(interval=200)
