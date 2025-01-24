import folium
from folium import plugins
import numpy as np
from typing import List, Dict
from branca.colormap import LinearColormap
import webbrowser
import os


class FoliumPlotter:
    """Creates interactive map visualizations of vessel tracks using Folium."""

    def __init__(self, vessel_states: List[Dict], wind_data=None):
        """Initialize the Folium plotter.

        Args:
            vessel_states: List of vessel state dictionaries
            wind_data: Optional wind data for visualization
        """
        self.vessel_states = vessel_states
        self.wind_data = wind_data

        # Extract vessel track
        self.positions = [
            (state["position"][0], state["position"][1]) for state in vessel_states
        ]

        # Calculate map center
        center_lat = np.mean([pos[0] for pos in self.positions])
        center_lon = np.mean([pos[1] for pos in self.positions])

        # Create base map
        self.map = folium.Map(
            location=[center_lat, center_lon], zoom_start=6, tiles="OpenStreetMap"
        )

    def add_vessel_track(self):
        """Add vessel track to the map."""
        # Create track line
        folium.PolyLine(
            locations=self.positions, weight=2, color="red", opacity=0.8
        ).add_to(self.map)

        # Add markers for start and end positions
        folium.Marker(
            self.positions[0],
            popup="Start",
            icon=folium.Icon(color="green", icon="info-sign"),
        ).add_to(self.map)

        folium.Marker(
            self.positions[-1],
            popup="End",
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(self.map)

    def add_vessel_points(self):
        """Add points along the vessel track with speed information."""
        for state in self.vessel_states:
            pos = state["position"]
            folium.CircleMarker(
                location=[pos[0], pos[1]],
                radius=3,
                popup=f"Speed: {state['speed']:.1f} knots<br>"
                f"Heading: {state['heading']:.1f}°",
                color="blue",
                fill=True,
            ).add_to(self.map)

    def save_and_show(self, output_file="vessel_track.html"):
        """Save the map to HTML and open in browser.

        Args:
            output_file: Name of output HTML file
        """
        self.map.save(output_file)
        webbrowser.open("file://" + os.path.realpath(output_file))


def plot_vessel_track(vessel_states: List[Dict]):
    """Convenience function to create and show a vessel track plot.

    Args:
        vessel_states: List of vessel state dictionaries
    """
    plotter = FoliumPlotter(vessel_states)
    plotter.add_vessel_track()
    plotter.add_vessel_points()
    plotter.save_and_show()
