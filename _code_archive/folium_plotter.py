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

    def add_wind_bounds(self):
        """Add wind data boundaries to the map."""
        if self.wind_data is None:
            return

        # Get bounds from wind data
        north = float(self.wind_data.ds.latitude.max())
        south = float(self.wind_data.ds.latitude.min())
        east = float(self.wind_data.ds.longitude.max())
        west = float(self.wind_data.ds.longitude.min())

        # Create rectangle bounds
        bounds = [[south, west], [north, west], [north, east], [south, east]]
        folium.Polygon(
            locations=bounds,
            color="blue",
            weight=2,
            fill=True,
            fill_color="blue",
            fill_opacity=0.1,
            popup="Wind Data Bounds",
        ).add_to(self.map)

    def add_vessel_track(self):
        """Add vessel track to the map."""
        # Create track line with timestamps
        points = []
        for state in self.vessel_states:
            pos = state["position"]
            # Convert numpy.datetime64 to string
            time_str = np.datetime_as_string(state["time"], unit="m")
            points.append([pos[0], pos[1], f"Time: {time_str}"])

        # Add the track line
        line = folium.PolyLine(
            locations=[[p[0], p[1]] for p in points], weight=2, color="red", opacity=0.8
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

        # Add annotations along the track
        for i, point in enumerate(points):
            if i % 4 == 0:  # Add annotation every 4th point to avoid clutter
                folium.Popup(point[2], parse_html=True).add_to(
                    folium.CircleMarker(
                        location=[point[0], point[1]], radius=3, color="blue", fill=True
                    )
                )

    def add_vessel_points(self):
        """Add points along the vessel track with speed information."""
        for state in self.vessel_states:
            pos = state["position"]
            # Convert numpy.datetime64 to string
            time_str = np.datetime_as_string(state["time"], unit="m")

            # Create detailed popup content
            popup_content = f"""
            <div style='font-family: monospace;'>
                <b>Time:</b> {time_str}<br>
                <b>Speed:</b> {state['speed']:.1f} knots<br>
                <b>Heading:</b> {state['heading']:.1f}°<br>
                <b>Wind Speed:</b> {state.get('wind_speed', 'N/A')} knots<br>
                <b>Wind Dir:</b> {state.get('wind_direction', 'N/A')}°
            </div>
            """

            folium.CircleMarker(
                location=[pos[0], pos[1]],
                radius=3,
                popup=folium.Popup(popup_content, max_width=200),
                color="blue",
                fill=True,
            ).add_to(self.map)

    def add_wind_arrows(self):
        """Add wind arrows to the map."""
        if self.wind_data is None:
            return

        # Create wind arrow layer
        arrow_locations = []
        for state in self.vessel_states[::5]:  # Sample every 5th state to avoid clutter
            pos = state["position"]
            if "wind_speed" in state and "wind_direction" in state:
                arrow_locations.append(
                    {
                        "coordinates": [pos[0], pos[1]],
                        "wind_speed": state["wind_speed"],
                        "wind_direction": state["wind_direction"],
                    }
                )

        for loc in arrow_locations:
            plugins.BoatMarker(
                loc["coordinates"],
                heading=loc["wind_direction"],
                color="#8b0000",
                wind_speed=loc["wind_speed"],
                wind_heading=loc["wind_direction"],
            ).add_to(self.map)

    def save_and_show(self, output_file="vessel_track.html"):
        """Save the map to HTML and open in browser.

        Args:
            output_file: Name of output HTML file
        """
        # Add a time slider if we have timestamps
        if self.vessel_states and "time" in self.vessel_states[0]:
            # Convert numpy.datetime64 to string
            times = [
                np.datetime_as_string(state["time"], unit="m")
                for state in self.vessel_states
            ]

            # Add TimestampedGeoJson for animated vessel movement
            features = []
            for i, state in enumerate(self.vessel_states):
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [state["position"][1], state["position"][0]],
                    },
                    "properties": {
                        "time": times[i],
                        "popup": f"Speed: {state['speed']:.1f} knots",
                        "icon": "marker",
                        "iconstyle": {
                            "iconUrl": "https://leafletjs.com/examples/custom-icons/leaf-green.png",
                            "iconSize": [12, 12],
                        },
                    },
                }
                features.append(feature)

            plugins.TimestampedGeoJson(
                {"type": "FeatureCollection", "features": features},
                period="PT1H",
                add_last_point=True,
                auto_play=True,
                loop=True,
                max_speed=5,
                loop_button=True,
                date_options="YYYY-MM-DD HH:mm",
                time_slider_drag_update=True,
            ).add_to(self.map)

        self.map.save(output_file)
        webbrowser.open("file://" + os.path.realpath(output_file))


def plot_vessel_track(vessel_states: List[Dict], wind_data=None):
    """Convenience function to create and show a vessel track plot.

    Args:
        vessel_states: List of vessel state dictionaries
        wind_data: Optional wind data for visualization
    """
    plotter = FoliumPlotter(vessel_states, wind_data)
    plotter.add_wind_bounds()
    plotter.add_vessel_track()
    plotter.add_vessel_points()
    plotter.add_wind_arrows()
    plotter.save_and_show()
