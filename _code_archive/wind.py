import xarray as xr
import numpy as np
import cdsapi
import xarray as xr
import logging
from typing import Union, Tuple, List, Dict, Any
import calendar
from datetime import datetime, timedelta
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ERA5DataCollector:
    def __init__(self):
        try:
            self.c = cdsapi.Client()
            # Verify cfgrib is available for xarray
            import cfgrib
        except ImportError as e:
            logger.error("cfgrib package is missing. Please install it.")
            raise ImportError("Please install cfgrib: pip install cfgrib") from e
        except Exception as e:
            logger.error(f"Failed to initialize CDS API client: {str(e)}")
            raise ConnectionError(
                "Could not connect to ERA5 database. Check ~/.cdsapirc"
            )

    def _generate_date_lists(
        self, start_date: str, end_date: str
    ) -> tuple[List[str], List[str], List[str]]:
        """Generate lists of years, months, and days for the date range.

        Args:
            start_date (str): Start date in format 'YYYY-MM-DD'
            end_date (str): End date in format 'YYYY-MM-DD'

        Returns:
            tuple: Lists of years, months, and days
        """
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        # Generate years list
        years = sorted(set([str(year) for year in range(start.year, end.year + 1)]))

        # Generate months list (all months in range, padded with zeros)
        months = []
        current = start
        while current <= end:
            if current.month not in [int(m) for m in months]:
                months.append(f"{current.month:02d}")
            current += timedelta(days=32)  # Jump roughly one month ahead
            current = current.replace(day=1)  # Reset to first day of month
        months.sort()

        # Generate days list (all possible days for the months in range)
        days = []
        for month in months:
            # Get the maximum number of days for this month
            # Using the last year in range to handle leap years if present
            max_days = calendar.monthrange(end.year, int(month))[1]
            days.extend([f"{day:02d}" for day in range(1, max_days + 1)])
        days = sorted(set(days))  # Remove duplicates and sort

        return years, months, days

    def fetch_wind_data(
        self,
        north: float,
        south: float,
        east: float,
        west: float,
        start_date: str,
        end_date: str,
        variables: List[str] = [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
        ],
        output_dir: str = "wind_data",
    ) -> Tuple[bool, Union[xr.Dataset, None]]:
        """
        Fetch marine wind data from ERA5 for a specified rectangular region.

        Easy to understand docs can be found here:
        - https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download

        Complete list of variables can be found here:
        - https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation

        Args:
            north (float): Northern boundary latitude (90 to -90)
            south (float): Southern boundary latitude (90 to -90)
            east (float): Eastern boundary longitude (180 to -180)
            west (float): Western boundary longitude (180 to -180)
            start_date (str): Start date in format 'YYYY-MM-DD'
            end_date (str): End date in format 'YYYY-MM-DD'
            output_path (str): Path to save the GRIB file
            variables (List[str]): List of variables to fetch (default: ["10m_u_component_of_wind", "10m_v_component_of_wind"])

        Returns:
            Tuple[bool, Union[xr.Dataset, None]]: Success flag and dataset
        """

        # Validate input
        try:
            self._validate_area(north, south, east, west)
            self._validate_dates(start_date, end_date)
        except ValueError as e:
            logger.error(f"Input validation failed: {str(e)}")
            return False, None

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir,
            f"{start_date}_{end_date}_{north}_{south}_{east}_{west}_ERA5_data.grib",
        )

        # Generate date lists for the request
        years, months, days = self._generate_date_lists(start_date, end_date)

        # Prepare the request parameters
        request_params = {
            "format": "grib",
            "product_type": "reanalysis",
            "variable": variables,
            "year": years,
            "month": months,
            "day": days,
            "time": [
                "00:00",
                "01:00",
                "02:00",
                "03:00",
                "04:00",
                "05:00",
                "06:00",
                "07:00",
                "08:00",
                "09:00",
                "10:00",
                "11:00",
                "12:00",
                "13:00",
                "14:00",
                "15:00",
                "16:00",
                "17:00",
                "18:00",
                "19:00",
                "20:00",
                "21:00",
                "22:00",
                "23:00",
            ],
            "area": [north, west, south, east],
        }

        try:
            logger.info(
                f"Requesting ERA5 marine data for area: "
                f"N:{north}, S:{south}, E:{east}, W:{west}"
            )
            self.c.retrieve(
                "reanalysis-era5-single-levels", request_params, output_path
            )

            dataset = xr.open_dataset(output_path, engine="cfgrib")
            logger.info("Successfully retrieved and loaded ERA5 marine data")
            return True, dataset, output_path

        except Exception as e:
            import traceback

            logger.error(f"Error retrieving ERA5 data: {str(e)}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            return False, None

    def _validate_area(
        self, north: float, south: float, east: float, west: float
    ) -> None:
        """Validate area boundary coordinates."""
        if not (-90 <= north <= 90):
            raise ValueError("North latitude must be between -90 and 90 degrees")
        if not (-90 <= south <= 90):
            raise ValueError("South latitude must be between -90 and 90 degrees")
        if not (-180 <= east <= 180):
            raise ValueError("East longitude must be between -180 and 180 degrees")
        if not (-180 <= west <= 180):
            raise ValueError("West longitude must be between -180 and 180 degrees")

        if north <= south:
            raise ValueError("North latitude must be greater than south latitude")

    def _validate_dates(self, start_date: str, end_date: str) -> None:
        """Validate date formats and ranges."""
        for date in [start_date, end_date]:
            try:
                year, month, day = map(int, date.split("-"))
                if not (1940 <= year <= 2024):
                    raise ValueError(f"Year must be between 1940 and present: {year}")
                if not (1 <= month <= 12):
                    raise ValueError(f"Month must be between 1 and 12: {month}")
                if not (1 <= day <= 31):
                    raise ValueError(f"Day must be between 1 and 31: {day}")
            except ValueError as e:
                raise ValueError(f"Invalid date format. Use YYYY-MM-DD: {str(e)}")


def check_ERA5_data():
    # Example for a location in the North Sea
    collector = ERA5DataCollector()

    start_date = "2024-01-01"
    end_date = "2024-01-31"
    output_dir = "wind_data"

    success, data, output_path = collector.fetch_wind_data(
        north=50.0,
        south=40.0,
        east=-5.0,
        west=-15.0,
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir,
    )

    if success:
        print("Data retrieved successfully!")
        print(data)
        print(output_path)
    else:
        print("Failed to retrieve data")


class LoadWindData:
    """Load wind data from a GRIB file.

    Parameters:
        file_path: Path to the GRIB file
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.ds = xr.open_dataset(self.file_path, engine="cfgrib")

    def _process_data(self):
        self.u10 = list(self.ds.data_vars)[0]  # u component
        self.v10 = list(self.ds.data_vars)[1]  # v component

        # Calculate wind speed and direction
        self.wind_speed = np.sqrt(self.ds[self.u10] ** 2 + self.ds[self.v10] ** 2)
        self.wind_direction = np.arctan2(self.ds[self.v10], self.ds[self.u10])


class WindDataProcessor:
    """Process wind data from GRIB files with spatial and temporal interpolation."""

    def __init__(self, file_path: str):
        """Initialize wind data processor.

        Args:
            file_path: Path to the GRIB file containing wind data
        """
        self.file_path = file_path
        self.ds = xr.open_dataset(file_path, engine="cfgrib")

        # Get wind component variable names
        self.u10 = list(self.ds.data_vars)[0]  # u component
        self.v10 = list(self.ds.data_vars)[1]  # v component

        # Store dimensions for quick access
        self.times = self.ds.time.values
        self.latitudes = self.ds.latitude.values
        self.longitudes = self.ds.longitude.values

        # Calculate wind speed and direction for the entire dataset
        self.wind_speed = np.sqrt(self.ds[self.u10] ** 2 + self.ds[self.v10] ** 2)
        self.wind_direction = (
            np.degrees(np.arctan2(self.ds[self.v10], self.ds[self.u10])) % 360
        )

    def get_wind_at_position(
        self, latitude: float, longitude: float, time: datetime = None
    ) -> Tuple[float, float]:
        """Get interpolated wind speed and direction at a specific position and time.

        Args:
            latitude: Latitude in degrees
            longitude: Longitude in degrees
            time: Datetime for wind data (defaults to first timestamp if None)

        Returns:
            Tuple of (wind_speed, wind_direction)
            - wind_speed in knots
            - wind_direction in degrees (0=North, 90=East)
        """
        # Default to first timestamp if none provided
        if time is None:
            time = self.times[0]
        else:
            # Convert to numpy datetime64 if needed
            if isinstance(time, datetime):
                time = np.datetime64(time)

        # Find nearest time index (for now - could be interpolated)
        time_idx = np.abs(self.times - time).argmin()

        # Create interpolator for the specific time
        u_interpolator = (
            self.ds[self.u10]
            .isel(time=time_idx)
            .interp(latitude=latitude, longitude=longitude, method="linear")
        )

        v_interpolator = (
            self.ds[self.v10]
            .isel(time=time_idx)
            .interp(latitude=latitude, longitude=longitude, method="linear")
        )

        # Get interpolated wind components
        u_wind = float(u_interpolator)
        v_wind = float(v_interpolator)

        # Calculate speed and direction
        speed = np.sqrt(u_wind**2 + v_wind**2)
        direction = np.degrees(np.arctan2(v_wind, u_wind)) % 360

        # Convert speed from m/s to knots
        speed_knots = speed * 1.944

        return speed_knots, direction

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the wind dataset.

        Returns:
            Dictionary containing:
            - shape: Dataset shape
            - times: List of timestamps
            - latitudes: List of latitudes
            - longitudes: List of longitudes
        """
        return {
            "shape": self.ds[self.u10].shape,
            "times": self.times.tolist(),
            "latitudes": self.latitudes.tolist(),
            "longitudes": self.longitudes.tolist(),
        }


if __name__ == "__main__":
    # check_ERA5_data()
    pass
