from setuptools import setup, find_packages

setup(
    name="windy",
    version="0.0.1",
    description="A sailing vessel simulation and reinforcement learning environment",
    author="James Hancock",
    author_email="j.hancock354@gmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "xarray>=2023.12.0",
        "cdsapi>=0.6.1",
        "cfgrib>=0.9.10.4",
        "cartopy>=0.22.0",
        "shapely>=2.0.0",
        "geopandas>=0.14.0",
        "folium>=0.15.0",
        "branca>=0.6.0",
        "matplotlib>=3.7.0",
        "pandas>=2.1.0",
        "gymnasium>=0.29.0",
        "torch>=2.1.0",
        "tensorboard>=2.15.0",
        "stable-baselines3[extra]>=2.2.1",
    ],
    python_requires=">=3.9",
)
