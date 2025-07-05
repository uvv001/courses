import math
import pandas as pd
import plotly.express as px
from typing import Optional, Tuple

def calculate_cargo_travel_time(
    origin_coords: Tuple[float, float],
    destination_coords: Tuple[float, float],
    cruising_speed_kmh: Optional[float] = 750.0,  # Average speed for cargo planes
) -> float:
    """
    Calculate the travel time for a cargo plane between two points on Earth using great-circle distance.

    Args:
        origin_coords: Tuple of (latitude, longitude) for the starting point
        destination_coords: Tuple of (latitude, longitude) for the destination
        cruising_speed_kmh: Optional cruising speed in km/h (defaults to 750 km/h for typical cargo planes)

    Returns:
        float: The estimated travel time in hours

    Example:
        >>> # Chicago (41.8781° N, 87.6298° W) to Sydney (33.8688° S, 151.2093° E)
        >>> result = calculate_cargo_travel_time((41.8781, -87.6298), (-33.8688, 151.2093))
    """

    def to_radians(degrees: float) -> float:
        return degrees * (math.pi / 180)

    # Extract coordinates
    lat1, lon1 = map(to_radians, origin_coords)
    lat2, lon2 = map(to_radians, destination_coords)

    # Earth's radius in kilometers
    EARTH_RADIUS_KM = 6371.0

    # Calculate great-circle distance using the haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
    distance = EARTH_RADIUS_KM * c

    # Add 10% to account for non-direct routes and air traffic controls
    actual_distance = distance * 1.1

    # Calculate flight time
    # Add 1 hour for takeoff and landing procedures
    flight_time = (actual_distance / cruising_speed_kmh) + 1.0

    # Format the results
    return round(flight_time, 2)

gotham_coords = (40.7128, -74.0060)  # Latitude, Longitude

locations = [
    {"name": "Chicago Board of Trade Building (The Dark Knight)", "latitude": 41.8789, "longitude": -87.6359},
    {"name": "499-469 W Van Buren St, Chicago, IL 60607 (The Dark Knight)", "latitude": 41.876782, "longitude": -87.639278},
    {"name": "Criterion Restaurant, London (The Dark Knight)", "latitude": 51.5101, "longitude": -0.1343},
    {"name": "Ferrari Factory (Maranello, Italy)", "latitude": 44.5373, "longitude": 10.8605},
    {"name": "Lamborghini Factory (Sant'Agata Bolognese, Italy)", "latitude": 44.6632, "longitude": 11.1316},
    {"name": "McLaren Technology Centre (Woking, Surrey, England)", "latitude": 51.3192, "longitude": -0.5597},
    {"name": "Porsche Factory (Stuttgart, Germany)", "latitude": 48.8211, "longitude": 9.1735},
    {"name": "Bugatti Factory (Molsheim, France)", "latitude": 48.5409, "longitude": 7.4534},
    {"name": "Aston Martin Headquarters (Gaydon, Warwickshire, England)", "latitude": 52.1655, "longitude": -1.5595}
]

data = []
for location in locations:
    origin_coords = (location["latitude"], location["longitude"])
    travel_time = calculate_cargo_travel_time(origin_coords=origin_coords, destination_coords=gotham_coords)
    data.append({
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "name": location["name"],
        "travel_time": travel_time
    })

df = pd.DataFrame(data)

fig = px.scatter_map(df, lat="latitude", lon="longitude", text="name", color="travel_time",
                     color_continuous_scale="Viridis", size_max=15, zoom=2)

fig.write_image("debug_saved_map.png")