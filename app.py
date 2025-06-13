import streamlit as st
import pandas as pd
import numpy as np
from geopy.distance import great_circle
import folium
from streamlit_folium import st_folium
from folium import PolyLine
from io import StringIO
import random

# Helper to calculate route distance in km
def calculate_route_distance(route):
    distance = 0.0
    for i in range(len(route) - 1):
        distance += great_circle(route[i], route[i+1]).km
    return distance

# Helper to create delivery sequence using nearest neighbor heuristic
def get_delivery_sequence(cluster_df, source_latlon=None):
    points = cluster_df[['Latitude', 'Longitude']].values.tolist()
    names = cluster_df['Society'].tolist()

    if len(points) <= 1:
        return names, points

    visited = [False] * len(points)
    sequence = []
    order = []

    if source_latlon is not None:
        current_index = np.argmin([great_circle(source_latlon, pt).km for pt in points])
    else:
        current_index = 0

    sequence.append(names[current_index])
    order.append(points[current_index])
    visited[current_index] = True

    for _ in range(1, len(points)):
        last_point = points[current_index]
        min_dist = float('inf')
        next_index = -1
        for i in range(len(points)):
            if not visited[i]:
                dist = great_circle(last_point, points[i]).km
                if dist < min_dist:
                    min_dist = dist
                    next_index = i
        visited[next_index] = True
        sequence.append(names[next_index])
        order.append(points[next_index])
        current_index = next_index

    return sequence, order

# Check if candidate is within 2km from seed
def is_within_seed_radius(seed_coord, coord, max_dist_km=2.0):
    return great_circle(seed_coord, coord).km <= max_dist_km

# Sample test data
@st.cache_data
def get_test_data():
    data = {
        "Society ID": [1, 2, 3, 4, 5, 6],
        "Society": ["A", "B", "C", "D", "E", "F"],
        "Latitude": [12.98, 12.981, 12.982, 12.983, 12.984, 12.985],
        "Longitude": [77.785, 77.786, 77.787, 77.788, 77.789, 77.790],
        "Orders": [50, 60, 40, 30, 20, 10],
        "Hub ID": [1, 1, 1, 1, 1, 1]
    }
    return pd.DataFrame(data)

st.title("Milk & Grocery Delivery Clustering Tool")

# Template file download
st.subheader("Download Template")
template = pd.DataFrame({"Society ID": [], "Society": [], "Latitude": [], "Longitude": [], "Orders": [], "Hub ID": []})
st.download_button("Download Template CSV", data=template.to_csv(index=False), file_name="society_template.csv")

# File upload
st.subheader("Upload Society Data")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# Supply source input
st.sidebar.subheader("Enter Supply Source Coordinates")
supply_lat = st.sidebar.number_input("Supply Latitude", value=12.989708618922553, format="%.6f")
supply_lon = st.sidebar.number_input("Supply Longitude", value=77.78625342251868, format="%.6f")
supply_source = (supply_lat, supply_lon)

MIN_ORDER_THRESHOLD = 180

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = get_test_data()
    st.info("Using built-in test data")

st.write("Input Data", df)

# Your clustering and mapping logic follows here...

# Placeholder for map rendering and clustering logic.
# Ensure your loop and distance calculations utilize the MIN_ORDER_THRESHOLD variable where required.
