import streamlit as st
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
import folium
from streamlit_folium import st_folium
from folium import PolyLine, Marker
from folium.features import DivIcon
from io import StringIO
import random
import logging

st.title("RK - Societies Delivery Clustering Tool")

# Sidebar - Logging option
log_to_file = st.sidebar.checkbox("Log unassigned cluster warnings to file")
log_to_display = st.sidebar.checkbox("Display unassigned cluster warnings")
log_file_path = "unassigned_clusters.log"
if log_to_file:
    logging.basicConfig(filename=log_file_path, filemode='a', level=logging.INFO)
else:
    logging.basicConfig(level=logging.ERROR)

log_messages = []

# Helper to calculate route distance in km
def calculate_route_distance(route):
    distance = 0.0
    for i in range(len(route) - 1):
        distance += calculate_distance_km(route[i][0], route[i][1], route[i+1][0], route[i+1][1])
    return distance

# Helper to calculate distance between two lat/long points using Haversine formula
def calculate_distance_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return round(R * c, 2)

# Optimized delivery sequence from first society using nearest neighbor heuristic
def get_delivery_sequence(cluster_df, depot_lat, depot_long):
    points = cluster_df[['Latitude', 'Longitude']].values.tolist()
    names = cluster_df['Society'].tolist()
    if len(points) <= 1:
        return names, points, [], []
    visited = [False] * len(points)
    sequence = []
    order = []
    distances = []
    inefficiencies = []
    current_index = 0
    current_point = points[current_index]
    visited[current_index] = True
    sequence.append(f"{names[current_index]} (0 km)")
    order.append(current_point)
    distances.append(0)

    for _ in range(len(points) - 1):
        min_dist = float('inf')
        next_index = -1
        for i in range(len(points)):
            if not visited[i]:
                dist = calculate_distance_km(current_point[0], current_point[1], points[i][0], points[i][1])
                if dist < min_dist:
                    min_dist = dist
                    next_index = i
        if next_index == -1:
            break
        visited[next_index] = True
        inefficiency_flag = " 🚩" if min_dist > 1.5 else ""
        sequence.append(f"{names[next_index]} ({min_dist} km){inefficiency_flag}")
        distances.append(min_dist)
        current_point = points[next_index]
        order.append(current_point)

    total_seq_distance = round(sum(distances), 2)
    sequence.append(f"Total Distance: {total_seq_distance} km")
    return sequence, order, distances, inefficiencies

# Check if candidate is within 2km from seed
def is_within_seed_radius(seed_coord, coord, max_dist_km=2.0):
    return calculate_distance_km(seed_coord[0], seed_coord[1], coord[0], coord[1]) <= max_dist_km

# Sidebar source location input
st.sidebar.subheader("Supply Source Location")
def_lat = 12.989708618922553
def_long = 77.78625342251868
source_lat = st.sidebar.number_input("Depot Latitude", value=def_lat, format="%.8f")
source_long = st.sidebar.number_input("Depot Longitude", value=def_long, format="%.8f")

# Sidebar cost settings
st.sidebar.subheader("Cost Settings")
van_cost = st.sidebar.number_input("Van Cost (₹)", value=834, step=1)
cee_cost = st.sidebar.number_input("CEE Cost (₹)", value=333, step=1)

# Template file download
st.subheader("Download Template")
template = pd.DataFrame({"Society ID": [], "Society": [], "Latitude": [], "Longitude": [], "Orders": [], "Hub ID": []})
st.download_button("Download Template CSV", data=template.to_csv(index=False), file_name="society_template.csv")

# File upload
st.subheader("Upload Society Data")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Society data uploaded successfully.")

    df = df.sort_values(by='Orders', ascending=False).reset_index(drop=True)
    df['Cluster'] = -1
    cluster_id = 0

    for hub in df['Hub ID'].unique():
        hub_df = df[(df['Cluster'] == -1) & (df['Hub ID'] == hub)]
        max_attempts = 100
        attempt = 0
        while (hub_df['Cluster'] == -1).any() and attempt < max_attempts:
            attempt += 1
            unassigned = hub_df[hub_df['Cluster'] == -1]
            seed_idx = unassigned.index[0]
            seed = df.loc[seed_idx]
            seed_coord = (seed['Latitude'], seed['Longitude'])
            cluster_members = [seed_idx]
            cluster_orders = seed['Orders']
            for idx in unassigned.index[1:]:
                candidate = df.loc[idx]
                candidate_coord = (candidate['Latitude'], candidate['Longitude'])
                if cluster_orders + candidate['Orders'] <= 220 and is_within_seed_radius(seed_coord, candidate_coord):
                    cluster_members.append(idx)
                    cluster_orders += candidate['Orders']
            if cluster_orders >= 180:
                df.loc[cluster_members, 'Cluster'] = cluster_id
                cluster_id += 1
            hub_df = df[(df['Cluster'] == -1) & (df['Hub ID'] == hub)]
        if (hub_df['Cluster'] == -1).any():
            msg = f"Cluster creation hit max attempts for Hub ID {hub}. Remaining unassigned societies may exist."
            log_messages.append(msg)
            if log_to_file:
                logging.info(msg)

    if log_to_display and log_messages:
        st.warning("\n".join(log_messages))
