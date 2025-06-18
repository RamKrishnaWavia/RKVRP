import streamlit as st
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
import folium
from streamlit_folium import st_folium
from folium import PolyLine, Marker
from io import StringIO
import random
import logging

logging.basicConfig(level=logging.ERROR)

def calculate_route_distance(route):
    distance = 0.0
    for i in range(len(route) - 1):
        distance += calculate_distance_km(route[i][0], route[i][1], route[i+1][0], route[i+1][1])
    return distance

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

def get_delivery_sequence(cluster_df, depot_lat, depot_long):
    points = cluster_df[['Latitude', 'Longitude']].values.tolist()
    names = cluster_df['Society'].tolist()
    if len(points) == 1:
        sequence = [f"{names[0]} (0 km)", "Total Distance: 0.0 km"]
        return sequence, [points[0]], [0.0], 0.0, ""
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

    path_summary = []
    for i in range(len(order) - 1):
        dist = calculate_distance_km(order[i][0], order[i][1], order[i+1][0], order[i+1][1])
        path_summary.append(f"{names[i]} -> {names[i+1]} ({dist} km)")
    delivery_path = " | ".join(path_summary)
    sequence.append("Delivery Path: " + delivery_path)

    return sequence, order, distances, total_seq_distance, delivery_path

def is_within_seed_radius(seed_coord, coord, max_dist_km=2.0):
    return calculate_distance_km(seed_coord[0], seed_coord[1], coord[0], coord[1]) <= max_dist_km

st.sidebar.header("Depot Settings")
depot_lat = st.sidebar.number_input("Depot Latitude", value=12.9716, format="%.6f")
depot_long = st.sidebar.number_input("Depot Longitude", value=77.5946, format="%.6f")

st.sidebar.header("Micro Cluster Cost Settings")
micro_van_cost = st.sidebar.number_input("Van Cost (₹) [Micro]", value=500)
micro_cee_cost = st.sidebar.number_input("CEE Cost (₹) [Micro]", value=167)

st.sidebar.header("Cluster View")
main_cluster_id = st.sidebar.text_input("Select Main Cluster ID")
micro_cluster_id = st.sidebar.text_input("Select Micro Cluster ID")

st.header("Cluster Map Display")
st.text("Map and cluster outputs will appear here based on selected Cluster ID")

# Example Data (replace with actual logic in production)
data = {
    "Cluster ID": ["M1", "M1", "MIC1", "MIC1", "UN1"],
    "Society": ["Soc A", "Soc B", "Soc C", "Soc D", "Soc E"],
    "Latitude": [12.9716, 12.9725, 12.9735, 12.9740, 12.9750],
    "Longitude": [77.5946, 77.5955, 77.5965, 77.5970, 77.5980],
    "Orders": [60, 70, 40, 60, 50],
    "Cluster Type": ["main", "main", "micro", "micro", "unclustered"]
}
df = pd.DataFrame(data)

summary_rows = []
for cluster_id in df['Cluster ID'].unique():
    cluster_data = df[df['Cluster ID'] == cluster_id]
    cluster_type = cluster_data['Cluster Type'].iloc[0]
    cost = 0
    if cluster_type == "main":
        cost = 35000 / max(sum(cluster_data['Orders']), 1)
    elif cluster_type == "micro":
        cost = (micro_van_cost + micro_cee_cost) / max(sum(cluster_data['Orders']), 1)

    sequence, route_points, dists, total_dist, delivery_path = get_delivery_sequence(cluster_data, depot_lat, depot_long)

    for i, row in cluster_data.iterrows():
        summary_rows.append({
            "Cluster ID": row['Cluster ID'],
            "Society": row['Society'],
            "Latitude": row['Latitude'],
            "Longitude": row['Longitude'],
            "Orders": row['Orders'],
            "Cluster Type": cluster_type,
            "Total Cost": round(cost * row['Orders'], 2),
            "Delivery Path": delivery_path,
            "Total Distance (km)": total_dist
        })

summary_df = pd.DataFrame(summary_rows)
st.download_button("Download Cluster Summary", summary_df.to_csv(index=False), file_name="cluster_summary.csv")

# Display Main Cluster Map
if main_cluster_id:
    main_df = df[(df['Cluster ID'] == main_cluster_id) & (df['Cluster Type'] == 'main')]
    if not main_df.empty:
        main_map = folium.Map(location=[depot_lat, depot_long], zoom_start=14)
        sequence, path_points, dists, total_dist, path_summary = get_delivery_sequence(main_df, depot_lat, depot_long)
        for i, row in main_df.iterrows():
            folium.Marker([row['Latitude'], row['Longitude']], tooltip=row['Society']).add_to(main_map)
        if len(path_points) > 1:
            folium.PolyLine(path_points, color="blue", weight=3).add_to(main_map)
        st.subheader("Main Cluster Map")
        st_folium(main_map, width=700, height=500)

# Display Micro Cluster Map
if micro_cluster_id:
    micro_df = df[(df['Cluster ID'] == micro_cluster_id) & (df['Cluster Type'] == 'micro')]
    if not micro_df.empty:
        micro_map = folium.Map(location=[depot_lat, depot_long], zoom_start=14)
        sequence, path_points, dists, total_dist, path_summary = get_delivery_sequence(micro_df, depot_lat, depot_long)
        for i, row in micro_df.iterrows():
            folium.Marker([row['Latitude'], row['Longitude']], tooltip=row['Society']).add_to(micro_map)
        if len(path_points) > 1:
            folium.PolyLine(path_points, color="green", weight=3).add_to(micro_map)
        st.subheader("Micro Cluster Map")
        st_folium(micro_map, width=700, height=500)

st.dataframe(summary_df)
