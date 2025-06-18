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

# Set logging level to ERROR to suppress warning-level logs
logging.basicConfig(level=logging.ERROR)

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
    if len(points) == 1:
        sequence = [f"{names[0]} (0 km)", "Total Distance: 0.0 km"]
        return sequence, [points[0]], [0.0], []
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

st.title("RK - Societies Clustering and Delivery Sequencing Tool")

def_lat = 12.989708618922553
def_long = 77.78625342251868
source_lat = st.sidebar.number_input("Depot Latitude", value=def_lat, format="%.8f")
source_long = st.sidebar.number_input("Depot Longitude", value=def_long, format="%.8f")

st.sidebar.subheader("Cost Settings")
van_cost = st.sidebar.number_input("Van Cost (₹)", value=834, step=1)
cee_cost = st.sidebar.number_input("CEE Cost (₹)", value=333, step=1)

st.subheader("Download Template")
template = pd.DataFrame({"Society ID": [], "Society": [], "Latitude": [], "Longitude": [], "Orders": [], "Hub ID": []})
st.download_button("Download Template CSV", data=template.to_csv(index=False), file_name="society_template.csv")

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
            current_points = [[seed['Latitude'], seed['Longitude']]]
            for idx in unassigned.index[1:]:
                candidate = df.loc[idx]
                candidate_coord = (candidate['Latitude'], candidate['Longitude'])
                if cluster_orders + candidate['Orders'] <= 220 and is_within_seed_radius(seed_coord, candidate_coord):
                    test_points = current_points + [[candidate['Latitude'], candidate['Longitude']]]
                    route_distance = calculate_route_distance(test_points)
                    if route_distance <= 10:
                        cluster_members.append(idx)
                        cluster_orders += candidate['Orders']
                        current_points.append([candidate['Latitude'], candidate['Longitude']])
            if cluster_orders >= 180:
                df.loc[cluster_members, 'Cluster'] = cluster_id
                cluster_id += 1
            hub_df = df[(df['Cluster'] == -1) & (df['Hub ID'] == hub)]

    # Micro cluster logic (non-destructive, 2km + 120 orders + 10km route distance from depot)
    micro_cluster_id = 10000
    df['Micro_Cluster'] = -1

    for hub in df['Hub ID'].unique():
        hub_df = df[(df['Cluster'] == -1) & (df['Hub ID'] == hub)]
        while not hub_df.empty:
            unassigned = hub_df.copy()
            seed_idx = unassigned.index[0]
            seed = df.loc[seed_idx]
            seed_coord = (seed['Latitude'], seed['Longitude'])
            cluster_members = [seed_idx]
            cluster_orders = seed['Orders']
            current_points = [[source_lat, source_long], [seed['Latitude'], seed['Longitude']]]

            for idx in unassigned.index[1:]:
                candidate = df.loc[idx]
                candidate_coord = [candidate['Latitude'], candidate['Longitude']]
                if (cluster_orders + candidate['Orders'] <= 120 and
                    is_within_seed_radius(seed_coord, candidate_coord) and
                    calculate_distance_km(source_lat, source_long, candidate_coord[0], candidate_coord[1]) <= 10):

                    test_points = current_points + [candidate_coord]
                    route_distance = calculate_route_distance(test_points)
                    if route_distance <= 10:
                        cluster_members.append(idx)
                        cluster_orders += candidate['Orders']
                        current_points.append(candidate_coord)

            df.loc[cluster_members, 'Micro_Cluster'] = micro_cluster_id
            micro_cluster_id += 1
            hub_df = df[(df['Cluster'] == -1) & (df['Hub ID'] == hub)]

    st.dataframe(df)

    st.subheader("Micro-Cluster Route Maps and Cost Summary")
    for micro_id in sorted(df['Micro_Cluster'].unique()):
        if micro_id == -1:
            continue
        mcluster_df = df[df['Micro_Cluster'] == micro_id]
        st.markdown(f"### Micro-Cluster {micro_id} (Orders: {mcluster_df['Orders'].sum()})")
        seq, points, dists, _ = get_delivery_sequence(mcluster_df, source_lat, source_long)
        st.text("\n".join(seq))

        m = folium.Map(location=[source_lat, source_long], zoom_start=13)
        folium.Marker([source_lat, source_long], popup="Depot", icon=folium.Icon(color="red")).add_to(m)

        for i, row in mcluster_df.iterrows():
            Marker([row['Latitude'], row['Longitude']], popup=row['Society'],
                   icon=folium.Icon(color='blue')).add_to(m)
        if len(points) > 1:
            PolyLine(points, color="blue", weight=2.5).add_to(m)
        st_folium(m, width=725)

        total_orders = mcluster_df['Orders'].sum()
        cost = van_cost + cee_cost
        cpo = round(cost / total_orders, 2) if total_orders else 0
        st.write(f"Total Orders: {total_orders}, Cost: ₹{cost}, Cost per Order: ₹{cpo}")
