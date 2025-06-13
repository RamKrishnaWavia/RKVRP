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
        return names, points, []
    visited = [False] * len(points)
    sequence = []
    order = []
    distances = []
    current_index = 0  # Start from the first society
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
        sequence.append(f"{names[next_index]} ({min_dist} km)")
        distances.append(min_dist)
        current_point = points[next_index]
        order.append(current_point)

    total_seq_distance = round(sum(distances), 2)
    sequence.append(f"Total Distance: {total_seq_distance} km")
    return sequence, order, distances

# Check if candidate is within 2km from seed
def is_within_seed_radius(seed_coord, coord, max_dist_km=2.0):
    return calculate_distance_km(seed_coord[0], seed_coord[1], coord[0], coord[1]) <= max_dist_km

st.title("RK - Societies Delivery Clustering Tool")

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
        if attempt == max_attempts:
            st.warning(f"Cluster creation hit max attempts for Hub ID {hub}. Remaining unassigned societies may exist.")

    cluster_summary = []

    for cluster in sorted(df['Cluster'].unique()):
        cluster_df = df[df['Cluster'] == cluster]
        hub_id = cluster_df['Hub ID'].iloc[0]
        society_ids = list(cluster_df['Society ID'])
        societies = list(cluster_df['Society'])
        num_societies = len(cluster_df)
        total_orders = cluster_df['Orders'].sum()
        seed_coord = (cluster_df.iloc[0]['Latitude'], cluster_df.iloc[0]['Longitude'])
        max_dist = max(calculate_distance_km(seed_coord[0], seed_coord[1], row['Latitude'], row['Longitude']) for _, row in cluster_df.iterrows())
        sequence, route, _ = get_delivery_sequence(cluster_df, source_lat, source_long)
        full_route = [(source_lat, source_long)] + route + [(source_lat, source_long)]
        total_distance = calculate_route_distance(full_route)
        est_route_distance = calculate_route_distance([(source_lat, source_long)] + route)
        first_to_last_distance = calculate_distance_km(route[0][0], route[0][1], route[-1][0], route[-1][1]) if len(route) >= 2 else 0.0
        total_society_distance = calculate_route_distance(route)
        valid_cluster = 180 <= total_orders <= 220 and max_dist <= 2.0
        delivery_path = " -> ".join(sequence)
        cost_per_order = round((van_cost + cee_cost) / total_orders, 2)

        cluster_summary.append({
            "Cluster ID": cluster,
            "Hub ID": hub_id,
            "Society IDs": ", ".join(map(str, society_ids)),
            "Societies": ", ".join(societies),
            "No. of Societies": num_societies,
            "Total Orders": total_orders,
            "Total Distance (km) incl. return": round(total_distance, 2),
            "Est. Route Distance (no return, km)": round(est_route_distance, 2),
            "Dist. from First to Last Society (km)": round(first_to_last_distance, 2),
            "Total Distance Between All Societies (km)": round(total_society_distance, 2),
            "Max Distance from Seed (km)": round(max_dist, 2),
            "Valid Cluster (180 to 220 Orders & <2km)": valid_cluster,
            "Delivery Sequence": delivery_path,
            "Cost Per Order (₹)": cost_per_order
        })

    cluster_summary_df = pd.DataFrame(cluster_summary)
    st.sidebar.subheader("Select Cluster to View")
    cluster_options = cluster_summary_df.apply(lambda row: f"Cluster {row['Cluster ID']} ({row['No. of Societies']} Societies)", axis=1).tolist()
    selected_option = st.sidebar.selectbox("Choose a Cluster", cluster_options)
    selected_cluster_id = int(selected_option.split()[1])

    selected_cluster_df = df[df['Cluster'] == selected_cluster_id]
    selected_summary = cluster_summary_df[cluster_summary_df['Cluster ID'] == selected_cluster_id]
    sequence, route, distances = get_delivery_sequence(selected_cluster_df, source_lat, source_long)

    # Show cluster metrics
    st.subheader("Cluster Details Summary")
    for col, val in selected_summary.iloc[0].items():
        st.markdown(f"**{col}**: {val}")

    st.subheader(f"Map for Cluster {selected_cluster_id}")
    cluster_map = folium.Map(location=[source_lat, source_long], zoom_start=13)
    folium.Marker([source_lat, source_long], popup="Depot", icon=folium.Icon(color='green')).add_to(cluster_map)
    for i, (soc_name, coord) in enumerate(zip(sequence[:-1], route)):
        Marker(coord, popup=f"{soc_name} (S{i+1})", icon=folium.Icon(color='blue')).add_to(cluster_map)
        folium.map.Marker(
            coord,
            icon=DivIcon(
                icon_size=(150, 36),
                icon_anchor=(0, 0),
                html=f'<div style="font-size: 12pt">S{i+1}</div>'
            )
        ).add_to(cluster_map)
    PolyLine([(source_lat, source_long)] + route + [(source_lat, source_long)], color="blue", weight=2.5, opacity=1).add_to(cluster_map)
    st_folium(cluster_map, width=700, height=500)

    st.subheader("Cluster Summary Table")
    st.dataframe(selected_summary)

    # Export to CSV
    csv = cluster_summary_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Cluster Summary CSV", data=csv, file_name="cluster_summary.csv", mime='text/csv')
