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
def get_delivery_sequence(cluster_df, depot_coord=None):
    points = cluster_df[['Latitude', 'Longitude']].values.tolist()
    names = cluster_df['Society'].tolist()

    if len(points) <= 1:
        return names, points

    visited = [False] * len(points)
    sequence = []
    order = []

    if depot_coord is None:
        current_index = 0
    else:
        min_dist = float('inf')
        current_index = 0
        for i, pt in enumerate(points):
            dist = great_circle(depot_coord, pt).km
            if dist < min_dist:
                min_dist = dist
                current_index = i

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

st.title("RK - Societies Delivery Clustering Tool")

# Sidebar settings for Supply DC coordinates
st.sidebar.header("Settings")
dc_lat = st.sidebar.number_input("Supply DC Latitude", value=12.9716, format="%.6f")
dc_lon = st.sidebar.number_input("Supply DC Longitude", value=77.5946, format="%.6f")
dc_coord = (dc_lat, dc_lon)

# Template file download
st.subheader("Download Template")
template = pd.DataFrame({"Society ID": [], "Society": [], "Latitude": [], "Longitude": [], "Orders": [], "Hub ID": []})
st.download_button("Download Template CSV", data=template.to_csv(index=False), file_name="society_template.csv")

# File upload
st.subheader("Upload Society Data")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data", df)

    df = df.sort_values(by='Orders', ascending=False).reset_index(drop=True)
    df['Cluster'] = -1
    cluster_id = 0

    for hub in df['Hub ID'].unique():
        hub_df = df[(df['Cluster'] == -1) & (df['Hub ID'] == hub)]
        while (hub_df['Cluster'] == -1).any():
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

            df.loc[cluster_members, 'Cluster'] = cluster_id
            cluster_id += 1
            hub_df = df[(df['Cluster'] == -1) & (df['Hub ID'] == hub)]

    cluster_counts = df.groupby('Cluster').size().to_dict()
    cluster_labels = [f"Cluster {cid} ({cluster_counts[cid]} societies)" for cid in sorted(cluster_counts.keys())]
    cluster_id_map = {label: cid for label, cid in zip(cluster_labels, sorted(cluster_counts.keys()))}
    selected_label = st.sidebar.selectbox("Select Cluster ID to View Map", options=["All"] + cluster_labels)
    selected_cluster = "All" if selected_label == "All" else cluster_id_map[selected_label]

    cluster_summary = []
    cluster_map = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=12)

    color_palette = [
        "red", "blue", "green", "orange", "purple", "darkred", "darkblue", "darkgreen",
        "cadetblue", "pink", "gray", "black", "teal"
    ]

    cluster_filter = df['Cluster'].unique() if selected_cluster == "All" else [selected_cluster]
    for label in sorted(cluster_filter):
        cluster_df = df[df['Cluster'] == label]
        if cluster_df.empty:
            continue
        total_orders = cluster_df['Orders'].sum()
        coords = list(zip(cluster_df['Latitude'], cluster_df['Longitude']))
        distance_km = calculate_route_distance(coords)

        seed_coord = (cluster_df.iloc[0]['Latitude'], cluster_df.iloc[0]['Longitude'])
        max_dist_km = max(great_circle(seed_coord, (row['Latitude'], row['Longitude'])).km for _, row in cluster_df.iterrows())

        valid_cluster = 180 <= total_orders <= 220 and max_dist_km <= 2.0
        hub_color_map = {hub: color_palette[i % len(color_palette)] for i, hub in enumerate(df['Hub ID'].unique())}
        cluster_color_map = {cid: color for cid, color in zip(df['Cluster'].unique(), [
            "red", "blue", "green", "orange", "purple", "darkred", "darkblue", "darkgreen",
            "cadetblue", "pink", "gray", "black", "teal", "lightblue", "lightgreen", "beige", "brown"
        ])}
        color = cluster_color_map[label]

        delivery_sequence, route_points = get_delivery_sequence(cluster_df, depot_coord=dc_coord)

        for _, row in cluster_df.iterrows():
            folium.Circle(
                location=(row['Latitude'], row['Longitude']),
                radius=500,
                color=color,
                fill=True,
                fill_opacity=0.1,
                tooltip=f"Cluster {label}: {total_orders} Orders, {len(cluster_df)} Societies"
            ).add_to(cluster_map)

        for idx, (point, society_name) in enumerate(zip(route_points, delivery_sequence)):
            marker_label = f"{idx+1}: {society_name}"
            folium.Marker(
                location=point,
                popup=f"{marker_label}\nCluster ID: {label}",
                tooltip=marker_label,
                icon=folium.DivIcon(html=f'<div style="font-size:12px; color:{color}; font-weight:bold">{idx+1}</div>')
            ).add_to(cluster_map)

        if len(route_points) > 1:
            for i in range(len(route_points) - 1):
                dist = great_circle(route_points[i], route_points[i+1]).km
                folium.plugins.AntPath(
                    locations=[route_points[i], route_points[i+1]],
                    color=color,
                    weight=4,
                    opacity=0.9,
                    tooltip=f"{dist:.2f} km"
                ).add_to(cluster_map)

    st_data = st_folium(cluster_map, width=725)

    # Cluster summary table
    st.subheader("Cluster Summary")
    summary_df = pd.DataFrame(cluster_summary)
    st.dataframe(summary_df)

    csv = summary_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Cluster Summary CSV", data=csv, file_name="cluster_summary.csv")
