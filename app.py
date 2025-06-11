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
def get_delivery_sequence(cluster_df):
    points = cluster_df[['Latitude', 'Longitude']].values.tolist()
    names = cluster_df['Society'].tolist()
    visited = [False] * len(points)
    sequence = []
    order = []

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

st.title("Milk & Grocery Delivery Clustering Tool")

# Template file download
st.subheader("Download Template")
template = pd.DataFrame({"Society ID": [], "Society": [], "Latitude": [], "Longitude": [], "Orders": []})
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

    while (df['Cluster'] == -1).any():
        unassigned = df[df['Cluster'] == -1]
        seed_idx = unassigned.index[0]
        seed = df.loc[seed_idx]
        seed_coord = (seed['Latitude'], seed['Longitude'])

        cluster_members = [seed_idx]
        cluster_orders = seed['Orders']

        for idx in unassigned.index[1:]:
            candidate = df.loc[idx]
            candidate_coord = (candidate['Latitude'], candidate['Longitude'])
            if cluster_orders + candidate['Orders'] <= 230 and is_within_seed_radius(seed_coord, candidate_coord):
                cluster_members.append(idx)
                cluster_orders += candidate['Orders']

        df.loc[cluster_members, 'Cluster'] = cluster_id
        cluster_id += 1

    cluster_summary = []
    cluster_map = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=12)

    color_palette = [
        "red", "blue", "green", "orange", "purple", "darkred", "darkblue", "darkgreen",
        "cadetblue", "pink", "gray", "black", "teal"
    ]

    for label in sorted(df['Cluster'].unique()):
        cluster_df = df[df['Cluster'] == label]
        total_orders = cluster_df['Orders'].sum()
        coords = list(zip(cluster_df['Latitude'], cluster_df['Longitude']))
        distance_km = calculate_route_distance(coords)

        # Seed info
        seed_coord = (cluster_df.iloc[0]['Latitude'], cluster_df.iloc[0]['Longitude'])
        max_dist_km = max(great_circle(seed_coord, (row['Latitude'], row['Longitude'])).km for _, row in cluster_df.iterrows())

        valid_cluster = 190 <= total_orders <= 230 and max_dist_km <= 2.0
        color = color_palette[label % len(color_palette)]

        # Draw circle for each society in the cluster
        for _, row in cluster_df.iterrows():
            folium.Circle(
                location=(row['Latitude'], row['Longitude']),
                radius=500,
                color=color,
                fill=True,
                fill_opacity=0.1,
                tooltip=f"Cluster {label}: {total_orders} Orders, {len(cluster_df)} Societies"
            ).add_to(cluster_map)

        # Add markers (highlight seed separately)
        for i, row in cluster_df.reset_index().iterrows():
            if i == 0:
                folium.Marker(
                    location=[row['Latitude'], row['Longitude']],
                    popup=f"{row['Society']}\nOrders: {row['Orders']}\nCluster ID: {label} (Seed)",
                    tooltip=f"SEED: {row['Society']} ({row['Orders']} orders) - Cluster {label}",
                    icon=folium.Icon(color="darkpurple", icon='star')
                ).add_to(cluster_map)
            else:
                folium.Marker(
                    location=[row['Latitude'], row['Longitude']],
                    popup=f"{row['Society']}\nOrders: {row['Orders']}\nCluster ID: {label}",
                    tooltip=f"{row['Society']} ({row['Orders']} orders) - Cluster {label}",
                    icon=folium.Icon(color=color, icon='info-sign')
                ).add_to(cluster_map)

        # Route
        delivery_sequence, route_points = get_delivery_sequence(cluster_df)
        if len(route_points) > 1:
            PolyLine(locations=route_points, color=color, weight=4, opacity=0.9).add_to(cluster_map)

        cluster_name = str(label)

        cluster_summary.append({
            "Cluster ID": label,
            "Society IDs": ", ".join(cluster_df['Society ID'].astype(str).tolist()),
            "Societies": ", ".join(cluster_df['Society'].tolist()),
            "No. of Societies": len(cluster_df),
            "Total Orders": total_orders,
            "Total Distance (km)": round(distance_km, 2),
            "Max Distance from Seed (km)": round(max_dist_km, 2),
            "Valid Cluster (190–230 Orders & ≤2km from seed)": "Yes" if valid_cluster else "No",
            "Delivery Sequence": " → ".join(delivery_sequence)
        })

    st.subheader("Overall Cluster Map")
st_data = st_folium(cluster_map, width=725)

# Individual maps for each cluster
st.subheader("Individual Cluster Maps")
for label in sorted(df['Cluster'].unique()):
    cluster_df = df[df['Cluster'] == label]
    cluster_center = [cluster_df['Latitude'].mean(), cluster_df['Longitude'].mean()]
    individual_map = folium.Map(location=cluster_center, zoom_start=14)

    color = color_palette[label % len(color_palette)]
    seed_coord = (cluster_df.iloc[0]['Latitude'], cluster_df.iloc[0]['Longitude'])

    for _, row in cluster_df.iterrows():
        folium.Circle(
            location=(row['Latitude'], row['Longitude']),
            radius=500,
            color=color,
            fill=True,
            fill_opacity=0.1,
            tooltip=f"Cluster {label}: {cluster_df['Orders'].sum()} Orders, {len(cluster_df)} Societies"
        ).add_to(individual_map)

    for i, row in cluster_df.reset_index().iterrows():
        if i == 0:
            folium.Marker(
                location=[row['Latitude'], row['Longitude']],
                popup=f"{row['Society']}
Orders: {row['Orders']}
Cluster ID: {label} (Seed)",
                tooltip=f"SEED: {row['Society']} ({row['Orders']} orders) - Cluster {label}",
                icon=folium.Icon(color="darkpurple", icon='star')
            ).add_to(individual_map)
        else:
            folium.Marker(
                location=[row['Latitude'], row['Longitude']],
                popup=f"{row['Society']}
Orders: {row['Orders']}
Cluster ID: {label}",
                tooltip=f"{row['Society']} ({row['Orders']} orders) - Cluster {label}",
                icon=folium.Icon(color=color, icon='info-sign')
            ).add_to(individual_map)

    delivery_sequence, route_points = get_delivery_sequence(cluster_df)
    if len(route_points) > 1:
        PolyLine(locations=route_points, color=color, weight=4, opacity=0.9).add_to(individual_map)

    st.markdown(f"### Cluster {label} Map")
    st_folium(individual_map, width=725)

    summary_df = pd.DataFrame(cluster_summary)
    st.subheader("Cluster Summary")
    st.dataframe(summary_df)

    csv = summary_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Cluster Summary CSV", data=csv, file_name="cluster_summary.csv")
