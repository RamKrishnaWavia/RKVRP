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

def estimate_timings(route_points, start_time="03:30"):
    from datetime import datetime, timedelta
    timings = []
    current_time = datetime.strptime(start_time, "%H:%M")
    timings.append(current_time.strftime("%H:%M"))
    for i in range(1, len(route_points)):
        dist = great_circle(route_points[i-1], route_points[i]).km
        speed_kmph = 20.0  # bike speed
        minutes = (dist / speed_kmph) * 60
        current_time += timedelta(minutes=minutes)
        timings.append(current_time.strftime("%H:%M"))
    return timings

def is_within_seed_radius(seed_coord, coord, max_dist_km=2.0):
    return great_circle(seed_coord, coord).km <= max_dist_km

st.title("Milk & Grocery Delivery Clustering Tool")

# Template file download
st.subheader("Download Template")
template = pd.DataFrame({"Society ID": [], "Society": [], "Latitude": [], "Longitude": [], "Orders": [], "Hub ID": []})
st.download_button("Download Template CSV", data=template.to_csv(index=False), file_name="society_template.csv")

# File upload
st.subheader("Upload Society Data")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    selected_hub = st.sidebar.selectbox("Select Hub ID to View Map", options=["All"] + sorted(df['Hub ID'].unique()))
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

    # color map
    hub_color_map = {hub: color for hub, color in zip(df['Hub ID'].unique(), [
        "red", "blue", "green", "orange", "purple", "darkred", "darkblue", "darkgreen",
        "cadetblue", "pink", "gray", "black", "teal"
    ])}

    cluster_summary = []
    cluster_map = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=12)
    cluster_filter = df['Cluster'].unique() if selected_hub == "All" else df[df['Hub ID'] == selected_hub]['Cluster'].unique()

    for label in sorted(cluster_filter):
        cluster_df = df[df['Cluster'] == label] if selected_hub == "All" else df[(df['Cluster'] == label) & (df['Hub ID'] == selected_hub)]
        total_orders = cluster_df['Orders'].sum()
        coords = list(zip(cluster_df['Latitude'], cluster_df['Longitude']))
        distance_km = calculate_route_distance(coords)

        seed_coord = (cluster_df.iloc[0]['Latitude'], cluster_df.iloc[0]['Longitude'])
        max_dist_km = max(great_circle(seed_coord, (row['Latitude'], row['Longitude'])).km for _, row in cluster_df.iterrows())

        valid_cluster = 180 <= total_orders <= 220 and max_dist_km <= 2.0
        color = hub_color_map[cluster_df['Hub ID'].iloc[0]]

        for _, row in cluster_df.iterrows():
            folium.Circle(
                location=(row['Latitude'], row['Longitude']),
                radius=500,
                color=color,
                fill=True,
                fill_opacity=0.1,
                tooltip=f"Cluster {label}: {total_orders} Orders, {len(cluster_df)} Societies"
            ).add_to(cluster_map)

        for i, row in cluster_df.reset_index().iterrows():
            popup_text = f"{row['Society']}\\nOrders: {row['Orders']}\\nCluster ID: {label}\\nHub ID: {row['Hub ID']}"
            if i == 0:
                popup_text += " (Seed)"
                folium.Marker(
                    location=[row['Latitude'], row['Longitude']],
                    popup=popup_text,
                    tooltip=f"SEED: {row['Society']} ({row['Orders']} orders) - Cluster {label}",
                    icon=folium.Icon(color="darkpurple", icon='star')
                ).add_to(cluster_map)
            else:
                folium.Marker(
                    location=[row['Latitude'], row['Longitude']],
                    popup=popup_text,
                    tooltip=f"{row['Society']} ({row['Orders']} orders) - Cluster {label}",
                    icon=folium.Icon(color=color, icon='info-sign')
                ).add_to(cluster_map)

        delivery_sequence, route_points = get_delivery_sequence(cluster_df)
        delivery_timings = estimate_timings(route_points)
        if len(route_points) > 1:
            PolyLine(locations=route_points, color=color, weight=4, opacity=0.9).add_to(cluster_map)

        cluster_summary.append({
            "Cluster ID": label,
            "Hub ID": cluster_df['Hub ID'].iloc[0],
            "Society IDs": ", ".join(cluster_df['Society ID'].astype(str).tolist()),
            "Societies": ", ".join(cluster_df['Society'].tolist()),
            "No. of Societies": len(cluster_df),
            "Total Orders": total_orders,
            "Total Distance (km)": round(distance_km, 2),
            "Max Distance from Seed (km)": round(max_dist_km, 2),
            "Valid Cluster (180–220 Orders & ≤2km from seed)": "Yes" if valid_cluster else "No",
            "Delivery Sequence": " → ".join(delivery_sequence),
            "Estimated Timings": " → ".join(delivery_timings)
        })

    st.subheader("Overall Cluster Map")
    st_folium(cluster_map, width=725)

    st.subheader("Cluster Summary")
    summary_df = pd.DataFrame(cluster_summary)
    st.dataframe(summary_df)
    csv = summary_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Cluster Summary CSV", data=csv, file_name="cluster_summary.csv")
