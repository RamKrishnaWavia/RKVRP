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
supply_source = (supply_lat, supply_lon) if supply_lat and supply_lon else None

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

# Draw main map only once before looping through clusters
st.subheader("Overall Cluster Map")
# Initialize and build the main map before this line
cluster_filter = df['Cluster'].unique() if selected_cluster == "All" else [selected_cluster]
cluster_map = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=12)
st_data = st_folium(cluster_map, width=725)

for label in sorted(cluster_filter):
    cluster_df = df[df['Cluster'] == label]
    if cluster_df.empty:
        continue
    total_orders = cluster_df['Orders'].sum()
    coords = list(zip(cluster_df['Latitude'], cluster_df['Longitude']))
    distance_km = calculate_route_distance(coords)

    # Seed info
    seed_coord = (cluster_df.iloc[0]['Latitude'], cluster_df.iloc[0]['Longitude'])
    max_dist_km = max(great_circle(seed_coord, (row['Latitude'], row['Longitude'])).km for _, row in cluster_df.iterrows())

    valid_cluster = 180 <= total_orders <= 220 and max_dist_km <= 2.0
    hub_color_map = {hub: color_palette[i % len(color_palette)] for i, hub in enumerate(df['Hub ID'].unique())}
    cluster_color_map = {cid: color for cid, color in zip(df['Cluster'].unique(), [
        "red", "blue", "green", "orange", "purple", "darkred", "darkblue", "darkgreen",
        "cadetblue", "pink", "gray", "black", "teal", "lightblue", "lightgreen", "beige", "brown"
    ])}
    color = cluster_color_map[label]

    delivery_sequence, route_points = get_delivery_sequence(cluster_df, supply_source)

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

    # Add numbered markers based on delivery sequence
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
            mid_point = [(route_points[i][0] + route_points[i+1][0]) / 2, (route_points[i][1] + route_points[i+1][1]) / 2]
            folium.plugins.AntPath(
                locations=[route_points[i], route_points[i+1]],
                color=color,
                weight=4,
                opacity=0.9,
                tooltip=f"{dist:.2f} km"
            ).add_to(cluster_map)

            folium.RegularPolygonMarker(
                location=route_points[i+1],
                number_of_sides=3,
                radius=8,
                rotation=0,
                color=color,
                fill_color=color,
                fill_opacity=1
            ).add_to(cluster_map)

    cluster_name = str(label)

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
        "Single Society Cluster": "Yes" if len(cluster_df) == 1 else "No",
        "DC to First Society (km)": round(great_circle(supply_source, route_points[0]).km, 2) if supply_source and route_points else 0.0,
        "Total Distance via DC to last point (km)": round(great_circle(supply_source, route_points[0]).km + distance_km, 2) if supply_source and route_points else round(distance_km, 2),
        "Round Trip Distance (DC → Cluster → DC) (km)": round(great_circle(supply_source, route_points[0]).km + distance_km + great_circle(route_points[-1], supply_source).km, 2) if supply_source and route_points else round(distance_km, 2)
    })

# Individual maps for each cluster
st.subheader("Individual Cluster Maps")
for label in sorted(df['Cluster'].unique()):
        cluster_df = df[df['Cluster'] == label]
        if cluster_df.empty:
            continue
        cluster_center = [cluster_df['Latitude'].mean(), cluster_df['Longitude'].mean()]
        individual_map = folium.Map(location=cluster_center, zoom_start=14)

        color = hub_color_map[cluster_df['Hub ID'].iloc[0]]
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
                    popup=f"{row['Society']}\nOrders: {row['Orders']}\nCluster ID: {label} (Seed)",
                    tooltip=f"SEED: {row['Society']} ({row['Orders']} orders) - Cluster {label}",
                    icon=folium.Icon(color="darkpurple", icon='star')
                ).add_to(individual_map)
            else:
                folium.Marker(
                    location=[row['Latitude'], row['Longitude']],
                    popup=f"{row['Society']}\nOrders: {row['Orders']}\nCluster ID: {label}",
                    tooltip=f"{row['Society']} ({row['Orders']} orders) - Cluster {label}",
                    icon=folium.Icon(color=color, icon='info-sign')
                ).add_to(individual_map)

        delivery_sequence, route_points = get_delivery_sequence(cluster_df, supply_source)

        # Add DC marker
        if supply_source:
            folium.Marker(
                location=supply_source,
                popup="Supply DC",
                tooltip="Supply DC",
                icon=folium.Icon(color="black", icon="home")
            ).add_to(individual_map)

            # Line from DC to first point
            folium.PolyLine(
                locations=[supply_source, route_points[0]],
                color="black",
                weight=3,
                dash_array='5,5',
                tooltip=f"{great_circle(supply_source, route_points[0]).km:.2f} km"
            ).add_to(individual_map)

        if len(route_points) > 1:
            for i in range(len(route_points) - 1):
                dist = great_circle(route_points[i], route_points[i+1]).km
                mid_point = [(route_points[i][0] + route_points[i+1][0]) / 2, (route_points[i][1] + route_points[i+1][1]) / 2]
                folium.plugins.AntPath(
                locations=[route_points[i], route_points[i+1]],
                color=color,
                weight=4,
                opacity=0.9,
                tooltip=f"{dist:.2f} km"
            ).add_to(individual_map)

            folium.RegularPolygonMarker(
                location=route_points[i+1],
                number_of_sides=3,
                radius=8,
                rotation=0,
                color=color,
                fill_color=color,
                fill_opacity=1
            ).add_to(individual_map)

        st.markdown("<br><br>", unsafe_allow_html=True)
        total_orders = cluster_df['Orders'].sum()
        total_distance = calculate_route_distance(list(zip(cluster_df['Latitude'], cluster_df['Longitude'])))
        st.markdown(f"### Cluster {label} Map")
        max_leg_distance = 0.0
        if len(route_points) > 1:
            max_leg_distance = max(great_circle(route_points[i], route_points[i+1]).km for i in range(len(route_points)-1))
        dc_to_first = great_circle(supply_source, route_points[0]).km if supply_source and route_points else 0.0
        dc_total_distance = great_circle(supply_source, route_points[0]).km + total_distance if supply_source and route_points else total_distance
        dc_back_distance = great_circle(route_points[-1], supply_source).km if supply_source and route_points else 0.0
        round_trip_distance = dc_total_distance + dc_back_distance
        st.markdown(f"**Total Orders:** {total_orders} | **Total Distance Travelled in Cluster:** {total_distance:.2f} km | **No. of Societies:** {len(cluster_df)} | **Max Leg Distance between the society:** {max_leg_distance:.2f} km | **DC to First Society:** {dc_to_first:.2f} km | **Total Distance via DC to last delivery point:** {dc_total_distance:.2f} km | **Total Round Trip (DC → Cluster → DC):** {round_trip_distance:.2f} km")
        st_folium(individual_map, width=725)

        summary_df = pd.DataFrame(cluster_summary)
        st.subheader("Cluster Summary")
        st.dataframe(summary_df)

        csv = summary_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Cluster Summary CSV", data=csv, file_name="cluster_summary.csv")
