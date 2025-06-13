import streamlit as st
import pandas as pd
import numpy as np
from geopy.distance import great_circle
import folium
from streamlit_folium import st_folium
from folium import PolyLine
from folium.features import DivIcon
from io import StringIO
import random

# Helper to calculate route distance in km
def calculate_route_distance(route):
    distance = 0.0
    for i in range(len(route) - 1):
        distance += calculate_distance_km(route[i][0], route[i][1], route[i+1][0], route[i+1][1])
    return distance

# Helper to calculate distance between two lat/long points
def calculate_distance_km(lat1, lon1, lat2, lon2):
    return round(great_circle((lat1, lon1), (lat2, lon2)).km, 2)

# Optimized delivery sequence from depot using nearest neighbor heuristic
def get_delivery_sequence(cluster_df, depot_lat, depot_long):
    points = cluster_df[['Latitude', 'Longitude']].values.tolist()
    names = cluster_df['Society'].tolist()

    if len(points) <= 1:
        return names, points

    visited = [False] * len(points)
    sequence = []
    order = []

    current_point = (depot_lat, depot_long)

    for _ in range(len(points)):
        min_dist = float('inf')
        next_index = -1
        for i in range(len(points)):
            if not visited[i]:
                dist = calculate_distance_km(current_point[0], current_point[1], points[i][0], points[i][1])
                if dist < min_dist:
                    min_dist = dist
                    next_index = i
        visited[next_index] = True
        sequence.append(names[next_index])
        order.append(points[next_index])
        current_point = points[next_index]

    return sequence, order

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

            if cluster_orders >= 180:
                df.loc[cluster_members, 'Cluster'] = cluster_id
                cluster_id += 1

            hub_df = df[(df['Cluster'] == -1) & (df['Hub ID'] == hub)]

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
        sequence, route = get_delivery_sequence(cluster_df, source_lat, source_long)
        full_route = [(source_lat, source_long)] + route + [(source_lat, source_long)]
        total_distance = calculate_route_distance(full_route)
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
            "Total Distance (km)": round(total_distance, 2),
            "Max Distance from Seed (km)": round(max_dist, 2),
            "Valid Cluster (180 to 220 Orders & <2km)": valid_cluster,
            "Delivery Sequence": delivery_path,
            "Cost Per Order (₹)": cost_per_order
        })

    cluster_counts = df.groupby('Cluster').size().to_dict()
    cluster_labels = [f"Cluster {cid} ({cluster_counts[cid]} societies)" for cid in sorted(cluster_counts.keys())]
    cluster_id_map = {label: cid for label, cid in zip(cluster_labels, sorted(cluster_counts.keys()))}
    selected_label = st.sidebar.selectbox("Select Cluster ID to View Map", options=["All"] + cluster_labels)
    selected_cluster = "All" if selected_label == "All" else cluster_id_map[selected_label]

    if selected_cluster == "All":
        map_center = [df['Latitude'].mean(), df['Longitude'].mean()]
        m = folium.Map(location=map_center, zoom_start=12)
        for _, row in df.iterrows():
            folium.Marker(
                location=[row['Latitude'], row['Longitude']],
                popup=f"{row['Society']} (Orders: {row['Orders']}) - Cluster {row['Cluster']}",
                icon=folium.Icon(color="blue")
            ).add_to(m)
    else:
        cluster_df = df[df['Cluster'] == selected_cluster]
        sequence, route = get_delivery_sequence(cluster_df, source_lat, source_long)
        m = folium.Map(location=[cluster_df['Latitude'].mean(), cluster_df['Longitude'].mean()], zoom_start=13)

        society_names_line = " -> ".join(sequence)
        st.markdown(f"**Societies in Delivery Sequence:** {society_names_line}")

        folium.Marker(
            location=[source_lat, source_long],
            popup="Depot",
            icon=folium.Icon(color="red", icon="home")
        ).add_to(m)

        full_route = [(source_lat, source_long)] + route + [(source_lat, source_long)]

        for idx, point in enumerate(full_route):
            if idx < len(full_route) - 1:
                dist = calculate_distance_km(point[0], point[1], full_route[idx+1][0], full_route[idx+1][1])
                midpoint = [(point[0] + full_route[idx+1][0]) / 2, (point[1] + full_route[idx+1][1]) / 2]
                line_color = "orange" if idx == 0 else "blue"
                folium.PolyLine(locations=[point, full_route[idx+1]], color=line_color, weight=5, dash_array='10' if idx == 0 else None).add_to(m)
                folium.map.Marker(
                    location=midpoint,
                    icon=DivIcon(
                        icon_size=(150,36),
                        icon_anchor=(0,0),
                        html=f'<div style="font-size: 10pt; color: black">{dist:.2f} km</div>',
                    )
                ).add_to(m)

        for idx, point in enumerate(route):
            folium.Marker(
                location=point,
                popup=f"{sequence[idx]} (Orders: {cluster_df.iloc[idx]['Orders']})",
                icon=folium.Icon(color="green", icon="info-sign")
            ).add_to(m)

        total_orders = cluster_df['Orders'].sum()
        cost_per_order = round((van_cost + cee_cost) / total_orders, 2)

        st.subheader(f"Delivery Summary for Cluster {selected_cluster}")
        st.write(f"Total Orders: {total_orders}")
        st.write(f"Total Societies: {len(cluster_df)}")
        st.write(f"Estimated Route Distance (including return): {calculate_route_distance(full_route):.2f} km")
        st.write(f"Cost Per Order: ₹{cost_per_order}")

    st_data = st_folium(m, width=800, height=500)

    st.subheader("Cluster Summary Table")
    summary_df = pd.DataFrame(cluster_summary)
    st.dataframe(summary_df)
    st.download_button("Download Cluster Summary CSV", data=summary_df.to_csv(index=False), file_name="cluster_summary.csv")
