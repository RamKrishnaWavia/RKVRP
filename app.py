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
        distance += great_circle(route[i], route[i+1]).km
    return distance

# Helper to create delivery sequence using nearest neighbor heuristic
def get_delivery_sequence(cluster_df):
    points = cluster_df[['Latitude', 'Longitude']].values.tolist()
    names = cluster_df['Society'].tolist()

    if len(points) <= 1:
        return names, points

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

st.title("RK - Societies Delivery Clustering Tool")

# Sidebar source location input
st.sidebar.subheader("Supply Source Location")
def_lat = 12.989708618922553
def_long = 77.78625342251868
source_lat = st.sidebar.number_input("Depot Latitude", value=def_lat, format="%.8f")
source_long = st.sidebar.number_input("Depot Longitude", value=def_long, format="%.8f")

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

    cluster_summary = []

    for cluster in sorted(df['Cluster'].unique()):
        cluster_df = df[df['Cluster'] == cluster]
        hub_id = cluster_df['Hub ID'].iloc[0]
        society_ids = list(cluster_df['Society ID'])
        societies = list(cluster_df['Society'])
        num_societies = len(cluster_df)
        total_orders = cluster_df['Orders'].sum()
        seed_coord = (cluster_df.iloc[0]['Latitude'], cluster_df.iloc[0]['Longitude'])
        max_dist = max(great_circle(seed_coord, (row['Latitude'], row['Longitude'])).km for _, row in cluster_df.iterrows())
        sequence, route = get_delivery_sequence(cluster_df)
        total_distance = calculate_route_distance([(source_lat, source_long)] + route)
        valid_cluster = 180 <= total_orders <= 220 and max_dist <= 2.0
        delivery_path = " -> ".join(sequence)
        cost_per_order = round((35000 / total_orders), 2)

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
        sequence, route = get_delivery_sequence(cluster_df)
        m = folium.Map(location=[cluster_df['Latitude'].mean(), cluster_df['Longitude'].mean()], zoom_start=13)

        society_names_line = " -> ".join(sequence)
        st.markdown(f"**Societies in Delivery Sequence:** {society_names_line}")

        folium.Marker(
            location=[source_lat, source_long],
            popup="Depot",
            icon=folium.Icon(color="red", icon="home")
        ).add_to(m)

        full_route = [(source_lat, source_long)] + route

        for idx, point in enumerate(full_route):
            if idx < len(full_route) - 1:
                dist = great_circle(point, full_route[idx+1]).km
                midpoint = [(point[0] + full_route[idx+1][0]) / 2, (point[1] + full_route[idx+1][1]) / 2]
                folium.PolyLine(locations=[point, full_route[idx+1]], color="blue").add_to(m)
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
        cost_per_order = round((35000 / total_orders), 2)

        st.subheader(f"Delivery Summary for Cluster {selected_cluster}")
        st.write(f"Total Orders: {total_orders}")
        st.write(f"Total Societies: {len(cluster_df)}")
        st.write(f"Estimated Route Distance: {calculate_route_distance(full_route):.2f} km")
        st.write(f"Cost Per Order: ₹{cost_per_order}")

    st_data = st_folium(m, width=800, height=500)

    st.subheader("Cluster Summary Table")
    summary_df = pd.DataFrame(cluster_summary)
    st.dataframe(summary_df)
    st.download_button("Download Cluster Summary CSV", data=summary_df.to_csv(index=False), file_name="cluster_summary.csv")
