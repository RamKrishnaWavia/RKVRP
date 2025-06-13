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

MIN_ORDER_THRESHOLD = 180

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

            if cluster_orders >= MIN_ORDER_THRESHOLD:
                df.loc[cluster_members, 'Cluster'] = cluster_id
                cluster_id += 1
            hub_df = df[(df['Cluster'] == -1) & (df['Hub ID'] == hub)]

    cluster_counts = df.groupby('Cluster').size().to_dict()
    cluster_labels = [f"Cluster {cid} ({cluster_counts[cid]} societies)" for cid in sorted(cluster_counts.keys())]
    cluster_id_map = {label: cid for label, cid in zip(cluster_labels, sorted(cluster_counts.keys()))}
    selected_label = st.sidebar.selectbox("Select Cluster ID to View Map", options=["All"] + cluster_labels)
    selected_cluster = "All" if selected_label == "All" else cluster_id_map[selected_label]
    # Show cluster-wise summary when a specific cluster is selected
    if selected_cluster != "All":
        filtered_summary = [c for c in cluster_summary if c['Cluster ID'] == selected_cluster]
    if filtered_summary:
        cluster_info = filtered_summary[0]
        st.markdown("### Cluster Summary")
        st.markdown(f"- **Total Orders:** {cluster_info['Total Orders']}")
        st.markdown(f"- **No. of Societies:** {cluster_info['No. of Societies']}")
        st.markdown(f"- **Max Distance from Seed (km):** {cluster_info['Max Distance from Seed (km)']:.2f}")
        st.markdown(f"- **DC to First Society (km):** {cluster_info['DC to First Society (km)']:.2f}")
        st.markdown(f"- **Total Distance via DC (km):** {cluster_info['Total Distance via DC (km)']:.2f}")
        st.markdown(f"- **Round Trip Distance (DC → Cluster → DC):** {cluster_info['Round Trip Distance (DC → Cluster → DC) (km)']:.2f}")

    cluster_summary = []

# Draw main map only once before looping through clusters
st.subheader("Overall Cluster Map")

...
