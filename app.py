import streamlit as st
import pandas as pd
import numpy as np
from geopy.distance import great_circle
import folium
from streamlit_folium import st_folium
from folium import PolyLine
from io import StringIO
import folium.plugins  # for AntPath

# Constants
VEHICLE_COST_PER_MONTH = 35000
WORKING_DAYS = 30
DELIVERIES_PER_DAY_PER_VEHICLE = 200
COST_PER_ORDER_DEFAULT = round(VEHICLE_COST_PER_MONTH / (WORKING_DAYS * DELIVERIES_PER_DAY_PER_VEHICLE), 2)
color_palette = ["red", "blue", "green", "orange", "purple", "darkred", "darkblue", "darkgreen",
                 "cadetblue", "pink", "gray", "black", "teal", "lightblue", "lightgreen", "beige", "brown"]

# Helper to calculate route distance in km
def calculate_route_distance(route):
    distance = 0.0
    for i in range(len(route) - 1):
        distance += great_circle(route[i], route[i+1]).km
    return distance

# Delivery sequence using nearest neighbor heuristic
def get_delivery_sequence(cluster_df, source_latlon=None):
    points = cluster_df[['Latitude', 'Longitude']].values.tolist()
    names = cluster_df['Society'].tolist()
    if len(points) <= 1:
        return names, points
    visited = [False] * len(points)
    sequence = []
    order = []
    current_index = np.argmin([great_circle(source_latlon, pt).km for pt in points]) if source_latlon else 0
    sequence.append(names[current_index])
    order.append(points[current_index])
    visited[current_index] = True
    for _ in range(1, len(points)):
        last_point = points[current_index]
        next_index = np.argmin([great_circle(last_point, pt).km if not visited[i] else float('inf') for i, pt in enumerate(points)])
        visited[next_index] = True
        sequence.append(names[next_index])
        order.append(points[next_index])
        current_index = next_index
    return sequence, order

# Check if candidate is within 2km from seed
def is_within_seed_radius(seed_coord, coord, max_dist_km=2.0):
    return great_circle(seed_coord, coord).km <= max_dist_km

# Streamlit app
st.title("Milk & Grocery Delivery Clustering Tool")

# Template Download
st.subheader("Download Template")
template = pd.DataFrame({"Society ID": [], "Society": [], "Latitude": [], "Longitude": [], "Orders": [], "Hub ID": []})
st.download_button("Download Template CSV", data=template.to_csv(index=False), file_name="society_template.csv")

# Upload
st.subheader("Upload Society Data")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# Supply source input
st.sidebar.subheader("Enter Supply Source Coordinates")
supply_lat = st.sidebar.number_input("Supply Latitude", value=12.989709, format="%.6f")
supply_lon = st.sidebar.number_input("Supply Longitude", value=77.786253, format="%.6f")
supply_source = (supply_lat, supply_lon)

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

    st.subheader("Cluster Summary")
    cluster_summary = []
    for cid in sorted(df['Cluster'].unique()):
        cluster_df = df[df['Cluster'] == cid]
        delivery_sequence, route_points = get_delivery_sequence(cluster_df, supply_source)
        total_orders = cluster_df['Orders'].sum()
        total_distance = calculate_route_distance(route_points)
        dc_to_first = great_circle(supply_source, route_points[0]).km if supply_source else 0
        round_trip = dc_to_first + total_distance + great_circle(route_points[-1], supply_source).km if supply_source else total_distance
        cost_per_order = round((VEHICLE_COST_PER_MONTH / (WORKING_DAYS * total_orders)), 2) if total_orders else 0.0

        cluster_summary.append({
            "Cluster ID": cid,
            "Hub ID": cluster_df['Hub ID'].iloc[0],
            "Societies": len(cluster_df),
            "Total Orders": total_orders,
            "Total Distance (km)": round(total_distance, 2),
            "DC to First Society (km)": round(dc_to_first, 2),
            "Round Trip Distance (km)": round(round_trip, 2),
            "Cost per Order (INR)": cost_per_order
        })

        st.markdown(f"### Cluster {cid} Summary")
        st.markdown(f"**Total Orders:** {total_orders} | **Round Trip Distance:** {round_trip:.2f} km | **Cost per Order:** ₹{cost_per_order:.2f}")
        individual_map = folium.Map(location=[cluster_df['Latitude'].mean(), cluster_df['Longitude'].mean()], zoom_start=14)
        for idx, row in cluster_df.iterrows():
            folium.Circle(
                location=(row['Latitude'], row['Longitude']),
                radius=300,
                color=color_palette[cid % len(color_palette)],
                fill=True,
                fill_opacity=0.2,
                tooltip=f"{row['Society']} - {row['Orders']} orders"
            ).add_to(individual_map)
        if supply_source:
            folium.Marker(
                location=supply_source,
                popup="DC",
                icon=folium.Icon(color='black', icon='home')
            ).add_to(individual_map)
        for i in range(len(route_points) - 1):
            folium.plugins.AntPath(
                locations=[route_points[i], route_points[i+1]],
                color=color_palette[cid % len(color_palette)],
                weight=3,
                delay=800
            ).add_to(individual_map)
        st_folium(individual_map, width=725, height=500)

    summary_df = pd.DataFrame(cluster_summary)
    st.dataframe(summary_df)
    csv = summary_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Cluster Summary CSV", data=csv, file_name="cluster_summary.csv")
