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

# Set logging level
logging.basicConfig(level=logging.ERROR)

# Helper to calculate distance between two points
def calculate_distance_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return round(R * 2 * atan2(sqrt(a), sqrt(1 - a)), 2)

def is_within_radius(coord1, coord2, radius=2.0):
    return calculate_distance_km(*coord1, *coord2) <= radius

def cluster_societies(df, min_orders=180):
    clusters = []
    used = set()
    cluster_id = 1
    for i, row in df.iterrows():
        if i in used:
            continue
        seed = (row['Latitude'], row['Longitude'])
        members = [i]
        total_orders = row['Orders']
        for j, other in df.iterrows():
            if j != i and j not in used:
                coord = (other['Latitude'], other['Longitude'])
                if is_within_radius(seed, coord):
                    members.append(j)
                    total_orders += other['Orders']
                    if total_orders >= min_orders:
                        break
        if total_orders >= min_orders:
            for idx in members:
                df.at[idx, 'Cluster_ID'] = cluster_id
                df.at[idx, 'Cluster_Type'] = 'Green'
                used.update(members)
            cluster_id += 1
    # Remaining as blue or unclustered
    for i in df.index:
        if pd.isna(df.at[i, 'Cluster_ID']):
            df.at[i, 'Cluster_ID'] = cluster_id
            df.at[i, 'Cluster_Type'] = 'Blue'
            cluster_id += 1
    return df

def get_delivery_sequence(cluster_df, depot_lat, depot_long):
    points = cluster_df[['Latitude', 'Longitude']].values.tolist()
    names = cluster_df['Society'].tolist()
    orders = cluster_df['Orders'].tolist()
    visited = [False] * len(points)
    sequence, order, distances = [], [], []
    current_index = 0
    current_point = points[current_index]
    visited[current_index] = True
    sequence.append(f"{names[current_index]} - {orders[current_index]} orders (0 km)")
    order.append(current_point)
    distances.append(0)
    for _ in range(len(points) - 1):
        min_dist, next_index = float('inf'), -1
        for i, point in enumerate(points):
            if not visited[i]:
                dist = calculate_distance_km(*current_point, *point)
                if dist < min_dist:
                    min_dist, next_index = dist, i
        if next_index == -1:
            break
        visited[next_index] = True
        current_point = points[next_index]
        sequence.append(f"{names[next_index]} - {orders[next_index]} orders ({min_dist} km)")
        distances.append(min_dist)
        order.append(current_point)
    sequence.append(f"Total Distance: {round(sum(distances), 2)} km")
    return sequence, order, distances

st.title("Milk Delivery Clustering Tool")

sample_csv = """Society_ID,Society,Latitude,Longitude,Orders
1001,Alpha,12.9611,77.6387,90
1002,Beta,12.9632,77.6401,110
1003,Gamma,12.9644,77.6423,30
1004,Delta,12.9688,77.6478,75
1005,Epsilon,12.9712,77.6499,65
1006,Zeta,12.9741,77.6551,80
"""

st.download_button("Download Sample CSV", sample_csv, "sample_societies.csv")

uploaded_file = st.file_uploader("Upload society order file (CSV)", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['Cluster_ID'] = np.nan
    df['Cluster_Type'] = ""
    depot_lat = st.number_input("Enter Depot Latitude", value=12.9611)
    depot_lon = st.number_input("Enter Depot Longitude", value=77.6387)
    df = cluster_societies(df)
    st.success("Clustering complete.")

    cluster_options = sorted(df['Cluster_ID'].unique())
    selected_id = st.selectbox("Select Cluster ID to view", cluster_options)
    selected_df = df[df['Cluster_ID'] == selected_id].reset_index(drop=True)

    st.subheader(f"Cluster {selected_id} - {selected_df['Cluster_Type'][0]}")
    st.dataframe(selected_df)

    seq, route, dists = get_delivery_sequence(selected_df, depot_lat, depot_lon)
    st.write("Delivery Sequence:")
    for line in seq:
        st.markdown(f"- {line}")

    m = folium.Map(location=[depot_lat, depot_lon], zoom_start=13)
    folium.Marker([depot_lat, depot_lon], popup="Depot", icon=folium.Icon(color='green')).add_to(m)

    full_route = [(depot_lat, depot_lon)] + route + [(depot_lat, depot_lon)]
    for i in range(len(full_route) - 1):
        start, end = full_route[i], full_route[i+1]
        d = calculate_distance_km(*start, *end)
        PolyLine([start, end], color="blue").add_to(m)
        mid = [(start[0]+end[0])/2, (start[1]+end[1])/2]
        folium.Marker(mid, icon=DivIcon(icon_size=(150,36), html=f'<div style="font-size:10pt; color:red">{d} km</div>')).add_to(m)

    for i, (idx, row) in enumerate(selected_df.iterrows()):
        coord = (row['Latitude'], row['Longitude'])
        Marker(coord, popup=f"{row['Society']} - {row['Orders']} orders", icon=folium.Icon(color='blue')).add_to(m)
        folium.Marker(coord, icon=DivIcon(icon_size=(150,36), html=f'<div style="font-size:12pt">S{i+1}</div>')).add_to(m)

    st_folium(m, height=500)

    st.download_button("Download Cluster Data", df.to_csv(index=False), file_name="clustered_societies.csv")
