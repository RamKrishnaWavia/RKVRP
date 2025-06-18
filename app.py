import streamlit as st
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
import folium
from streamlit_folium import st_folium
from folium import PolyLine, Marker
from io import StringIO
import random
import logging

logging.basicConfig(level=logging.ERROR)

# Utility Functions
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

def get_delivery_sequence(cluster_df):
    points = cluster_df[['Latitude', 'Longitude']].values.tolist()
    names = cluster_df['Society Name'].tolist()
    if not points:
        return [], [], 0.0, ""
    sequence = []
    visited = [False] * len(points)
    path = [0]
    visited[0] = True
    current = 0
    total_distance = 0
    while len(path) < len(points):
        nearest = None
        min_dist = float('inf')
        for i in range(len(points)):
            if not visited[i]:
                dist = calculate_distance_km(points[current][0], points[current][1], points[i][0], points[i][1])
                if dist < min_dist:
                    min_dist = dist
                    nearest = i
        if nearest is not None:
            visited[nearest] = True
            total_distance += min_dist
            path.append(nearest)
            current = nearest
    total_distance += calculate_distance_km(points[path[-1]][0], points[path[-1]][1], depot_lat, depot_long)
    delivery_seq = []
    delivery_path = []
    for i in range(len(path) - 1):
        a = path[i]
        b = path[i+1]
        dist = calculate_distance_km(points[a][0], points[a][1], points[b][0], points[b][1])
        delivery_seq.append(f"{names[a]} -> {names[b]} ({dist} km)")
        delivery_path.append((points[a], points[b], dist))
    if len(path) == 1:
        delivery_seq.append(f"{names[path[0]]} -> {names[path[0]]} (0 km)")
    return delivery_seq, delivery_path, total_distance, " | ".join(delivery_seq)

def calculate_cpo(total_orders, van_cost, cee_cost=0):
    if total_orders == 0:
        return 0
    return round((van_cost + cee_cost) / total_orders, 2)

# Sidebar Inputs
st.sidebar.header("Depot Settings")
depot_lat = st.sidebar.number_input("Depot Latitude", value=12.9716, format="%.6f")
depot_long = st.sidebar.number_input("Depot Longitude", value=77.5946, format="%.6f")

st.sidebar.header("Main Cluster Cost Settings")
main_van_cost = st.sidebar.number_input("Van Cost (₹) [Main]", value=834)
main_cee_cost = st.sidebar.number_input("CEE Cost (₹) [Main]", value=333)

st.sidebar.header("Micro Cluster Cost Settings")
micro_van_cost = st.sidebar.number_input("Van Cost (₹) [Micro]", value=500)
micro_cee_cost = st.sidebar.number_input("CEE Cost (₹) [Micro]", value=167)

st.sidebar.header("Mini Cluster Cost Settings")
mini_van_cost = st.sidebar.number_input("Van Cost (₹) [Mini]", value=700)
mini_cee_cost = st.sidebar.number_input("CEE Cost (₹) [Mini]", value=200)

st.sidebar.markdown("### Download Input Template")
template = pd.DataFrame({
    'Society ID': [], 'Society Name': [], 'Latitude': [], 'Longitude': [], 'Orders': [], 'Hub ID': []
})
st.sidebar.download_button("Download Template", template.to_csv(index=False), "input_template.csv")

# Upload File
st.title("Milk Delivery Cluster Optimizer")
file = st.file_uploader("Upload Society Data CSV", type=["csv"])
if file is None:
    st.stop()

df = pd.read_csv(file)
st.dataframe(df)

main_clusters = []
micro_clusters = []
mini_clusters = []
unclustered = []
used = set()

cluster_id = 1
for hub_id in df['Hub ID'].unique():
    hub_df = df[df['Hub ID'] == hub_id]
    for i, row in hub_df.iterrows():
        if row['Orders'] >= 180 and i not in used:
            base = (row['Latitude'], row['Longitude'])
            cluster_df = hub_df.loc[hub_df.index != i].copy()
            cluster_df['Distance'] = cluster_df.apply(lambda x: calculate_distance_km(base[0], base[1], x['Latitude'], x['Longitude']), axis=1)
            nearby = cluster_df[cluster_df['Distance'] <= 2.0]
            members = hub_df.loc[nearby.index.union([i])]
            main_clusters.append((cluster_id, members))
            used.update(members.index)
            cluster_id += 1

micro_id = 1
remaining = df.loc[~df.index.isin(used)]
remaining = remaining[remaining['Orders'] < 120]
for hub_id in remaining['Hub ID'].unique():
    hub_remaining = remaining[remaining['Hub ID'] == hub_id]
    while not hub_remaining.empty:
        seed = hub_remaining.iloc[0]
        base = (seed['Latitude'], seed['Longitude'])
        cluster_df = hub_remaining.copy()
        cluster_df['Distance'] = cluster_df.apply(lambda x: calculate_distance_km(base[0], base[1], x['Latitude'], x['Longitude']), axis=1)
        members = cluster_df[cluster_df['Distance'] <= 2.0]
        seq, _, total_dist, _ = get_delivery_sequence(members)
        if total_dist <= 20:
            micro_clusters.append((micro_id, members))
            used.update(members.index)
            micro_id += 1
        hub_remaining = hub_remaining.loc[~hub_remaining.index.isin(used)]

# Mini Cluster
mini_id = 1
midrange = df.loc[~df.index.isin(used)]
midrange = midrange[(midrange['Orders'] >= 121) & (midrange['Orders'] <= 179)]
for hub_id in midrange['Hub ID'].unique():
    hub_midrange = midrange[midrange['Hub ID'] == hub_id]
    while not hub_midrange.empty:
        seed = hub_midrange.iloc[0]
        base = (seed['Latitude'], seed['Longitude'])
        cluster_df = hub_midrange.copy()
        cluster_df['Distance'] = cluster_df.apply(lambda x: calculate_distance_km(base[0], base[1], x['Latitude'], x['Longitude']), axis=1)
        members = cluster_df[cluster_df['Distance'] <= 2.0]
        seq, _, total_dist, _ = get_delivery_sequence(members)
        if total_dist <= 15:
            mini_clusters.append((mini_id, members))
            used.update(members.index)
            mini_id += 1
        hub_midrange = hub_midrange.loc[~hub_midrange.index.isin(used)]

unclustered_df = df.loc[~df.index.isin(used)]

main_ids = [f"Main-{cid}" for cid, _ in main_clusters]
micro_ids = [f"Micro-{cid}" for cid, _ in micro_clusters]
mini_ids = [f"Mini-{cid}" for cid, _ in mini_clusters]
selected_main = st.selectbox("Select Main Cluster", ["None"] + main_ids)
selected_micro = st.selectbox("Select Micro Cluster", ["None"] + micro_ids)
selected_mini = st.selectbox("Select Mini Cluster", ["None"] + mini_ids)

def show_cluster_on_map(cluster_df, title):
    m = folium.Map(location=[depot_lat, depot_long], zoom_start=13)
    folium.Marker([depot_lat, depot_long], tooltip="Depot", icon=folium.Icon(color='blue', icon='home')).add_to(m)
    seq, path, total_dist, _ = get_delivery_sequence(cluster_df)
    for _, row in cluster_df.iterrows():
        folium.Marker([row['Latitude'], row['Longitude']], tooltip=f"{row['Society Name']} (Orders: {row['Orders']})").add_to(m)
    for start, end, dist in path:
        PolyLine([start, end], color="green", weight=2.5, opacity=1, tooltip=f"{dist} km").add_to(m)
    return st_folium(m, width=700, height=500)

summary_rows = []
def add_summary(cid, members, ctype, vcost, ccost):
    total_orders = members['Orders'].sum()
    seq, _, tdist, path = get_delivery_sequence(members)
    cpo = calculate_cpo(total_orders, vcost, ccost)
    return {
        'Cluster Type': ctype,
        'Cluster ID': cid,
        'No. of Societies': len(members),
        'Total Orders': total_orders,
        'Total Distance (km)': tdist,
        'CPO (₹)': cpo,
        'Delivery Sequence': path,
        'Total Routes': 1,
        'Total Clusters': 1
    }

for cid, members in main_clusters:
    summary_rows.append(add_summary(cid, members, "Main", main_van_cost, main_cee_cost))

for cid, members in micro_clusters:
    summary_rows.append(add_summary(cid, members, "Micro", micro_van_cost, micro_cee_cost))

for cid, members in mini_clusters:
    summary_rows.append(add_summary(cid, members, "Mini", mini_van_cost, mini_cee_cost))

for _, row in unclustered_df.iterrows():
    summary_rows.append({
        'Cluster Type': 'Unclustered',
        'Cluster ID': row['Society ID'],
        'No. of Societies': 1,
        'Total Orders': row['Orders'],
        'Total Distance (km)': 0,
        'CPO (₹)': 0,
        'Delivery Sequence': f"{row['Society Name']} -> {row['Society Name']} (0 km)",
        'Total Routes': 0,
        'Total Clusters': 0
    })

summary_df = pd.DataFrame(summary_rows)
st.subheader("Cluster Summary")
st.dataframe(summary_df)
st.download_button("Download Cluster Summary", summary_df.to_csv(index=False), file_name="cluster_summary.csv")

# Cumulative Summary
cumulative = summary_df.groupby('Cluster Type').agg({
    'No. of Societies': 'sum',
    'Total Orders': 'sum',
    'Total Distance (km)': 'sum',
    'Total Routes': 'sum',
    'Total Clusters': 'sum'
}).reset_index()
cumulative['Average CPO (₹)'] = summary_df.groupby('Cluster Type')['CPO (₹)'].mean().values

st.subheader("Cumulative Summary")
st.dataframe(cumulative)

if selected_main != "None":
    cid = int(selected_main.split('-')[1])
    members = next(c[1] for c in main_clusters if c[0] == cid)
    st.subheader(f"Main Cluster {cid} Summary")
    st.dataframe(pd.DataFrame([add_summary(cid, members, "Main", main_van_cost, main_cee_cost)]))
    st.subheader(f"Map for Main Cluster {cid}")
    show_cluster_on_map(members, f"Main Cluster {cid}")

if selected_micro != "None":
    cid = int(selected_micro.split('-')[1])
    members = next(c[1] for c in micro_clusters if c[0] == cid)
    st.subheader(f"Micro Cluster {cid} Summary")
    st.dataframe(pd.DataFrame([add_summary(cid, members, "Micro", micro_van_cost, micro_cee_cost)]))
    st.subheader(f"Map for Micro Cluster {cid}")
    show_cluster_on_map(members, f"Micro Cluster {cid}")

if selected_mini != "None":
    cid = int(selected_mini.split('-')[1])
    members = next(c[1] for c in mini_clusters if c[0] == cid)
    st.subheader(f"Mini Cluster {cid} Summary")
    st.dataframe(pd.DataFrame([add_summary(cid, members, "Mini", mini_van_cost, mini_cee_cost)]))
    st.subheader(f"Map for Mini Cluster {cid}")
    show_cluster_on_map(members, f"Mini Cluster {cid}")
