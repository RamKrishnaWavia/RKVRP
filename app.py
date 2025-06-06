# app.py
import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster, AntPath
from streamlit_folium import st_folium
from sklearn.cluster import DBSCAN
from shapely.geometry import MultiPoint
from math import radians, cos, sin, asin, sqrt
import io
import networkx as nx

st.set_page_config(layout="wide")
st.title("Milk Delivery Cluster Optimizer")

# Sidebar settings
st.sidebar.header("Clustering Settings")
VEHICLE_MIN_ORDERS = st.sidebar.number_input("Min Orders per Vehicle", value=150)
VEHICLE_MAX_ORDERS = st.sidebar.number_input("Max Orders per Vehicle", value=450)
CLUSTER_RADIUS_KM = st.sidebar.slider("Clustering Radius (km)", 0.1, 2.0, 0.5, step=0.1)

# Editable cost/capacity
VAN_COST = st.sidebar.number_input("Van Cost per Month", value=25000, step=1000)
CEE_COST = st.sidebar.number_input("CEE Cost per Month", value=10000, step=1000)
CEE_CAPACITY = st.sidebar.number_input("CEE Capacity (Orders)", value=200)
VEHICLE_CAPACITY = st.sidebar.number_input("Vehicle Order Capacity", value=450)

uploaded_file = st.file_uploader("Upload Society Orders CSV", type=["csv"])

# Haversine distance
def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 6371.0088 * 2 * asin(sqrt(a))

# Group societies to minimize CEE count
def auto_group_minimize_cees(cluster_df):
    points = cluster_df[['Latitude', 'Longitude', 'SocietyOrders', 'SocietyID']].values.tolist()
    unassigned = set(range(len(points)))
    cee_groups = []

    while unassigned:
        current = unassigned.pop()
        group = [current]
        group_order = points[current][2]

        close_points = []
        for idx in list(unassigned):
            dist = haversine_distance(points[current][0], points[current][1], points[idx][0], points[idx][1])
            if dist <= CLUSTER_RADIUS_KM:
                close_points.append((idx, dist))
        close_points.sort(key=lambda x: x[1])

        for idx, _ in close_points:
            if group_order + points[idx][2] <= CEE_CAPACITY:
                group.append(idx)
                group_order += points[idx][2]
                unassigned.remove(idx)
        cee_groups.append(group)

    cee_routes = []
    for group in cee_groups:
        G = nx.complete_graph(len(group))
        for i in G.nodes:
            G.nodes[i]['coord'] = (points[group[i]][0], points[group[i]][1])
        for i, j in G.edges:
            c1, c2 = G.nodes[i]['coord'], G.nodes[j]['coord']
            G[i][j]['weight'] = haversine_distance(c1[0], c1[1], c2[0], c2[1])
        tsp_path = nx.approximation.greedy_tsp(G)
        route = [points[group[i]] for i in tsp_path]
        cee_routes.append(route)

    return cee_routes

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=['Latitude', 'Longitude'])

    df['Latitude'] = df['Latitude'].astype(float)
    df['Longitude'] = df['Longitude'].astype(float)
    df['SocietyID'] = df.index  # Unique ID

    coords = df[['Latitude', 'Longitude']].to_numpy()
    db = DBSCAN(eps=CLUSTER_RADIUS_KM / 6371.0088, min_samples=1, algorithm='ball_tree', metric='haversine')
    db.fit(np.radians(coords))
    df['ClusterID'] = db.labels_

    cluster_summary = df.groupby('ClusterID').agg({
        'SocietyOrders': 'sum',
        'Latitude': 'mean',
        'Longitude': 'mean'
    }).reset_index()

    cluster_summary['ClusterType'] = cluster_summary['SocietyOrders'].apply(lambda x: 'Green' if x >= VEHICLE_MIN_ORDERS else 'Blue')
    cluster_summary['VehiclesRequired'] = np.ceil(cluster_summary['SocietyOrders'] / VEHICLE_CAPACITY).astype(int)
    cluster_summary['CEEs_Before_Clustering'] = np.ceil(cluster_summary['SocietyOrders'] / CEE_CAPACITY).astype(int)
    cluster_summary['TotalCost'] = cluster_summary['VehiclesRequired'] * VAN_COST + cluster_summary['CEEs_Before_Clustering'] * CEE_COST
    cluster_summary['CostPerOrder'] = (cluster_summary['TotalCost'] / cluster_summary['SocietyOrders']).round(2)

    df = df.merge(cluster_summary[['ClusterID', 'ClusterType']], on='ClusterID')

    m = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=13)
    marker_cluster = MarkerCluster().add_to(m)

    colors = ['green', 'blue', 'red', 'orange', 'purple', 'darkred', 'lightred','beige']
    all_routes = []
    cee_summary_records = []

    for cluster_id in df['ClusterID'].unique():
        cluster_df = df[df['ClusterID'] == cluster_id].copy()
        routes = auto_group_minimize_cees(cluster_df)
        for i, route in enumerate(routes):
            route_points = [(p[0], p[1]) for p in route]
            AntPath(route_points, color='blue', delay=1000).add_to(m)
            for p in route:
                df.loc[df['SocietyID'] == p[3], 'CEE_Group'] = f'{cluster_id}_{i+1}'
            cee_summary_records.append({
                'ClusterID': cluster_id,
                'CEE_Group': f'{cluster_id}_{i+1}',
                'TotalOrders': sum([p[2] for p in route]),
                'CEEsRequired': np.ceil(sum([p[2] for p in route]) / CEE_CAPACITY)
            })
        all_routes.extend(routes)

    st_data = st_folium(m, width=1000, height=600)

    df = df.merge(cluster_summary[['ClusterID', 'TotalCost', 'SocietyOrders']].rename(columns={'SocietyOrders': 'ClusterOrders'}), on='ClusterID')
    df['Society_CPO'] = (df['TotalCost'] * (df['SocietyOrders'] / df['ClusterOrders'])).round(2)

    cee_summary_df = pd.DataFrame(cee_summary_records)

    st.subheader("Cluster Summary")
    st.dataframe(cluster_summary)

    st.subheader("CEE Route Summary (Post Clustering)")
    st.dataframe(cee_summary_df)

    st.subheader("Overall Summary")
    total_orders = cluster_summary['SocietyOrders'].sum()
    total_cost = cluster_summary['TotalCost'].sum()
    overall_cpo = round(total_cost / total_orders, 2) if total_orders > 0 else 0
    st.markdown(f"**Total Orders:** {total_orders}")
    st.markdown(f"**Total Cost:** ₹{total_cost}")
    st.markdown(f"**Overall Cost per Order (CPO): ₹{overall_cpo}**")

    st.download_button("Download Cluster Summary", cluster_summary.to_csv(index=False), "cluster_summary.csv", "text/csv")
    st.download_button("Download CEE Route Summary", cee_summary_df.to_csv(index=False), "cee_summary.csv", "text/csv")
    st.download_button("Download Society-Level Mapping", df[[
        'SocietyID', 'Society Name', 'Latitude', 'Longitude', 'SocietyOrders',
        'ClusterID', 'ClusterType', 'CEE_Group', 'Society_CPO'
    ]].to_csv(index=False), "society_cluster_mapping.csv", "text/csv")
