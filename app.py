
import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster, AntPath
from streamlit_folium import st_folium
from sklearn.cluster import DBSCAN
from shapely.geometry import MultiPoint
from math import radians, cos, sin, asin, sqrt
import networkx as nx

st.set_page_config(layout="wide")
st.title("Milk Delivery Cluster Optimizer")

# Sidebar parameters
st.sidebar.header("Clustering Settings")
VEHICLE_MIN_ORDERS = st.sidebar.number_input("Min Orders per Vehicle", value=150, step=10)
VEHICLE_MAX_ORDERS = st.sidebar.number_input("Max Orders per Vehicle", value=450, step=10)
CLUSTER_RADIUS_KM = st.sidebar.slider("Clustering Radius (km)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
VAN_COST = st.sidebar.number_input("Van Cost per Month", value=25000, step=1000)
CEE_COST = st.sidebar.number_input("CEE Cost per Month", value=10000, step=1000)
CEE_CAPACITY = st.sidebar.number_input("CEE Min Orders", value=200, step=10)
VEHICLE_ORDER_CAPACITY = st.sidebar.number_input("Vehicle Order Capacity", value=450, step=10)

uploaded_file = st.file_uploader("Upload Society Orders CSV", type=["csv"])

def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1 
    dlon = lon2 - lon1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    km = 6371.0088 * c
    return km

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
            coord1 = G.nodes[i]['coord']
            coord2 = G.nodes[j]['coord']
            G[i][j]['weight'] = haversine_distance(coord1[0], coord1[1], coord2[0], coord2[1])

        tsp_path = nx.approximation.traveling_salesman_problem(G, cycle=False, method='greedy')
        route = [points[group[i]] for i in tsp_path]
        cee_routes.append(route)

    return cee_routes

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()
    REQUIRED_COLUMNS = ['SocietyID', 'Society Name', 'Latitude', 'Longitude', 'SocietyOrders', 'Current_CEEs']

    if not all(col in df.columns for col in REQUIRED_COLUMNS):
        st.error("Input file missing required columns. Expected: " + ", ".join(REQUIRED_COLUMNS))
        st.stop()

    df = df.dropna(subset=['Latitude', 'Longitude'])
    df['Latitude'] = df['Latitude'].astype(float)
    df['Longitude'] = df['Longitude'].astype(float)
    df['Current_CEEs'] = df['Current_CEEs'].astype(int)

    coords = df[['Latitude', 'Longitude']].to_numpy()
    kms_per_radian = 6371.0088
    epsilon = CLUSTER_RADIUS_KM / kms_per_radian

    db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine')
    db.fit(np.radians(coords))
    df['Cluster'] = db.labels_

    cluster_summary = df.groupby('Cluster').agg({
        'SocietyOrders': 'sum',
        'Latitude': 'mean',
        'Longitude': 'mean'
    }).reset_index()

    cluster_summary['ClusterType'] = cluster_summary['SocietyOrders'].apply(lambda x: 'Green' if x >= VEHICLE_MIN_ORDERS else 'Blue')
    cluster_summary['VehiclesRequired'] = (cluster_summary['SocietyOrders'] / VEHICLE_ORDER_CAPACITY).apply(np.ceil).astype(int)
    cluster_summary['CEEsRequired'] = (cluster_summary['SocietyOrders'] / CEE_CAPACITY).apply(np.ceil).astype(int)
    cluster_summary['TotalCost'] = cluster_summary['VehiclesRequired'] * VAN_COST + cluster_summary['CEEsRequired'] * CEE_COST
    cluster_summary['CostPerOrder'] = (cluster_summary['TotalCost'] / cluster_summary['SocietyOrders']).round(2)

    df = df.merge(cluster_summary[['Cluster', 'ClusterType']], on='Cluster', how='left')

    current_cee_alloc = df.groupby('Cluster')['Current_CEEs'].sum().reset_index().rename(columns={'Current_CEEs': 'Current_CEEs'})
    cluster_summary = cluster_summary.merge(current_cee_alloc, on='Cluster', how='left')
    cluster_summary['CEE_Delta'] = cluster_summary['CEEsRequired'] - cluster_summary['Current_CEEs']

    st.subheader("Cluster Summary")
    st.dataframe(cluster_summary)

    csv = cluster_summary.to_csv(index=False)
    st.download_button("Download Cluster Summary CSV", data=csv, file_name="cluster_summary.csv", mime='text/csv')
