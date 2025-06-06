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
VEHICLE_MIN_ORDERS = st.sidebar.number_input("Min Orders per Vehicle", value=150, step=10)
VEHICLE_MAX_ORDERS = st.sidebar.number_input("Max Orders per Vehicle", value=450, step=10)
CLUSTER_RADIUS_KM = st.sidebar.slider("Clustering Radius (km)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)

# Cost & capacity settings
st.sidebar.header("Cost and Capacity Settings")
VAN_COST = st.sidebar.number_input("Van Cost per Month", value=25000, step=1000)
CEE_COST = st.sidebar.number_input("CEE Cost per Month", value=10000, step=1000)
CEE_CAPACITY = st.sidebar.number_input("CEE Order Capacity", value=200, step=10)
VEHICLE_ORDER_CAPACITY = st.sidebar.number_input("Vehicle Order Capacity", value=450, step=10)

# Optional user-defined depot lat/lon
st.sidebar.header("Optional Depot Location")
user_lat = st.sidebar.text_input("Depot Latitude (optional)")
user_lon = st.sidebar.text_input("Depot Longitude (optional)")

# File upload
uploaded_file = st.file_uploader("Upload Society Orders CSV", type=["csv"])

# Distance function
def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return 6371.0088 * 2 * asin(sqrt(a))

# CEE route grouping logic
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
        tsp_path = nx.approximation.traveling_salesman_problem(G, cycle=False, method='greedy')
        route = [points[group[i]] for i in tsp_path]
        cee_routes.append(route)

    return cee_routes

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    df = df.dropna(subset=['Latitude', 'Longitude'])
    df['Latitude'] = df['Latitude'].astype(float)
    df['Longitude'] = df['Longitude'].astype(float)
    df['SocietyOrders'] = df['SocietyOrders'].astype(int)
    df['CEEsAllocated'] = df.get('CEEsAllocated', 0)

    df['SocietyID'] = df['SocietyID'].astype(str)

    coords = df[['Latitude', 'Longitude']].to_numpy()
    epsilon = CLUSTER_RADIUS_KM / 6371.0088

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

    m = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=13)
    marker_cluster = MarkerCluster().add_to(m)

    colors = ['green', 'blue', 'red', 'orange', 'purple', 'darkred', 'lightred','beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white','pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray']

    df['Cluster'] = df['Cluster'].fillna(-1).astype(int)

    for _, row in df.iterrows():
        cluster_val = row['Cluster']
        color = colors[cluster_val % len(colors)]
        folium.CircleMarker(
            location=(row['Latitude'], row['Longitude']),
            radius=5,
            popup=f"{row['SocietyID']} ({row['SocietyOrders']} orders)\nCluster {row['Cluster']} ({row['ClusterType']})",
            color=color,
            fill=True,
            fill_opacity=0.7
        ).add_to(marker_cluster)

    all_cee_routes = []
    cee_allocation_summary = []

    for cluster_id in df['Cluster'].unique():
        cluster_df = df[df['Cluster'] == cluster_id].copy()
        routes = auto_group_minimize_cees(cluster_df)

        for i, route in enumerate(routes):
            path = [(p[0], p[1]) for p in route]
            AntPath(path, color='blue', delay=800).add_to(m)
            route_order_sum = sum([p[2] for p in route])
            cee_allocation_summary.append({'ClusterID': cluster_id, 'CEE Route': f'{cluster_id}_{i+1}', 'Orders': route_order_sum})
            for p in route:
                df.loc[df['SocietyID'] == str(p[3]), 'CEE_Group'] = f'{cluster_id}_{i+1}'

        all_cee_routes.extend(routes)

    st_data = st_folium(m, width=1000, height=600)

    # Final society-level CPO calculation
    df = df.merge(cluster_summary[['Cluster', 'TotalCost', 'SocietyOrders']].rename(columns={'Cluster': 'ClusterID'}), on='ClusterID', how='left')
    df['Society_CostPerOrder'] = (df['TotalCost'] * (df['SocietyOrders'] / df['SocietyOrders'])).round(2)

    # Overall summary
    st.subheader("Cluster Summary")
    st.dataframe(cluster_summary)

    st.subheader("CEE Route Allocation Summary")
    cee_summary_df = pd.DataFrame(cee_allocation_summary)
    st.dataframe(cee_summary_df)

    st.subheader("Overall Summary")
    total_orders = cluster_summary['SocietyOrders'].sum()
    total_cost = cluster_summary['TotalCost'].sum()
    overall_cpo = round(total_cost / total_orders, 2) if total_orders > 0 else 0

    st.write(f"**Total Orders:** {total_orders}")
    st.write(f"**Total Cost:** ₹{total_cost}")
    st.write(f"**Overall CPO:** ₹{overall_cpo}")

    # Download cluster-level summary
    st.download_button("Download Cluster Summary", cluster_summary.to_csv(index=False), file_name="cluster_summary.csv", mime="text/csv")

    # Download society-level summary
    df_export = df[[
        'SocietyID', 'Society Name', 'Latitude', 'Longitude', 'SocietyOrders',
        'ClusterID', 'ClusterType', 'CEE_Group', 'Society_CostPerOrder', 'CEEsAllocated'
    ]]
    st.download_button("Download Society-wise Cluster CSV", df_export.to_csv(index=False), file_name="society_cluster_mapping.csv", mime="text/csv")

    # Download route-level summary
    st.download_button("Download CEE Allocation Summary", cee_summary_df.to_csv(index=False), file_name="cee_allocation_summary.csv", mime="text/csv")
