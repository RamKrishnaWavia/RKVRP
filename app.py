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
st.title("RK - Delivery Cluster Optimizer")

# Sidebar parameters
st.sidebar.header("Clustering Settings")
VEHICLE_MIN_ORDERS = st.sidebar.number_input("Min Orders per Vehicle", value=150, step=10)
VEHICLE_MAX_ORDERS = st.sidebar.number_input("Max Orders per Vehicle", value=450, step=10)
CLUSTER_RADIUS_KM = st.sidebar.slider("Clustering Radius (km)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)

# Depot Location
st.sidebar.header("Depot Location")
DEPOT_LAT = st.sidebar.number_input("Depot Latitude", value=13.068218288167737, format="%f")
DEPOT_LON = st.sidebar.number_input("Depot Longitude", value=77.44607278434877, format="%f")

# User-editable costs and capacities
VAN_COST = st.sidebar.number_input("Van Cost per Month", value=25000, step=1000)
CEE_COST = st.sidebar.number_input("CEE Cost per Month", value=10000, step=1000)
CEE_CAPACITY = st.sidebar.number_input("CEE Min Orders", value=200, step=10)
VEHICLE_ORDER_CAPACITY = st.sidebar.number_input("Vehicle Order Capacity", value=450, step=10)

# Upload CSV file
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
    points = cluster_df[['Latitude', 'Longitude', 'SocietyOrders', 'Society Name']].values.tolist()
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
        route = [points[idx] for idx in group]
        cee_routes.append(route)

    return cee_routes

def optimize_route(points):
    G = nx.complete_graph(len(points))
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist = haversine_distance(points[i][0], points[i][1], points[j][0], points[j][1])
            G[i][j]['weight'] = dist
            G[j][i]['weight'] = dist

    tsp_route = nx.approximation.traveling_salesman_problem(G, cycle=False)
    return tsp_route

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    df = df.dropna(subset=['Latitude', 'Longitude'])
    df['Latitude'] = df['Latitude'].astype(float)
    df['Longitude'] = df['Longitude'].astype(float)

    coords = df[['Latitude', 'Longitude']].to_numpy()
    kms_per_radian = 6371.0088
    epsilon = CLUSTER_RADIUS_KM / kms_per_radian

    db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine')
    db.fit(np.radians(coords))
    df['Cluster'] = db.labels_

    cluster_summary = df.groupby('Cluster').agg({
        'Orders': 'sum',
        'Latitude': 'mean',
        'Longitude': 'mean'
    }).reset_index()

    cluster_summary['DistanceFromDepotKM'] = cluster_summary.apply(lambda row: round(haversine_distance(DEPOT_LAT, DEPOT_LON, row['Latitude'], row['Longitude']), 2), axis=1)
    cluster_summary['ClusterType'] = cluster_summary['Orders'].apply(lambda x: 'Green' if x >= VEHICLE_MIN_ORDERS else 'Blue')

    cluster_summary['VehiclesRequired'] = (cluster_summary['Orders'] / VEHICLE_ORDER_CAPACITY).apply(np.ceil).astype(int)
    cluster_summary['CEEsRequired'] = (cluster_summary['Orders'] / CEE_CAPACITY).apply(np.ceil).astype(int)
    cluster_summary['TotalCost'] = cluster_summary['VehiclesRequired'] * VAN_COST + cluster_summary['CEEsRequired'] * CEE_COST
    cluster_summary['CostPerOrder'] = (cluster_summary['TotalCost'] / cluster_summary['Orders']).round(2)

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
            popup=f"{row['Society Name']} ({row['Orders']} orders)\nCluster {row['Cluster']} ({row['ClusterType']})",
            color=color,
            fill=True,
            fill_opacity=0.7
        ).add_to(marker_cluster)

    df['SocietyID'] = df.index
    df = df.rename(columns={'Orders': 'SocietyOrders', 'Cluster': 'ClusterID'})

    optimized_routes = []
    all_cee_routes = []
    for cluster_id in df['ClusterID'].unique():
        cluster_df = df[df['ClusterID'] == cluster_id].copy()
        routes = auto_group_minimize_cees(cluster_df)
        for i, route in enumerate(routes):
            if len(route) > 1:
                tsp_order = optimize_route(route)
                route = [route[j] for j in tsp_order]

            route_points = [(DEPOT_LAT, DEPOT_LON)] + [(p[0], p[1]) for p in route] + [(DEPOT_LAT, DEPOT_LON)]
            AntPath(route_points, color='blue', delay=1000).add_to(m)

            for seq, p in enumerate(route, start=1):
                society_name = p[3]
                df.loc[df['Society Name'] == society_name, 'CEE_Group'] = f'{cluster_id}_{i+1}'
                df.loc[df['Society Name'] == society_name, 'RouteSeq'] = seq
                optimized_routes.append({
                    'ClusterID': cluster_id,
                    'CEE_Group': f'{cluster_id}_{i+1}',
                    'Society Name': society_name,
                    'Latitude': p[0],
                    'Longitude': p[1],
                    'Sequence': seq
                })

            all_cee_routes.append(route)

    st_data = st_folium(m, width=1000, height=600)

    df = df.merge(cluster_summary[['Cluster', 'TotalCost', 'Orders']].rename(columns={'Cluster': 'ClusterID'}), on='ClusterID', how='left')
    df['Society_CostPerOrder'] = (df['TotalCost'] * (df['SocietyOrders'] / df['Orders'])).round(2)

    st.subheader("Cluster Summary")
    st.dataframe(cluster_summary)

    st.subheader("Overall Summary")
    total_orders = cluster_summary['Orders'].sum()
    total_cost = cluster_summary['TotalCost'].sum()
    overall_cpo = round(total_cost / total_orders, 2) if total_orders > 0 else 0
    st.write(f"**Total Orders:** {total_orders}")
    st.write(f"**Total Cost:** ₹{total_cost}")
    st.write(f"**Overall Cost per Order (CPO):** ₹{overall_cpo}")

    csv = cluster_summary.to_csv(index=False)
    st.download_button(
        label="Download Cluster Summary CSV",
        data=csv,
        file_name='cluster_summary.csv',
        mime='text/csv'
    )

    society_csv = df[[
        'SocietyID', 'Society Name', 'Latitude', 'Longitude', 'SocietyOrders',
        'ClusterID', 'ClusterType', 'CEE_Group', 'Society_CostPerOrder', 'RouteSeq'
    ]].to_csv(index=False)

    st.download_button(
        label="Download Society-wise Cluster CSV",
        data=society_csv,
        file_name='society_cluster_mapping.csv',
        mime='text/csv'
    )

    if optimized_routes:
        routes_df = pd.DataFrame(optimized_routes)
        route_csv = routes_df.to_csv(index=False)
        st.download_button(
            label="Download Optimized Route Sequences",
            data=route_csv,
            file_name='optimized_routes.csv',
            mime='text/csv'
        )
