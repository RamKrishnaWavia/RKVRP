import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster, AntPath
from streamlit_folium import st_folium
from sklearn.cluster import DBSCAN
from math import radians, cos, sin, asin, sqrt
import networkx as nx

st.set_page_config(layout="wide")
st.title("Milk Delivery Cluster & CEE Optimizer")

# Sidebar parameters
st.sidebar.header("Clustering & Capacity Settings")
VEHICLE_MIN_ORDERS = st.sidebar.number_input("Min Orders per Vehicle", value=150, step=10)
VEHICLE_MAX_ORDERS = st.sidebar.number_input("Max Orders per Vehicle", value=450, step=10)
CLUSTER_RADIUS_KM = st.sidebar.slider("Clustering Radius (km)", 0.1, 2.0, 0.5, 0.1)

st.sidebar.header("Costs and Capacities")
VAN_COST = st.sidebar.number_input("Van Cost per Month (₹)", value=25000, step=1000)
CEE_COST = st.sidebar.number_input("CEE Cost per Month (₹)", value=10000, step=1000)
CEE_CAPACITY = st.sidebar.number_input("CEE Order Capacity", value=200, step=10)
VEHICLE_ORDER_CAPACITY = st.sidebar.number_input("Vehicle Order Capacity", value=450, step=10)

st.sidebar.header("Depot Location (Editable)")
DEPOT_LAT = st.sidebar.number_input("Depot Latitude", value=12.9716, format="%.6f")
DEPOT_LON = st.sidebar.number_input("Depot Longitude", value=77.5946, format="%.6f")

# Upload CSV file
uploaded_file = st.file_uploader("Upload Society Orders CSV (with columns: SocietyID, Society Name, Latitude, Longitude, Orders, Current_CEEs)", type=["csv"])

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance in km between two points."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1 
    dlon = lon2 - lon1 
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    km = 6371.0088 * c
    return km

def auto_group_minimize_cees(cluster_df):
    """
    Group societies within cluster into CEE routes,
    respecting the CEE capacity.
    Returns list of routes (each route = list of points).
    """
    points = cluster_df[['Latitude', 'Longitude', 'Orders', 'SocietyID', 'Society Name']].values.tolist()
    unassigned = set(range(len(points)))
    cee_groups = []

    while unassigned:
        current = unassigned.pop()
        group = [current]
        group_order = points[current][2]

        # Find close points within radius
        close_points = []
        for idx in list(unassigned):
            dist = haversine_distance(points[current][0], points[current][1], points[idx][0], points[idx][1])
            if dist <= CLUSTER_RADIUS_KM:
                close_points.append((idx, dist))
        close_points.sort(key=lambda x: x[1])

        # Add points while respecting CEE capacity
        for idx, _ in close_points:
            if group_order + points[idx][2] <= CEE_CAPACITY:
                group.append(idx)
                group_order += points[idx][2]
                unassigned.remove(idx)

        cee_groups.append(group)

    # For each CEE group, find a TSP route order for visualization
    cee_routes = []
    for group in cee_groups:
        G = nx.complete_graph(len(group))
        for i in G.nodes:
            G.nodes[i]['coord'] = (points[group[i]][0], points[group[i]][1])
        for i, j in G.edges:
            c1 = G.nodes[i]['coord']
            c2 = G.nodes[j]['coord']
            G[i][j]['weight'] = haversine_distance(c1[0], c1[1], c2[0], c2[1])
        tsp_path = nx.approximation.traveling_salesman_problem(G, cycle=False, method='greedy')
        route = [points[group[i]] for i in tsp_path]
        cee_routes.append(route)

    return cee_groups, cee_routes

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()
    required_cols = ['SocietyID', 'Society Name', 'Latitude', 'Longitude', 'Orders', 'Current_CEEs']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
    else:
        # Clean and convert types
        df = df.dropna(subset=['Latitude', 'Longitude'])
        df['Latitude'] = df['Latitude'].astype(float)
        df['Longitude'] = df['Longitude'].astype(float)
        df['Orders'] = df['Orders'].astype(int)
        df['Current_CEEs'] = df['Current_CEEs'].astype(int)
        df['SocietyID'] = df['SocietyID'].astype(str)

        coords = df[['Latitude', 'Longitude']].to_numpy()
        kms_per_radian = 6371.0088
        epsilon = CLUSTER_RADIUS_KM / kms_per_radian

        db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine')
        db.fit(np.radians(coords))
        df['ClusterID'] = db.labels_

        # Cluster summary aggregation
        cluster_summary = df.groupby('ClusterID').agg({
            'Orders': 'sum',
            'Latitude': 'mean',
            'Longitude': 'mean',
            'Current_CEEs': 'sum'
        }).reset_index()

        cluster_summary['ClusterType'] = cluster_summary['Orders'].apply(lambda x: 'Green' if x >= VEHICLE_MIN_ORDERS else 'Blue')
        cluster_summary['VehiclesRequired'] = np.ceil(cluster_summary['Orders'] / VEHICLE_ORDER_CAPACITY).astype(int)
        cluster_summary['CEEsRequired_Revised'] = np.ceil(cluster_summary['Orders'] / CEE_CAPACITY).astype(int)
        cluster_summary['TotalCost'] = cluster_summary['VehiclesRequired'] * VAN_COST + cluster_summary['CEEsRequired_Revised'] * CEE_COST
        cluster_summary['CostPerOrder'] = (cluster_summary['TotalCost'] / cluster_summary['Orders']).round(2)
        cluster_summary['CEE_Allocated_Current'] = cluster_summary['Current_CEEs']

        # Merge back to df for cluster type etc
        df = df.merge(cluster_summary[['ClusterID', 'ClusterType']], on='ClusterID', how='left')

        # Folium Map Setup
        m = folium.Map(location=[DEPOT_LAT, DEPOT_LON], zoom_start=12)
        folium.Marker(location=[DEPOT_LAT, DEPOT_LON], popup="Depot", icon=folium.Icon(color='red', icon='home')).add_to(m)
        marker_cluster = MarkerCluster().add_to(m)
        colors = ['green', 'blue', 'red', 'orange', 'purple', 'darkred', 'lightred','beige', 'darkblue', 'darkgreen']

        # Draw societies as markers colored by cluster
        for _, row in df.iterrows():
            cluster_val = row['ClusterID']
            color = colors[cluster_val % len(colors)]
            folium.CircleMarker(
                location=(row['Latitude'], row['Longitude']),
                radius=5,
                popup=(f"{row['Society Name']} ({row['Orders']} orders), "
                       f"Cluster {row['ClusterID']} ({row['ClusterType']})"),
                color=color,
                fill=True,
                fill_opacity=0.7
            ).add_to(marker_cluster)

        # CEE grouping per cluster and route visualization
        df['CEE_Group'] = ""
        cee_route_summary = []
        for cluster_id in df['ClusterID'].unique():
            cluster_df = df[df['ClusterID'] == cluster_id].copy()
            cee_groups, cee_routes = auto_group_minimize_cees(cluster_df)
            # Assign group IDs and prepare summary
            for idx, group in enumerate(cee_groups):
                group_societies = cluster_df.iloc[group]
                group_order_sum = group_societies['Orders'].sum()
                # Assign CEE group label to societies in df
                for sid in group_societies['SocietyID']:
                    df.loc[df['SocietyID'] == sid, 'CEE_Group'] = f"{cluster_id}_{idx+1}"
                cee_route_summary.append({
                    'ClusterID': cluster_id,
                    'CEE_Group': f"{cluster_id}_{idx+1}",
                    'Societies_Count': len(group),
                    'Total_Orders': group_order_sum,
                    'CEE_Required': 1  # Each group = 1 CEE needed since capped by capacity
                })
                # Draw route on map
                route_points = [(p[0], p[1]) for p in cee_routes[idx]]
                AntPath(route_points, color='blue', delay=1000).add_to(m)

        cee_route_df = pd.DataFrame(cee_route_summary)

        # Society-wise Cost per Order calculation
        df = df.merge(cluster_summary[['ClusterID', 'TotalCost', 'Orders']], on='ClusterID', how='left')
        df['Society_CostPerOrder'] = (df['TotalCost'] * (df['Orders'] / df['Orders'])).round(2)  # proportionate cost

        # Display outputs
        st.subheader("Cluster Summary")
        st.dataframe(cluster_summary)

        st.subheader("CEE Route Summary (Clusters & Groups)")
        st.dataframe(cee_route_df)

        st.subheader("Society-wise Cluster Mapping and CEE Group")
        st.dataframe(df[['SocietyID', 'Society Name', 'Latitude', 'Longitude', 'Orders', 'Current_CEEs', 'ClusterID', 'ClusterType', 'CEE_Group']])

        st.subheader("Interactive Map (Depot, Societies, CEE Routes)")
        st_folium(m, width=1000, height=600)

        # Download options
        st.download_button(
            label="Download Cluster Summary CSV",
            data=cluster_summary.to_csv(index=False),
            file_name='cluster_summary.csv',
            mime='text/csv'
        )
        st.download_button(
            label="Download CEE Route Summary CSV",
            data=cee_route_df.to_csv(index=False),
            file_name='cee_route_summary.csv',
            mime='text/csv'
        )
        st.download_button(
            label="Download Society-wise Mapping CSV",
            data=df[['SocietyID', 'Society Name', 'Latitude', 'Longitude', 'Orders', 'Current_CEEs', 'ClusterID', 'ClusterType', 'CEE_Group']].to_csv(index=False),
            file_name='society_cluster_cee_mapping.csv',
            mime='text/csv'
        )
else:
    st.info("Please upload a valid CSV file to proceed.")
