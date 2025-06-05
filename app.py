import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from sklearn.cluster import DBSCAN
from shapely.geometry import MultiPoint
from math import radians, cos, sin, asin, sqrt
import io

st.set_page_config(layout="wide")
st.title("Milk Delivery Cluster Optimizer")

# Sidebar parameters
st.sidebar.header("Clustering Settings")
VEHICLE_MIN_ORDERS = st.sidebar.number_input("Min Orders per Vehicle", value=150, step=10)
VEHICLE_MAX_ORDERS = st.sidebar.number_input("Max Orders per Vehicle", value=450, step=10)
CLUSTER_RADIUS_KM = st.sidebar.slider("Clustering Radius (km)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)

# User-editable costs and capacities
VAN_COST = st.sidebar.number_input("Van Cost per Month", value=25000, step=1000)
CEE_COST = st.sidebar.number_input("CEE Cost per Month", value=10000, step=1000)
CEE_CAPACITY = st.sidebar.number_input("CEE Max Orders", value=200, step=10)
VEHICLE_ORDER_CAPACITY = st.sidebar.number_input("Vehicle Order Capacity", value=450, step=10)

# Upload CSV file
uploaded_file = st.file_uploader("Upload Society Orders CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Drop rows with missing coordinates
    df = df.dropna(subset=['Latitude', 'Longitude'])

    # Convert coordinates to float
    df['Latitude'] = df['Latitude'].astype(float)
    df['Longitude'] = df['Longitude'].astype(float)

    # Prepare coordinates for clustering
    coords = df[['Latitude', 'Longitude']].to_numpy()
    kms_per_radian = 6371.0088
    epsilon = CLUSTER_RADIUS_KM / kms_per_radian

    db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine')
    db.fit(np.radians(coords))
    df['Cluster'] = db.labels_

    # Aggregate by clusters
    cluster_summary = df.groupby('Cluster').agg({
        'Orders': 'sum',
        'Latitude': 'mean',
        'Longitude': 'mean'
    }).reset_index()

    # Assign cluster types
    def cluster_type(orders):
        if orders >= VEHICLE_MIN_ORDERS:
            return 'Green'
        else:
            return 'Blue'

    cluster_summary['ClusterType'] = cluster_summary['Orders'].apply(cluster_type)

    # Calculate vehicles and cee needed
    cluster_summary['VehiclesRequired'] = (cluster_summary['Orders'] / VEHICLE_ORDER_CAPACITY).apply(np.ceil).astype(int)
    cluster_summary['CEEsRequired'] = (cluster_summary['Orders'] / CEE_CAPACITY).apply(np.ceil).astype(int)
    cluster_summary['TotalCost'] = cluster_summary['VehiclesRequired'] * VAN_COST + cluster_summary['CEEsRequired'] * CEE_COST
    cluster_summary['CostPerOrder'] = (cluster_summary['TotalCost'] / cluster_summary['Orders']).round(2)

    # Merge back to original data for map
    df = df.merge(cluster_summary[['Cluster', 'ClusterType']], on='Cluster', how='left')

    # Display map
    m = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=13)
    marker_cluster = MarkerCluster().add_to(m)

    colors = ['green', 'blue', 'red', 'orange', 'purple', 'darkred', 'lightred',
              'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white',
              'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray']

    df['Cluster'] = df['Cluster'].fillna(-1).astype(int)

    for i, row in df.iterrows():
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

    st_data = st_folium(m, width=1000, height=600)

    # Calculate society-wise cost per order
    df = df.merge(cluster_summary[['Cluster', 'TotalCost', 'Orders']], on='Cluster', suffixes=('', '_Cluster'), how='left')
    df['Society_CostPerOrder'] = (df['TotalCost'] * (df['Orders'] / df['Orders_Cluster'])).round(2)

    # Final rename
    df = df.rename(columns={
        'Orders': 'SocietyOrders',
        'Cluster': 'ClusterID'
    })

    # Display summary in app
    st.subheader("Cluster Summary")
    st.dataframe(cluster_summary)

    # Final Summary Stats
    st.subheader("Overall Summary")
    total_orders = cluster_summary['Orders'].sum()
    total_cost = cluster_summary['TotalCost'].sum()
    overall_cpo = round(total_cost / total_orders, 2) if total_orders > 0 else 0
    st.write(f"**Total Orders:** {total_orders}")
    st.write(f"**Total Cost:** ₹{total_cost}")
    st.write(f"**Overall Cost per Order (CPO):** ₹{overall_cpo}")

    # Downloadable cluster summary
    csv = cluster_summary.to_csv(index=False)
    st.download_button(
        label="Download Cluster Summary CSV",
        data=csv,
        file_name='cluster_summary.csv',
        mime='text/csv'
    )

    # Downloadable full society-cluster data
    society_csv = df[[
        'Society Name', 'Latitude', 'Longitude', 'SocietyOrders',
        'ClusterID', 'ClusterType', 'Society_CostPerOrder'
    ]].to_csv(index=False)

    st.download_button(
        label="Download Society-wise Cluster CSV",
        data=society_csv,
        file_name='society_cluster_mapping.csv',
        mime='text/csv'
    )
