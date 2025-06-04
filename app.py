import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from sklearn.cluster import DBSCAN
import io

st.set_page_config(layout="wide")
st.title("Milk Delivery Cluster Optimizer")

# Sidebar parameters
st.sidebar.header("Clustering Settings")
VEHICLE_MIN_ORDERS = st.sidebar.number_input("Min Orders per Vehicle", value=150, step=10)
VEHICLE_MAX_ORDERS = st.sidebar.number_input("Max Orders per Vehicle", value=450, step=10)
VAN_COST_PER_MONTH = 25000
CEE_COST_PER_MONTH = 10000
CEE_MAX_ORDERS = 200
CLUSTER_RADIUS_KM = st.sidebar.slider("Clustering Radius (km)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)

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

    # Calculate vans and cees required per cluster
    cluster_summary['VansRequired'] = (cluster_summary['Orders'] / VEHICLE_MAX_ORDERS).apply(np.ceil).astype(int)
    cluster_summary['CEEsRequired'] = (cluster_summary['Orders'] / CEE_MAX_ORDERS).apply(np.ceil).astype(int)

    # Calculate cost per cluster
    cluster_summary['TotalCost'] = (cluster_summary['VansRequired'] * VAN_COST_PER_MONTH) + (cluster_summary['CEEsRequired'] * CEE_COST_PER_MONTH)

    # Cost per order rounded
    cluster_summary['CostPerOrder'] = (cluster_summary['TotalCost'] / cluster_summary['Orders']).round(2)

    # Merge back to original data for map coloring
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

    st_folium(m, width=1000, height=600)

    # Display cluster summary
    st.subheader("Cluster Summary")
    st.dataframe(cluster_summary)

    # Final overall summary
    total_orders = cluster_summary['Orders'].sum()
    total_vans = cluster_summary['VansRequired'].sum()
    total_cees = cluster_summary['CEEsRequired'].sum()
    total_cost = cluster_summary['TotalCost'].sum()
    overall_cpo = round(total_cost / total_orders, 2) if total_orders > 0 else 0

    final_summary_df = pd.DataFrame({
        'Total Orders': [total_orders],
        'Total Vans Required': [total_vans],
        'Total CEEs Required': [total_cees],
        'Total Cost (₹)': [total_cost],
        'Overall Cost Per Order (₹)': [overall_cpo]
    })

    st.subheader("Final Overall Summary")
    st.dataframe(final_summary_df)

    # Downloadable cluster summary CSV
    csv = cluster_summary.to_csv(index=False)
    st.download_button(
        label="Download Cluster Summary CSV",
        data=csv,
        file_name='cluster_summary.csv',
        mime='text/csv'
    )

    # Downloadable full society-cluster mapping CSV
    society_csv = df.to_csv(index=False)
    st.download_button(
        label="Download Society-wise Cluster CSV",
        data=society_csv,
        file_name='society_cluster_mapping.csv',
        mime='text/csv'
    )
