import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from shapely.geometry import MultiPoint
import networkx as nx
from io import StringIO

st.set_page_config(layout="wide")
st.title("🚚 Milk Delivery Cluster Optimizer")

# Sidebar: Upload and settings
st.sidebar.header("📥 Upload Input File")
uploaded_file = st.sidebar.file_uploader("Choose CSV file", type="csv")

st.sidebar.header("📍 Depot Coordinates")
depot_lat = st.sidebar.number_input("Depot Latitude", value=12.9716, format="%.6f")
depot_lon = st.sidebar.number_input("Depot Longitude", value=77.5946, format="%.6f")

st.sidebar.markdown("📄 [Download Input Template](https://raw.githubusercontent.com/openai/templates/main/milk_clusters_template.csv)", unsafe_allow_html=True)

# Constants
MIN_ORDERS = 200
MAX_ORDERS = 225
DIST_THRESHOLD_KM = 2
EPS_RAD = DIST_THRESHOLD_KM / 6371.0088  # Earth's radius in km

def cluster_data(df):
    coords = df[['Latitude', 'Longitude']].to_numpy()
    kms_per_radian = 6371.0088
    db = DBSCAN(eps=EPS_RAD, min_samples=1, algorithm='ball_tree', metric='haversine')
    labels = db.fit_predict(np.radians(coords))
    df['Cluster_ID'] = labels
    return df

def filter_clusters(df):
    cluster_summary = df.groupby("Cluster_ID")["Orders"].sum().reset_index()
    valid_clusters = cluster_summary[
        (cluster_summary["Orders"] >= MIN_ORDERS) & 
        (cluster_summary["Orders"] <= MAX_ORDERS)
    ]["Cluster_ID"]
    return df[df["Cluster_ID"].isin(valid_clusters)].copy()

def build_tsp_path(df_cluster, depot):
    try:
        locations = [(depot[0], depot[1])] + df_cluster[['Latitude', 'Longitude']].values.tolist()
        G = nx.complete_graph(len(locations))
        for i in range(len(locations)):
            for j in range(len(locations)):
                if i != j:
                    dist = great_circle(locations[i], locations[j]).km
                    G[i][j]['weight'] = dist
        path = nx.approximation.traveling_salesman_problem(G, cycle=False, method="greedy")
        return path[1:]  # exclude depot
    except Exception as e:
        st.error(f"❌ TSP failed for cluster. Error:\n{e}")
        return list(range(len(df_cluster)))

def show_map(df, depot):
    m = folium.Map(location=depot, zoom_start=12)
    cluster_colors = folium.plugins.FastMarkerCluster(df[['Latitude', 'Longitude']].values.tolist())
    
    cluster_ids = df['Cluster_ID'].unique()
    color_palette = ['red', 'blue', 'green', 'orange', 'purple', 'darkred', 'cadetblue', 'darkblue']
    
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_df = df[df['Cluster_ID'] == cluster_id]
        tsp_sequence = build_tsp_path(cluster_df, depot)
        ordered_df = cluster_df.iloc[tsp_sequence].reset_index(drop=True)
        for i, row in ordered_df.iterrows():
            folium.Marker(
                location=[row['Latitude'], row['Longitude']],
                popup=f"{row['Society_Name']} ({row['Orders']} orders)",
                tooltip=f"S{i+1} - {row['Society_Name']}",
                icon=folium.Icon(color=color_palette[idx % len(color_palette)])
            ).add_to(m)
        # draw route
        route_coords = [[row['Latitude'], row['Longitude']] for _, row in ordered_df.iterrows()]
        folium.PolyLine(route_coords, color=color_palette[idx % len(color_palette)], weight=2.5).add_to(m)

    folium.Marker(location=depot, icon=folium.Icon(color='black', icon='home'), popup="Depot").add_to(m)
    st_folium(m, width=1000)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['Latitude'] = df['Latitude'].astype(float)
    df['Longitude'] = df['Longitude'].astype(float)
    df['Orders'] = df['Orders'].astype(int)
    
    st.subheader("📊 Raw Data Preview")
    st.dataframe(df)

    clustered_df = cluster_data(df)
    filtered_df = filter_clusters(clustered_df)

    st.subheader("📦 Valid Clusters (Orders Between 200–225)")
    st.dataframe(filtered_df)

    st.subheader("🗺️ Route Map")
    show_map(filtered_df, depot=(depot_lat, depot_lon))

    # Delivery sequence assignment
    results = []
    for cluster_id in filtered_df['Cluster_ID'].unique():
        cluster_df = filtered_df[filtered_df['Cluster_ID'] == cluster_id]
        tsp_sequence = build_tsp_path(cluster_df, depot=(depot_lat, depot_lon))
        ordered_df = cluster_df.iloc[tsp_sequence].reset_index(drop=True)
        ordered_df['Delivery_Sequence'] = ['S' + str(i+1) for i in range(len(ordered_df))]
        results.append(ordered_df)
    
    final_df = pd.concat(results)
    st.subheader("📄 Final Delivery Plan")
    st.dataframe(final_df[['Society_ID', 'Society_Name', 'Cluster_ID', 'Delivery_Sequence', 'Orders']])

    csv_output = final_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Delivery Plan CSV", data=csv_output, file_name="delivery_plan.csv", mime='text/csv')

else:
    st.info("👆 Upload your input CSV file to proceed.")

