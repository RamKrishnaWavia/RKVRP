import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from shapely.geometry import MultiPoint
import folium
from streamlit_folium import st_folium
from io import StringIO

# Helper function to calculate distance matrix
def create_distance_matrix(coords):
    n = len(coords)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = great_circle(coords[i], coords[j]).meters
            dist_matrix[i][j] = dist
            dist_matrix[j][i] = dist
    return dist_matrix

# Helper to calculate route distance
def calculate_route_distance(route):
    distance = 0.0
    for i in range(len(route) - 1):
        distance += great_circle(route[i], route[i+1]).meters
    return distance

st.title("Milk & Grocery Delivery Clustering Tool")

# Template file download
st.subheader("Download Template")
template = pd.DataFrame({"Society": [], "Latitude": [], "Longitude": [], "Orders": []})
st.download_button("Download Template CSV", data=template.to_csv(index=False), file_name="society_template.csv")

# File upload
st.subheader("Upload Society Data")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data", df)

    coords = list(zip(df['Latitude'], df['Longitude']))

    # DBSCAN clustering with max 2km distance (2000 meters)
    kms_per_radian = 6371.0088
    epsilon = 2 / kms_per_radian

    db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
    labels = db.labels_
    df['Cluster'] = labels

    cluster_summary = []
    cluster_map = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=12)

    for label in set(labels):
        cluster_df = df[df['Cluster'] == label]
        total_orders = cluster_df['Orders'].sum()
        if 200 <= total_orders <= 220:
            # Add markers to map
            coords = list(zip(cluster_df['Latitude'], cluster_df['Longitude']))
            for _, row in cluster_df.iterrows():
                folium.Marker(
                    location=[row['Latitude'], row['Longitude']],
                    popup=f"{row['Society']} ({row['Orders']} orders)",
                    icon=folium.Icon(color='blue', icon='info-sign')
                ).add_to(cluster_map)

            route = list(zip(cluster_df['Latitude'], cluster_df['Longitude']))
            distance = calculate_route_distance(route)

            cluster_summary.append({
                "Cluster": label,
                "Societies": ", ".join(cluster_df['Society'].tolist()),
                "No. of Societies": len(cluster_df),
                "Total Orders": total_orders,
                "Total Distance (m)": round(distance, 2)
            })

    # Show map
    st.subheader("Cluster Map")
    st_data = st_folium(cluster_map, width=725)

    # Show cluster summary
    summary_df = pd.DataFrame(cluster_summary)
    st.subheader("Cluster Summary")
    st.dataframe(summary_df)

    # Download summary
    csv = summary_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Cluster Summary CSV", data=csv, file_name="cluster_summary.csv")
