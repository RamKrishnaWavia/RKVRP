import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.cluster import DBSCAN
from geopy.distance import geodesic

# Helper to calculate distance in km
def haversine(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).km

# Sidebar input for depot coordinates
st.sidebar.title("Depot Location")
depot_lat = st.sidebar.number_input("Depot Latitude", value=12.9352)
depot_lon = st.sidebar.number_input("Depot Longitude", value=77.6146)

# Upload CSV
st.title("Milk Delivery Route Optimizer")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data")
    st.dataframe(df)

    # Clustering using DBSCAN with 2km radius (~0.018 radians)
    coords = df[["Latitude", "Longitude"]].to_numpy()
    kms_per_radian = 6371.0088
    epsilon = 2 / kms_per_radian
    db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
    df["Cluster_ID"] = db.labels_

    # Filter clusters with minimum 200 orders
    cluster_summary = df.groupby("Cluster_ID")["Orders"].sum().reset_index()
    valid_clusters = cluster_summary[cluster_summary["Orders"] >= 200]["Cluster_ID"]
    df = df[df["Cluster_ID"].isin(valid_clusters)]

    # TSP route optimization within each cluster
    route_results = []
    for cluster_id in df["Cluster_ID"].unique():
        cluster_df = df[df["Cluster_ID"] == cluster_id].copy()
        points = [(depot_lat, depot_lon)] + cluster_df[["Latitude", "Longitude"]].to_numpy().tolist()

        G = nx.complete_graph(len(points))
        for i in range(len(points)):
            for j in range(len(points)):
                if i != j:
                    dist = haversine(points[i][0], points[i][1], points[j][0], points[j][1])
                    G[i][j]["weight"] = dist

        tsp_path = nx.approximation.traveling_salesman_problem(G, cycle=False, method="greedy")
        delivery_seq = []
        for i, idx in enumerate(tsp_path[1:], 1):  # skip depot
            row = cluster_df.iloc[idx - 1]
            delivery_seq.append({
                "Cluster_ID": cluster_id,
                "Society_ID": row["Society_ID"],
                "Society_Name": row["Society_Name"],
                "Latitude": row["Latitude"],
                "Longitude": row["Longitude"],
                "Orders": row["Orders"],
                "Delivery_Sequence": f"S{i}"
            })

        route_results.extend(delivery_seq)

    routed_df = pd.DataFrame(route_results)
    st.subheader("Route Plan")
    st.dataframe(routed_df)

    # Download CSV
    csv = routed_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Route Plan CSV", csv, "routed_plan.csv", "text/csv")
