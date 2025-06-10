
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from geopy.point import Point

# Sidebar inputs
st.sidebar.header("Depot Location")
depot_lat = st.sidebar.number_input("Depot Latitude", value=12.9352)
depot_lon = st.sidebar.number_input("Depot Longitude", value=77.6101)

# File upload
uploaded_file = st.file_uploader("Upload Milk Delivery CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Raw Data", df)

    # Clustering with proximity < 2km and minimum 200 orders per cluster
    coords = df[["Latitude", "Longitude"]].to_numpy()
    kms_per_radian = 6371.0088
    epsilon = 2 / kms_per_radian

    db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
    df["Cluster_ID"] = db.labels_

    # Filter clusters with at least 200 orders
    cluster_orders = df.groupby("Cluster_ID")["Orders"].sum().reset_index()
    valid_clusters = cluster_orders[cluster_orders["Orders"] >= 200]["Cluster_ID"]
    df = df[df["Cluster_ID"].isin(valid_clusters)]

    # Sort by distance to depot
    df["Distance_from_depot"] = df.apply(lambda row: great_circle(
        (depot_lat, depot_lon), (row["Latitude"], row["Longitude"])).km, axis=1)
    df = df.sort_values(by=["Cluster_ID", "Distance_from_depot"])

    # Add sequence
    df["Sequence"] = df.groupby("Cluster_ID").cumcount() + 1
    df["Sequence_Label"] = df["Sequence"].apply(lambda x: f"S{x}")

    # Show output
    st.write("Clustered and Sequenced Data", df[[
             "Society_ID", "Society_Name", "Latitude", "Longitude", "Orders", "Cluster_ID", "Sequence_Label"]])

    # Download output
    df.to_csv("/mnt/data/milk_delivery_output.csv", index=False)
    st.download_button("Download Output CSV", "/mnt/data/milk_delivery_output.csv", file_name="milk_delivery_output.csv")
