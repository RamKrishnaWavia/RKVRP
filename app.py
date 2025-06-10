import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.cluster import DBSCAN
from haversine import haversine, Unit
from itertools import count

# Sidebar depot settings
st.sidebar.header("Depot Location (Editable)")
depot_lat = st.sidebar.number_input("Depot Latitude", value=12.935, format="%.6f")
depot_lon = st.sidebar.number_input("Depot Longitude", value=77.614, format="%.6f")

# File upload
uploaded_file = st.file_uploader("Upload Input CSV", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Required columns check
    required_cols = {"Society_ID", "Society_Name", "Latitude", "Longitude", "Orders"}
    if not required_cols.issubset(df.columns):
        st.error(f"CSV must include: {', '.join(required_cols)}")
    else:
        # Clustering
        coords = df[["Latitude", "Longitude"]].values
        kms_per_radian = 6371.0088
        epsilon = 2 / kms_per_radian  # 2 km radius

        db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine')
        df["Cluster_ID"] = db.fit_predict(np.radians(coords))

        # Minimum 200 orders filter
        cluster_summary = df.groupby("Cluster_ID")["Orders"].sum().reset_index()
        valid_clusters = cluster_summary[cluster_summary["Orders"] >= 200]["Cluster_ID"]
        df["Valid_Cluster"] = df["Cluster_ID"].apply(lambda x: x if x in valid_clusters.values else -1)

        cee_routes = []
        cee_counter = count(start=1)
        final_routes = []

        for cluster in sorted(df["Valid_Cluster"].unique()):
            cluster_df = df[df["Valid_Cluster"] == cluster].copy()
            if cluster == -1:
                for _, row in cluster_df.iterrows():
                    row["CEE_ID"] = next(cee_counter)
                    row["Delivery_Sequence"] = "S1"
                    final_routes.append(row)
                continue

            locations = [(row["Latitude"], row["Longitude"]) for _, row in cluster_df.iterrows()]
            depot = (depot_lat, depot_lon)
            G = nx.Graph()
            for i, loc1 in enumerate(locations):
                G.add_node(i, pos=loc1)
                for j, loc2 in enumerate(locations):
                    if i != j:
                        dist = haversine(loc1, loc2, unit=Unit.KILOMETERS)
                        G.add_edge(i, j, weight=dist)

            tsp_path = nx.approximation.traveling_salesman_problem(G, cycle=False, method="greedy")

            group_id = next(cee_counter)
            for seq, idx in enumerate(tsp_path):
                row = cluster_df.iloc[idx].copy()
                row["CEE_ID"] = group_id
                row["Delivery_Sequence"] = f"S{seq+1}"
                final_routes.append(row)

        routed_df = pd.DataFrame(final_routes)

        cee_summary = routed_df.groupby(["Cluster_ID", "CEE_ID"]).agg({
            "Society_ID": "count",
            "Orders": "sum"
        }).reset_index().rename(columns={"Society_ID": "Societies_Assigned", "Orders": "Total_Orders"})

        st.subheader("Clustered & Routed Delivery Plan")
        st.dataframe(routed_df[[
            "Society_ID", "Society_Name", "Latitude", "Longitude",
            "Orders", "CEE_ID", "Delivery_Sequence", "Cluster_ID"
        ]])

        st.subheader("CEE Group Summary (Post Clustering)")
        st.dataframe(cee_summary)

        # Downloadable output
        csv_output = routed_df.to_csv(index=False)
        st.download_button("Download Delivery Plan CSV", data=csv_output, file_name="cee_delivery_plan.csv", mime="text/csv")
