import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import haversine_distances
from math import radians

st.set_page_config(layout="wide")
st.title("Milk Delivery Cluster and Route Optimizer")

# Sidebar for Depot Lat/Long
st.sidebar.header("Depot Location")
depot_lat = st.sidebar.number_input("Depot Latitude", value=12.9352)
depot_lon = st.sidebar.number_input("Depot Longitude", value=77.6145)

# File upload
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    required_cols = {"Society_ID", "Orders", "Latitude", "Longitude"}
    if not required_cols.issubset(df.columns):
        st.error("Input file must have columns: Society_ID, Orders, Latitude, Longitude")
        st.stop()

    df['Latitude'] = df['Latitude'].astype(float)
    df['Longitude'] = df['Longitude'].astype(float)

    # Compute distance matrix in km
    coords = df[['Latitude', 'Longitude']].applymap(radians).to_numpy()
    dist_matrix = haversine_distances(coords) * 6371  # Earth radius in km

    # Clustering logic: societies within 2km radius and total orders >= 200
    clusters = []
    assigned = set()
    for i in range(len(df)):
        if i in assigned:
            continue
        cluster = [i]
        order_sum = df.iloc[i]['Orders']
        for j in range(i + 1, len(df)):
            if j not in assigned and dist_matrix[i][j] <= 2:
                cluster.append(j)
                order_sum += df.iloc[j]['Orders']
        if order_sum >= 200:
            clusters.append(cluster)
            assigned.update(cluster)

    # Assign cluster IDs
    cluster_map = {}
    for cluster_id, indices in enumerate(clusters, start=1):
        for idx in indices:
            cluster_map[idx] = cluster_id

    df["Cluster_ID"] = df.index.map(lambda x: cluster_map.get(x, -1))

    valid_clusters = df[df["Cluster_ID"] != -1]["Cluster_ID"].unique()
    st.success(f"Generated {len(valid_clusters)} clusters")

    all_routes = []
    for cluster_id in valid_clusters:
        cluster_df = df[df["Cluster_ID"] == cluster_id].copy().reset_index(drop=True)

        # Add depot as node 0
        nodes = [{"Society_ID": "Depot", "Latitude": depot_lat, "Longitude": depot_lon, "Orders": 0}]
        nodes += cluster_df.to_dict("records")
        route_df = pd.DataFrame(nodes)

        # Create graph
        G = nx.Graph()
        for i, row1 in route_df.iterrows():
            for j, row2 in route_df.iterrows():
                if i != j:
                    lat1, lon1 = radians(row1["Latitude"]), radians(row1["Longitude"])
                    lat2, lon2 = radians(row2["Latitude"]), radians(row2["Longitude"])
                    dist = haversine_distances([[lat1, lon1], [lat2, lon2]])[0][1] * 6371
                    G.add_edge(i, j, weight=dist)

        tsp_path = nx.approximation.traveling_salesman_problem(G, cycle=False, method="greedy")

        route_sequence = []
        for seq_num, idx in enumerate(tsp_path[1:], start=1):  # Skip depot
            row = cluster_df.iloc[idx - 1]  # depot is at 0
            route_sequence.append({
                "Cluster_ID": cluster_id,
                "Delivery_Seq": f"S{seq_num}",
                "Society_ID": row["Society_ID"],
                "Orders": row["Orders"],
                "Latitude": row["Latitude"],
                "Longitude": row["Longitude"]
            })

        route_df = pd.DataFrame(route_sequence)
        all_routes.append(route_df)

        st.subheader(f"Cluster {cluster_id} Route")
        st.dataframe(route_df)
        st.map(route_df[["Latitude", "Longitude"]])

    final_output = pd.concat(all_routes, ignore_index=True)
    st.download_button("Download Route Plan CSV", final_output.to_csv(index=False), file_name="delivery_routes.csv")
