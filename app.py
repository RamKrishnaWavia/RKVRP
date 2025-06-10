import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import networkx as nx
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import io

# Sidebar settings
st.sidebar.title("🚚 Delivery Cluster Settings")
depot_lat = st.sidebar.number_input("Depot Latitude", value=12.9166, format="%.6f")
depot_lon = st.sidebar.number_input("Depot Longitude", value=77.6101, format="%.6f")

# Download input template
template_df = pd.DataFrame({
    "Society_ID": ["S001", "S002"],
    "Society_Name": ["Prestige Lake View", "Sobha Silicon Oasis"],
    "Latitude": [12.9166, 12.9155],
    "Longitude": [77.6101, 77.6189],
    "Orders": [45, 60]
})
template_buffer = io.StringIO()
template_df.to_csv(template_buffer, index=False)
template_csv = template_buffer.getvalue()

st.sidebar.markdown("### 📥 Download Input Template")
st.sidebar.download_button(
    label="Download CSV Template",
    data=template_csv,
    file_name="milk_delivery_input_template.csv",
    mime="text/csv"
)

st.title("Milk Delivery Route Optimizer 🚚🗺️")

uploaded_file = st.file_uploader("Upload your Society Orders CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['Latitude'] = df['Latitude'].astype(float)
    df['Longitude'] = df['Longitude'].astype(float)

    # Clustering based on proximity and order threshold
    kms_per_radian = 6371.0088
    epsilon = 2 / kms_per_radian
    coords = df[['Latitude', 'Longitude']].to_numpy()
    db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
    df["Cluster_ID"] = db.labels_

    cluster_summary = df.groupby("Cluster_ID").agg({"Orders": "sum"}).reset_index()
    valid_clusters = cluster_summary[cluster_summary["Orders"] >= 200]["Cluster_ID"]
    df = df[df["Cluster_ID"].isin(valid_clusters)]

    st.subheader("Clustered Societies")
    st.dataframe(df)

    # Routing within clusters
    routes = []
    for cid in df["Cluster_ID"].unique():
        cluster_df = df[df["Cluster_ID"] == cid].copy()
        locs = [(depot_lat, depot_lon)] + list(zip(cluster_df["Latitude"], cluster_df["Longitude"]))
        G = nx.complete_graph(len(locs))
        pos = {i: loc for i, loc in enumerate(locs)}
        for i in G.nodes:
            for j in G.nodes:
                if i != j:
                    G[i][j]["weight"] = np.linalg.norm(np.array(locs[i]) - np.array(locs[j]))
        tsp_path = nx.approximation.traveling_salesman_problem(G, cycle=False, method="greedy")

        tsp_points = [pos[i] for i in tsp_path]
        path_df = pd.DataFrame(tsp_points[1:], columns=["Latitude", "Longitude"])
        cluster_df = cluster_df.merge(path_df, on=["Latitude", "Longitude"])
        cluster_df["Sequence"] = ["S" + str(i+1) for i in range(len(cluster_df))]
        cluster_df["Cluster_ID"] = cid
        routes.append(cluster_df)

    routed_df = pd.concat(routes)
    st.subheader("Optimized Delivery Routes")
    st.dataframe(routed_df[["Society_ID", "Society_Name", "Cluster_ID", "Sequence", "Orders", "Latitude", "Longitude"]])

    # Map visualization
    m = folium.Map(location=[depot_lat, depot_lon], zoom_start=13)
    cluster_group = MarkerCluster().add_to(m)

    for cid in routed_df["Cluster_ID"].unique():
        cluster_data = routed_df[routed_df["Cluster_ID"] == cid]
        cluster_color = f"#{np.random.randint(0, 0xFFFFFF):06x}"

        # Plot delivery points
        for _, row in cluster_data.iterrows():
            folium.CircleMarker(
                location=[row["Latitude"], row["Longitude"]],
                radius=5,
                popup=f"{row['Society_Name']}<br>Orders: {row['Orders']}",
                color=cluster_color,
                fill=True,
                fill_opacity=0.7
            ).add_to(cluster_group)

        # Draw lines for sequence
        path_coords = cluster_data.sort_values("Sequence")[["Latitude", "Longitude"]].values.tolist()
        path_coords.insert(0, [depot_lat, depot_lon])
        folium.PolyLine(path_coords, color=cluster_color, weight=2.5, opacity=0.8).add_to(m)

    st.subheader("🗺️ Route Map by Cluster")
    st_folium(m, width=1000, height=600)
else:
    st.info("Please upload a valid input file to proceed.")
