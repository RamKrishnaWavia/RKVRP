import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from geopy.distance import geodesic
import folium
from streamlit_folium import st_folium
from networkx.algorithms.approximation import traveling_salesman_problem
from io import BytesIO

st.set_page_config(layout="wide")

st.title("📦 Milk Delivery Route Optimizer")

# Sidebar depot settings
with st.sidebar:
    st.header("Depot Settings")
    depot_lat = st.number_input("Depot Latitude", value=12.9716)
    depot_lon = st.number_input("Depot Longitude", value=77.5946)
    st.markdown("---")
    st.download_button(
        label="📥 Download Input Template",
        data=pd.DataFrame({
            "Society_ID": ["S1", "S2"],
            "Society_Name": ["Society A", "Society B"],
            "Orders": [120, 100],
            "Latitude": [12.95, 12.96],
            "Longitude": [77.59, 77.61]
        }).to_csv(index=False),
        file_name="milk_delivery_template.csv",
        mime="text/csv"
    )

uploaded_file = st.file_uploader("Upload Society Orders CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    required_cols = {"Society_ID", "Society_Name", "Orders", "Latitude", "Longitude"}
    if not required_cols.issubset(df.columns):
        st.error(f"Missing columns. Required: {required_cols}")
        st.stop()

    df["Latitude"] = df["Latitude"].astype(float)
    df["Longitude"] = df["Longitude"].astype(float)

    # Compute pairwise distances
    coords = df[["Latitude", "Longitude"]].to_numpy()
    dist_matrix = np.zeros((len(coords), len(coords)))
    for i in range(len(coords)):
        for j in range(len(coords)):
            dist_matrix[i, j] = geodesic(coords[i], coords[j]).km

    # Clustering based on proximity < 2 km and total orders >= 200
    visited = set()
    cluster_id = 1
    clusters = []

    for i in range(len(df)):
        if i in visited:
            continue
        group = [i]
        visited.add(i)
        for j in range(i + 1, len(df)):
            if j not in visited and dist_matrix[i, j] < 2:
                group.append(j)
                visited.add(j)
        total_orders = df.iloc[group]["Orders"].sum()
        if total_orders >= 200:
            clusters.append((cluster_id, group))
            cluster_id += 1
        else:
            # Assign leftover clusters separate ID > 9000
            clusters.append((9000 + i, [i]))

    cluster_rows = []
    for cid, members in clusters:
        for idx in members:
            row = df.iloc[idx].to_dict()
            row["Cluster_ID"] = cid
            cluster_rows.append(row)

    df_clustered = pd.DataFrame(cluster_rows)

    # TSP route generation per cluster
    all_routes = []

    for cid in sorted(df_clustered["Cluster_ID"].unique()):
        cluster_df = df_clustered[df_clustered["Cluster_ID"] == cid].copy()
        locations = [(depot_lat, depot_lon)] + list(zip(cluster_df["Latitude"], cluster_df["Longitude"]))

        if len(locations) < 3:
            cluster_df["Sequence"] = [1]
            cluster_df["Sequence_Label"] = ["S1"]
            cluster_df["TSP_Index"] = [1]
            all_routes.append(cluster_df)
            continue

        G = nx.complete_graph(len(locations))
        for i in range(len(locations)):
            for j in range(len(locations)):
                if i != j:
                    G[i][j]["weight"] = geodesic(locations[i], locations[j]).km

        try:
            tsp_path = traveling_salesman_problem(G, cycle=False, method="greedy")
        except Exception as e:
            st.warning(f"❌ TSP failed for cluster {cid}. Error: {e}")
            continue

        delivery_order = tsp_path[1:] if tsp_path[0] == 0 else tsp_path[:-1]
        cluster_df["TSP_Index"] = delivery_order
        cluster_df = cluster_df.sort_values("TSP_Index").copy()
        cluster_df["Sequence"] = range(1, len(cluster_df) + 1)
        cluster_df["Sequence_Label"] = ["S" + str(i) for i in cluster_df["Sequence"]]
        all_routes.append(cluster_df)

    routed_df = pd.concat(all_routes, ignore_index=True)

    st.subheader("📋 Clustered & Sequenced Deliveries")
    st.dataframe(routed_df[["Society_ID", "Society_Name", "Orders", "Cluster_ID", "Sequence_Label"]])

    st.download_button(
        label="📤 Download Route Plan",
        data=routed_df.to_csv(index=False),
        file_name="milk_delivery_routes.csv",
        mime="text/csv"
    )

    # Visualize on map
    m = folium.Map(location=[depot_lat, depot_lon], zoom_start=12)

    for cid, group in routed_df.groupby("Cluster_ID"):
        color = f"#{np.random.randint(0, 0xFFFFFF):06x}"

        sorted_group = group.sort_values("Sequence")
        path = [(lat, lon) for lat, lon in zip(sorted_group["Latitude"], sorted_group["Longitude"])]
        folium.PolyLine(path, color=color, weight=4, opacity=0.7).add_to(m)

        for _, row in sorted_group.iterrows():
            popup = f"{row['Society_Name']}<br>Orders: {row['Orders']}<br>Seq: {row['Sequence_Label']}"
            folium.CircleMarker(
                location=(row["Latitude"], row["Longitude"]),
                radius=6,
                color=color,
                fill=True,
                fill_opacity=0.9,
                popup=popup
            ).add_to(m)

    st.subheader("🗺️ Delivery Map by Cluster & Sequence")
    st_data = st_folium(m, width=1000, height=600)
