import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from sklearn.cluster import DBSCAN
from geopy.distance import geodesic
import networkx as nx
import math
import io

# --- Sidebar Settings ---
st.sidebar.header("Depot Coordinates")
depot_lat = st.sidebar.number_input("Depot Latitude", value=12.935, format="%.6f")
depot_lon = st.sidebar.number_input("Depot Longitude", value=77.609, format="%.6f")
min_orders = 200
max_orders = 225
max_distance_km = 2

# --- CSV Template Download ---
def create_template():
    data = {
        "Society_ID": [],
        "Society_Name": [],
        "Latitude": [],
        "Longitude": [],
        "Orders": []
    }
    df = pd.DataFrame(data)
    return df

st.sidebar.download_button(
    label="📥 Download CSV Template",
    data=create_template().to_csv(index=False).encode("utf-8"),
    file_name="society_template.csv",
    mime="text/csv"
)

st.title("📦 Society Clustering & Route Planner")

uploaded_file = st.file_uploader("Upload Society CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # --- Clustering ---
    coords = df[["Latitude", "Longitude"]].to_numpy()
    kms_per_radian = 6371.0088
    epsilon = max_distance_km / kms_per_radian
    db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine')
    db.fit(np.radians(coords))
    df["Cluster_ID"] = db.labels_

    # --- Cluster Summaries ---
    cluster_summary = df.groupby("Cluster_ID")["Orders"].sum().reset_index()
    valid_clusters = cluster_summary[
        (cluster_summary["Orders"] >= min_orders) & 
        (cluster_summary["Orders"] <= max_orders)
    ]["Cluster_ID"]

    df["Cluster_Type"] = df["Cluster_ID"].apply(
        lambda x: "Green" if x in valid_clusters.values else "Blue"
    )

    # --- Route Planning ---
    final_routes = []
    for cluster_id in valid_clusters:
        cluster_df = df[df["Cluster_ID"] == cluster_id].copy()
        nodes = [(row.Latitude, row.Longitude, row.Society_Name, row.Orders, row.Society_ID)
                 for _, row in cluster_df.iterrows()]
        nodes.insert(0, (depot_lat, depot_lon, "Depot", 0, "Depot"))

        # Build graph
        G = nx.complete_graph(len(nodes))
        for i in G.nodes:
            G.nodes[i]["pos"] = (nodes[i][0], nodes[i][1])

        for i in G.nodes:
            for j in G.nodes:
                if i != j:
                    coord1 = (nodes[i][0], nodes[i][1])
                    coord2 = (nodes[j][0], nodes[j][1])
                    dist = geodesic(coord1, coord2).km
                    G[i][j]["weight"] = dist

        try:
            path = nx.approximation.traveling_salesman_problem(G, cycle=False, method="greedy")
        except Exception as e:
            st.error(f"❌ TSP failed for cluster {cluster_id}. Error: {e}")
            continue

        # Build route output
        seq = 1
        for i in range(1, len(path)):
            prev_idx = path[i - 1]
            curr_idx = path[i]
            prev = nodes[prev_idx]
            curr = nodes[curr_idx]
            dist = round(geodesic((prev[0], prev[1]), (curr[0], curr[1])).km, 2)
            final_routes.append({
                "Cluster_ID": cluster_id,
                "Cluster_Type": "Green",
                "Delivery_Sequence": f"S{seq}",
                "Society_ID": curr[4],
                "Society_Name": curr[2],
                "Latitude": curr[0],
                "Longitude": curr[1],
                "Orders": curr[3],
                "Distance_from_Previous": dist
            })
            seq += 1

    route_df = pd.DataFrame(final_routes)

    st.subheader("📋 Final Routed Deliveries")
    st.dataframe(route_df)

    # --- Map Display ---
    st.subheader("🗺️ Delivery Map")
    m = folium.Map(location=[depot_lat, depot_lon], zoom_start=12)

    for cluster_id in route_df["Cluster_ID"].unique():
        cluster_data = route_df[route_df["Cluster_ID"] == cluster_id]
        color = "green" if cluster_data["Cluster_Type"].iloc[0] == "Green" else "blue"
        points = [(row["Latitude"], row["Longitude"]) for _, row in cluster_data.iterrows()]
        folium.PolyLine(points, color=color, weight=3).add_to(m)

        for _, row in cluster_data.iterrows():
            popup = (
                f"<b>{row['Society_Name']}</b><br/>"
                f"Orders: {row['Orders']}<br/>"
                f"Distance: {row['Distance_from_Previous']} km"
            )
            folium.Marker(
                location=[row["Latitude"], row["Longitude"]],
                popup=popup,
                icon=folium.Icon(color=color)
            ).add_to(m)

    folium.Marker(
        location=[depot_lat, depot_lon],
        popup="Depot",
        icon=folium.Icon(color="red", icon="home")
    ).add_to(m)

    st_data = st_folium(m, width=1000)

    # --- Export Output ---
    st.download_button(
        label="📤 Download Final Route CSV",
        data=route_df.to_csv(index=False).encode("utf-8"),
        file_name="clustered_routes.csv",
        mime="text/csv"
    )

else:
    st.info("Please upload a society order CSV file.")
