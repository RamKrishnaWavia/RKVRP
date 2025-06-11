import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import networkx as nx
from geopy.distance import geodesic
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from io import StringIO

st.set_page_config(page_title="Milk Delivery Route Optimizer", layout="wide")
st.title("🥛 Milk Delivery Route Optimizer")

# --- Constants ---
MIN_ORDERS = 200
MAX_ORDERS = 225
DISTANCE_THRESHOLD_KM = 2

# --- Helper Functions ---

def haversine_dist(a, b):
    return geodesic(a, b).km

def create_distance_matrix(coords):
    n = len(coords)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_matrix[i][j] = haversine_dist(coords[i], coords[j])
    return dist_matrix

def optimize_route(coords):
    G = nx.complete_graph(len(coords))
    for i in range(len(coords)):
        for j in range(len(coords)):
            if i != j:
                dist = haversine_dist(coords[i], coords[j])
                G[i][j]['weight'] = dist
    try:
        path = nx.approximation.traveling_salesman_problem(G, cycle=False, method='greedy')
        return path
    except Exception as e:
        st.warning(f"TSP optimization failed: {e}")
        return list(range(len(coords)))

def cluster_societies(df):
    coords = df[["Latitude", "Longitude"]].values
    db = DBSCAN(eps=DISTANCE_THRESHOLD_KM/6371, min_samples=1, metric='haversine').fit(np.radians(coords))
    df["Temp_Cluster"] = db.labels_

    clustered_df = pd.DataFrame()
    cluster_id = 0
    for temp_cluster in df["Temp_Cluster"].unique():
        cluster_group = df[df["Temp_Cluster"] == temp_cluster]
        current_batch = []
        batch_orders = 0
        for _, row in cluster_group.iterrows():
            current_batch.append(row)
            batch_orders += row["Orders"]
            if batch_orders >= MIN_ORDERS:
                if batch_orders <= MAX_ORDERS:
                    for r in current_batch:
                        clustered_df = pd.concat([clustered_df, pd.DataFrame([{
                            **r,
                            "Cluster_ID": cluster_id,
                            "Cluster_Type": "Green"
                        }])])
                    cluster_id += 1
                current_batch = []
                batch_orders = 0
        # Remaining societies
        for r in current_batch:
            clustered_df = pd.concat([clustered_df, pd.DataFrame([{
                **r,
                "Cluster_ID": cluster_id,
                "Cluster_Type": "Blue"
            }])])
        cluster_id += 1
    clustered_df.reset_index(drop=True, inplace=True)
    return clustered_df

def add_delivery_sequence(df, depot_coords):
    final_df = pd.DataFrame()
    for cluster_id in df["Cluster_ID"].unique():
        cluster_df = df[df["Cluster_ID"] == cluster_id].copy()
        coords = [depot_coords] + cluster_df[["Latitude", "Longitude"]].values.tolist()
        route_order = optimize_route(coords)

        delivery_points = []
        total_dist = 0
        for i, idx in enumerate(route_order[1:]):  # Skip depot
            row = cluster_df.iloc[idx - 1]
            dist_from_prev = (
                haversine_dist(coords[route_order[i]], coords[route_order[i+1]])
                if i+1 < len(route_order) else 0
            )
            total_dist += dist_from_prev
            row["Delivery_Sequence"] = f"S{i+1}"
            row["Distance_from_prev_km"] = round(dist_from_prev, 2)
            delivery_points.append(row)
        cluster_result = pd.DataFrame(delivery_points)
        final_df = pd.concat([final_df, cluster_result])
    final_df.reset_index(drop=True, inplace=True)
    return final_df

def plot_map(df, depot_coords):
    m = folium.Map(location=depot_coords, zoom_start=12)
    folium.Marker(depot_coords, tooltip="Depot", icon=folium.Icon(color="black")).add_to(m)

    for cluster_id in df["Cluster_ID"].unique():
        cluster_df = df[df["Cluster_ID"] == cluster_id]
        color = "green" if cluster_df["Cluster_Type"].iloc[0] == "Green" else "blue"
        points = [depot_coords] + cluster_df.sort_values("Delivery_Sequence")[["Latitude", "Longitude"]].values.tolist()

        for _, row in cluster_df.iterrows():
            tooltip = f"{row['Society_Name']}<br>Orders: {row['Orders']}<br>Distance: {row['Distance_from_prev_km']} km"
            folium.Marker(
                [row["Latitude"], row["Longitude"]],
                tooltip=tooltip,
                icon=folium.Icon(color=color)
            ).add_to(m)

        folium.PolyLine(points, color=color, weight=2.5, opacity=0.8).add_to(m)
    return m

def generate_template():
    data = {
        "Society_ID": [],
        "Society_Name": [],
        "Latitude": [],
        "Longitude": [],
        "Orders": []
    }
    df = pd.DataFrame(data)
    csv = df.to_csv(index=False)
    return csv

# --- UI ---

st.sidebar.header("Upload Input CSV")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

st.sidebar.markdown("### Depot Coordinates")
default_lat = st.sidebar.number_input("Depot Latitude", value=12.935, format="%.6f")
default_lon = st.sidebar.number_input("Depot Longitude", value=77.614, format="%.6f")
depot_coords = [default_lat, default_lon]

st.sidebar.markdown("### Download Template")
template_csv = generate_template()
st.sidebar.download_button("📥 Download Template", template_csv, file_name="milk_delivery_template.csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    required_cols = {"Society_ID", "Society_Name", "Latitude", "Longitude", "Orders"}

    if not required_cols.issubset(df.columns):
        st.error("Uploaded file missing required columns.")
    else:
        clustered_df = cluster_societies(df)
        routed_df = add_delivery_sequence(clustered_df, depot_coords)
        map_obj = plot_map(routed_df, depot_coords)

        st.subheader("🗺️ Clustered Delivery Route Map")
        st_data = st_folium(map_obj, width=1000)

        st.subheader("📄 Route Output Table")
        st.dataframe(routed_df[["Society_ID", "Society_Name", "Orders", "Cluster_ID", "Cluster_Type", "Delivery_Sequence", "Distance_from_prev_km"]])

        csv_out = routed_df.to_csv(index=False)
        st.download_button("📤 Download Final Output CSV", csv_out, file_name="clustered_delivery_output.csv")
else:
    st.info("Please upload a CSV file with Society info to proceed.")
