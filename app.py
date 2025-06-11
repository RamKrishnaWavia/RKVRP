import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from geopy.distance import geodesic
from sklearn.cluster import DBSCAN
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import io

# -------------------- Sidebar for Depot Coordinates --------------------
st.sidebar.header("Depot Location")
depot_lat = st.sidebar.number_input("Depot Latitude", value=12.9352, format="%.6f")
depot_lon = st.sidebar.number_input("Depot Longitude", value=77.6142, format="%.6f")

# -------------------- Template Download Button --------------------
@st.cache_data
def get_template_csv():
    df_template = pd.DataFrame({
        "Society_ID": [],
        "Society_Name": [],
        "Latitude": [],
        "Longitude": [],
        "Orders": []
    })
    return df_template.to_csv(index=False).encode('utf-8')

st.sidebar.download_button("📥 Download Input Template", get_template_csv(), "milk_delivery_template.csv", "text/csv")

# -------------------- File Upload --------------------
st.title("🛵 Milk Delivery Route Optimizer")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if not uploaded_file:
    st.stop()

df = pd.read_csv(uploaded_file)
required_cols = ["Society_ID", "Society_Name", "Latitude", "Longitude", "Orders"]
if not all(col in df.columns for col in required_cols):
    st.error("Uploaded file must contain columns: " + ", ".join(required_cols))
    st.stop()

# -------------------- Clustering --------------------
st.subheader("📦 Clustering Societies")
coords = df[["Latitude", "Longitude"]].values
kms_per_radian = 6371.0088
epsilon = 2 / kms_per_radian

db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
df["Cluster_ID"] = db.labels_

# -------------------- Filter clusters by total orders between 200 and 225 --------------------
cluster_summary = df.groupby("Cluster_ID")["Orders"].sum().reset_index()
valid_clusters = cluster_summary[(cluster_summary["Orders"] >= 200) & (cluster_summary["Orders"] <= 225)]["Cluster_ID"]
df = df[df["Cluster_ID"].isin(valid_clusters)].copy()
df["Cluster_Type"] = "Green"

# -------------------- TSP Route & Distance --------------------
def compute_distance_km(latlon1, latlon2):
    return round(geodesic(latlon1, latlon2).km, 2)

def solve_tsp(cluster_df):
    locations = [(depot_lat, depot_lon)] + list(cluster_df[["Latitude", "Longitude"]].itertuples(index=False, name=None))
    G = nx.complete_graph(len(locations))

    for i in range(len(locations)):
        for j in range(i + 1, len(locations)):
            dist = compute_distance_km(locations[i], locations[j])
            G[i][j]['weight'] = dist
            G[j][i]['weight'] = dist

    try:
        tsp_order = nx.approximation.traveling_salesman_problem(G, cycle=False, method='greedy')
        tsp_path = [p for p in tsp_order if p != 0]  # remove depot from sequence
        sequence = []
        total_dist = 0
        for i, idx in enumerate(tsp_path):
            row = cluster_df.iloc[idx - 1]  # -1 since depot was added at beginning
            if i == 0:
                dist = compute_distance_km((depot_lat, depot_lon), (row["Latitude"], row["Longitude"]))
            else:
                prev_row = cluster_df.iloc[tsp_path[i - 1] - 1]
                dist = compute_distance_km((prev_row["Latitude"], prev_row["Longitude"]), (row["Latitude"], row["Longitude"]))
            total_dist += dist
            sequence.append({
                "Society_ID": row["Society_ID"],
                "Society_Name": row["Society_Name"],
                "Latitude": row["Latitude"],
                "Longitude": row["Longitude"],
                "Orders": row["Orders"],
                "Delivery_Sequence": f"S{i+1}",
                "Distance_From_Prev_km": dist,
                "Cluster_ID": row["Cluster_ID"],
                "Cluster_Type": row["Cluster_Type"]
            })
        return pd.DataFrame(sequence)
    except Exception as e:
        st.warning(f"❌ TSP failed for cluster. Error: {e}")
        return pd.DataFrame()

# -------------------- Run TSP for each cluster --------------------
final_df = pd.DataFrame()
for cluster_id in df["Cluster_ID"].unique():
    cluster_data = df[df["Cluster_ID"] == cluster_id].reset_index(drop=True)
    routed_cluster = solve_tsp(cluster_data)
    final_df = pd.concat([final_df, routed_cluster], ignore_index=True)

# -------------------- Map Display --------------------
st.subheader("🗺️ Delivery Map")

map_center = [df["Latitude"].mean(), df["Longitude"].mean()]
m = folium.Map(location=map_center, zoom_start=13)

colors = ["red", "blue", "green", "purple", "orange", "darkred", "cadetblue", "darkgreen"]
for idx, cluster_id in enumerate(final_df["Cluster_ID"].unique()):
    cluster_df = final_df[final_df["Cluster_ID"] == cluster_id]
    color = colors[idx % len(colors)]

    points = [(depot_lat, depot_lon)] + list(cluster_df[["Latitude", "Longitude"]].itertuples(index=False, name=None))
    for i, row in cluster_df.iterrows():
        tooltip = f"{row['Society_Name']}<br>Orders: {row['Orders']}<br>Distance from prev: {row['Distance_From_Prev_km']} km"
        folium.Marker(
            location=[row["Latitude"], row["Longitude"]],
            popup=tooltip,
            tooltip=tooltip,
            icon=folium.Icon(color=color)
        ).add_to(m)

    folium.PolyLine(locations=points, color=color, weight=3, opacity=0.8).add_to(m)

# Show depot
folium.Marker([depot_lat, depot_lon], tooltip="Depot", icon=folium.Icon(color="black", icon="home")).add_to(m)

st_folium(m, width=1000, height=600)

# -------------------- Output CSV --------------------
st.subheader("📄 Final Output")
st.dataframe(final_df)
csv = final_df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Final Route CSV", csv, "final_routed_societies.csv", "text/csv")
