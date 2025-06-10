import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from geopy.distance import geodesic
import folium
from streamlit_folium import st_folium
import networkx as nx
from networkx.algorithms.approximation import traveling_salesman_problem
from io import BytesIO

st.set_page_config(layout="wide")

st.title("🛒 Milk Delivery Clustering & Routing Tool")

# Sidebar inputs
with st.sidebar:
    st.header("Depot Settings")
    depot_lat = st.number_input("Depot Latitude", value=12.9716, format="%.6f")
    depot_lon = st.number_input("Depot Longitude", value=77.5946, format="%.6f")

    st.markdown("---")
    st.download_button(
        label="📥 Download Template",
        data="Society_ID,Society_Name,Latitude,Longitude,Orders\n",
        file_name="milk_delivery_template.csv",
        mime="text/csv"
    )

# Upload CSV
uploaded_file = st.file_uploader("Upload Society Orders CSV", type=["csv"])
if not uploaded_file:
    st.stop()

df = pd.read_csv(uploaded_file)
required_cols = {"Society_ID", "Society_Name", "Latitude", "Longitude", "Orders"}
if not required_cols.issubset(set(df.columns)):
    st.error("Input file must contain columns: Society_ID, Society_Name, Latitude, Longitude, Orders")
    st.stop()

# Convert to float
df["Latitude"] = df["Latitude"].astype(float)
df["Longitude"] = df["Longitude"].astype(float)
df["Orders"] = df["Orders"].astype(int)

# Step 1: Cluster based on 2km radius
coords = df[["Latitude", "Longitude"]].values
kms_per_radian = 6371.0088
epsilon = 2 / kms_per_radian  # 2km radius
db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine')
db.fit(np.radians(coords))
df["Cluster_ID"] = db.labels_

# Step 2: Filter clusters with >= 200 orders
cluster_summary = df.groupby("Cluster_ID")["Orders"].sum().reset_index()
valid_clusters = cluster_summary[cluster_summary["Orders"] >= 200]["Cluster_ID"]
df["Valid_Cluster"] = df["Cluster_ID"].apply(lambda x: x if x in valid_clusters.values else -1)

# Step 3: Assign delivery sequence within valid clusters
delivery_data = []
for cluster_id in df["Valid_Cluster"].unique():
    if cluster_id == -1:
        continue
    cluster_df = df[df["Valid_Cluster"] == cluster_id].copy()
    G = nx.complete_graph(len(cluster_df))
    locs = list(zip(cluster_df["Latitude"], cluster_df["Longitude"]))

    # Add depot to locs and index
    locs_with_depot = [(depot_lat, depot_lon)] + locs
    labels = ["Depot"] + list(cluster_df["Society_ID"].values)

    H = nx.complete_graph(len(locs_with_depot))
    for i in range(len(locs_with_depot)):
        for j in range(len(locs_with_depot)):
            if i != j:
                dist = geodesic(locs_with_depot[i], locs_with_depot[j]).km
                H[i][j]["weight"] = dist

    tsp_order = traveling_salesman_problem(H, cycle=False, method="greedy")
    tsp_labels = [labels[i] for i in tsp_order if i != 0]

    # Assign sequence
    for i, sid in enumerate(tsp_labels):
        row = cluster_df[cluster_df["Society_ID"] == sid].iloc[0]
        delivery_data.append({
            "Cluster_ID": cluster_id,
            "Sequence": f"S{i+1}",
            "Society_ID": row["Society_ID"],
            "Society_Name": row["Society_Name"],
            "Latitude": row["Latitude"],
            "Longitude": row["Longitude"],
            "Orders": row["Orders"]
        })

routed_df = pd.DataFrame(delivery_data)
st.subheader("📦 Clustered & Sequenced Delivery Plan")
st.dataframe(routed_df)

# Step 4: Map
st.subheader("🗺️ Delivery Route Map")

m = folium.Map(location=[df["Latitude"].mean(), df["Longitude"].mean()], zoom_start=12)

for cluster_id in routed_df["Cluster_ID"].unique():
    cluster_df = routed_df[routed_df["Cluster_ID"] == cluster_id]
    color = f"#{np.random.randint(0, 0xFFFFFF):06x}"

    points = []
    for _, row in cluster_df.iterrows():
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=6,
            popup=f"{row['Society_Name']} (Orders: {row['Orders']})",
            color=color,
            fill=True,
            fill_opacity=0.8
        ).add_to(m)
        points.append((row["Latitude"], row["Longitude"]))

    folium.PolyLine(points, color=color, weight=3, opacity=0.7).add_to(m)

# Add depot marker
folium.Marker(
    location=[depot_lat, depot_lon],
    popup="Depot",
    icon=folium.Icon(color='red', icon='home')
).add_to(m)

st_folium(m, width=1200, height=600)

# Download output CSV
st.download_button(
    label="📤 Download Delivery Plan",
    data=routed_df.to_csv(index=False),
    file_name="clustered_routes.csv",
    mime="text/csv"
)
