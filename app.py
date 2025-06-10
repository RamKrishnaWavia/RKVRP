import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from networkx.algorithms.approximation import traveling_salesman_problem, greedy_tsp
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from io import BytesIO
import base64

# ------------------------
# Helper Functions
# ------------------------

def haversine(coord1, coord2):
    return great_circle(coord1, coord2).km

def generate_cluster(df, order_threshold, distance_threshold_km):
    coords = df[["Latitude", "Longitude"]].to_numpy()
    kms_per_radian = 6371.0088
    epsilon = distance_threshold_km / kms_per_radian
    db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine')
    db.fit(np.radians(coords))
    df["Cluster_ID"] = db.labels_

    cluster_summary = df.groupby("Cluster_ID").agg({"Orders": "sum"}).reset_index()
    valid_clusters = cluster_summary[cluster_summary["Orders"] >= order_threshold]["Cluster_ID"]
    df["Cluster_Type"] = df["Cluster_ID"].apply(lambda x: "Green" if x in valid_clusters.values else "Blue")
    return df

def solve_tsp(locations_df):
    locations = locations_df[["Latitude", "Longitude"]].to_numpy()
    G = nx.Graph()
    for i in range(len(locations)):
        for j in range(i + 1, len(locations)):
            dist = haversine(locations[i], locations[j])
            G.add_edge(i, j, weight=dist)
    try:
        tsp_path = traveling_salesman_problem(G, cycle=False, method=greedy_tsp)
        tsp_order = locations_df.iloc[tsp_path].copy()
        tsp_order["Delivery_Sequence"] = ["S" + str(i + 1) for i in range(len(tsp_order))]
        return tsp_order
    except Exception as e:
        st.warning(f"❌ TSP failed for cluster. Error: {str(e)}")
        return None

def create_map(df, cluster_col="Cluster_ID", sequence_col="Delivery_Sequence"):
    m = folium.Map(location=[df["Latitude"].mean(), df["Longitude"].mean()], zoom_start=12)
    colors = ["green", "blue", "orange", "purple", "darkred", "cadetblue", "pink", "black", "darkblue"]
    for cluster_id in df[cluster_col].unique():
        cluster_data = df[df[cluster_col] == cluster_id]
        color = colors[cluster_id % len(colors)]
        cluster = MarkerCluster().add_to(m)
        for _, row in cluster_data.iterrows():
            popup = f"{row['Society_Name']}<br>Orders: {row['Orders']}<br>{row[sequence_col]}"
            folium.Marker(
                location=[row["Latitude"], row["Longitude"]],
                popup=popup,
                icon=folium.Icon(color=color, icon="info-sign")
            ).add_to(cluster)
        # Draw lines
        path = cluster_data.sort_values(by=sequence_col)[["Latitude", "Longitude"]].values.tolist()
        folium.PolyLine(path, color=color, weight=2.5, opacity=0.7).add_to(m)
    return m

def generate_template():
    df = pd.DataFrame({
        "Society_ID": ["S101", "S102"],
        "Society_Name": ["Prestige Lakeview", "Sobha Greenfields"],
        "Latitude": [12.934, 12.938],
        "Longitude": [77.610, 77.620],
        "Orders": [120, 100]
    })
    return df

def get_table_download_link(df, filename="sample.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Download Template</a>'

# ------------------------
# Streamlit App
# ------------------------

st.set_page_config(layout="wide")
st.title("📦 Milk Delivery Society Clustering & Route Optimizer")

with st.sidebar:
    st.header("Depot Settings")
    depot_lat = st.number_input("Depot Latitude", value=12.935, format="%.6f")
    depot_lon = st.number_input("Depot Longitude", value=77.610, format="%.6f")
    st.markdown(get_table_download_link(generate_template(), "milk_input_template.csv"), unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Input File (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if not {"Society_ID", "Society_Name", "Latitude", "Longitude", "Orders"}.issubset(df.columns):
        st.error("❌ Input file must contain columns: Society_ID, Society_Name, Latitude, Longitude, Orders")
    else:
        df["Latitude"] = df["Latitude"].astype(float)
        df["Longitude"] = df["Longitude"].astype(float)
        df["Orders"] = df["Orders"].astype(int)

        st.success("✅ Data Loaded")
        df = generate_cluster(df, order_threshold=200, distance_threshold_km=2.0)

        final_routes = []
        for cluster_id in df["Cluster_ID"].unique():
            cluster_df = df[df["Cluster_ID"] == cluster_id].copy()
            cluster_df.reset_index(drop=True, inplace=True)
            tsp_result = solve_tsp(cluster_df)
            if tsp_result is not None:
                tsp_result["Cluster_ID"] = cluster_id
                final_routes.append(tsp_result)

        if final_routes:
            full_routes_df = pd.concat(final_routes)
            st.dataframe(full_routes_df[["Society_ID", "Society_Name", "Orders", "Cluster_ID", "Delivery_Sequence"]])

            # Show map
            st.subheader("📍 Route Map with Delivery Sequence")
            route_map = create_map(full_routes_df)
            st_folium(route_map, width=1000, height=600)

            # Export route
            st.download_button("📥 Download Delivery Plan CSV", data=full_routes_df.to_csv(index=False), file_name="delivery_plan.csv", mime="text/csv")
        else:
            st.warning("⚠ No valid clusters found.")
else:
    st.info("📤 Please upload a CSV file to begin.")
