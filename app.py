import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from geopy.distance import geodesic
import networkx as nx
from networkx.algorithms.approximation import traveling_salesman_problem
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from io import StringIO
import base64

st.set_page_config(layout="wide")

# ----------------- Sidebar -----------------
st.sidebar.header("Depot Location Settings")
default_lat = 12.935
default_lon = 77.614
depot_lat = st.sidebar.number_input("Depot Latitude", value=default_lat, format="%.6f")
depot_lon = st.sidebar.number_input("Depot Longitude", value=default_lon, format="%.6f")

# ----------------- CSV Template -----------------
@st.cache_data
def get_template():
    df_template = pd.DataFrame({
        "Society_ID": ["S001", "S002"],
        "Society_Name": ["Sobha Dream Acres", "Prestige Lakeview"],
        "Latitude": [12.935, 12.938],
        "Longitude": [77.614, 77.622],
        "Orders": [120, 90]
    })
    return df_template

def generate_download_link(df, filename="template.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Download Template CSV</a>'
    return href

st.sidebar.markdown("### Download Input Template")
st.sidebar.markdown(generate_download_link(get_template()), unsafe_allow_html=True)

# ----------------- Upload Input -----------------
st.title("🚚 Milk Delivery Route Clustering App")
uploaded_file = st.file_uploader("Upload your society data CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df["Latitude"] = df["Latitude"].astype(float)
    df["Longitude"] = df["Longitude"].astype(float)
    df["Orders"] = df["Orders"].astype(int)

    depot_point = (depot_lat, depot_lon)
    coords = df[["Latitude", "Longitude"]].to_numpy()

    # ----------------- Clustering -----------------
    def haversine_eps_km_to_radians(km):
        earth_radius_km = 6371.0088
        return km / earth_radius_km

    db = DBSCAN(eps=haversine_eps_km_to_radians(2), min_samples=1, algorithm='ball_tree', metric='haversine')
    radians_coords = np.radians(coords)
    df["Cluster_ID"] = db.fit_predict(radians_coords)

    # Filter clusters with total orders >= 200
    cluster_summary = df.groupby("Cluster_ID")["Orders"].sum().reset_index()
    valid_clusters = cluster_summary[cluster_summary["Orders"] >= 200]["Cluster_ID"]
    df = df[df["Cluster_ID"].isin(valid_clusters)].reset_index(drop=True)

    # ----------------- Route Sequencing -----------------
    all_routes = []
    for cluster_id in sorted(df["Cluster_ID"].unique()):
        cluster_df = df[df["Cluster_ID"] == cluster_id].copy()
        locations = [(depot_lat, depot_lon)] + list(zip(cluster_df["Latitude"], cluster_df["Longitude"]))

        G = nx.complete_graph(len(locations))
        for i in range(len(locations)):
            for j in range(len(locations)):
                if i != j:
                    G[i][j]["weight"] = geodesic(locations[i], locations[j]).km

        tsp_order = traveling_salesman_problem(G, cycle=False, method="greedy")

        # Skip depot (index 0) in delivery path
        delivery_order = tsp_order[1:] if tsp_order[0] == 0 else tsp_order[:-1]
        cluster_df["Sequence"] = range(1, len(delivery_order) + 1)
        cluster_df["Sequence_Label"] = ["S" + str(i) for i in cluster_df["Sequence"]]

        # Sort for output
        cluster_df["TSP_Index"] = delivery_order
        cluster_df.sort_values("TSP_Index", inplace=True)

        all_routes.append(cluster_df)

    final_df = pd.concat(all_routes).reset_index(drop=True)

    # ----------------- Show Map -----------------
    st.subheader("🗺️ Clustered Delivery Routes")

    map_center = [df["Latitude"].mean(), df["Longitude"].mean()]
    folium_map = folium.Map(location=map_center, zoom_start=13)

    colors = ["red", "blue", "green", "purple", "orange", "darkred", "cadetblue", "black", "darkgreen", "darkblue"]
    for i, cluster_id in enumerate(sorted(final_df["Cluster_ID"].unique())):
        cluster_color = colors[i % len(colors)]
        cluster_data = final_df[final_df["Cluster_ID"] == cluster_id]
        folium.Marker(location=[depot_lat, depot_lon], icon=folium.Icon(color='black'), popup="Depot").add_to(folium_map)

        for idx, row in cluster_data.iterrows():
            folium.Marker(
                location=[row["Latitude"], row["Longitude"]],
                tooltip=f'{row["Society_Name"]} - {row["Orders"]} orders',
                popup=row["Sequence_Label"],
                icon=folium.Icon(color=cluster_color)
            ).add_to(folium_map)

        # Draw lines
        path = [(depot_lat, depot_lon)] + list(zip(cluster_data["Latitude"], cluster_data["Longitude"]))
        folium.PolyLine(path, color=cluster_color, weight=2.5).add_to(folium_map)

    st_folium(folium_map, width=1000, height=600)

    # ----------------- Output -----------------
    st.subheader("📄 Route Summary")
    st.dataframe(final_df[["Cluster_ID", "Society_ID", "Society_Name", "Orders", "Sequence_Label"]])

    # Download result
    def get_result_csv():
        return final_df[["Cluster_ID", "Society_ID", "Society_Name", "Latitude", "Longitude", "Orders", "Sequence_Label"]].to_csv(index=False)

    st.download_button("📥 Download Route Plan CSV", get_result_csv(), file_name="route_plan.csv")

else:
    st.info("Please upload the input file to continue.")
