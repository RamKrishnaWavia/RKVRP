import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import folium
from streamlit_folium import st_folium
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

st.set_page_config(page_title="RK Delivery Route Optimizer", layout="wide")
st.title("RK Delivery Route Optimizer")
st.markdown(
    "Upload delivery points CSV (Society ID, Society Name, City, Drop Point, Latitude, Longitude, Orders). "
    "Optimize routes ≤200 orders/vehicle, with depot start/end, clustering, OR-Tools route sequencing, "
    "cost per order calculation, interactive map with filters, and CSV download."
)

DEPOT = {"Name": "Soukya Road Depot", "Latitude": 13.0426, "Longitude": 77.6611}  
MAX_ORDERS = 200
DEFAULT_VEHICLE_COST = 35000

# Upload CSV
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV with columns: Society ID, Society Name, City, Drop Point, Latitude, Longitude, Orders", type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    required_cols = ["Society ID", "Society Name", "City", "Drop Point", "Latitude", "Longitude", "Orders"]
    if not all(col in df.columns for col in required_cols):
        st.error(f"Missing required columns: {required_cols}")
        st.stop()
    if df[["Latitude", "Longitude", "Orders"]].isnull().any().any():
        st.error("Latitude, Longitude and Orders must not have missing values.")
        st.stop()

    vehicle_cost = st.sidebar.number_input("Vehicle Monthly Cost (₹)", min_value=0, value=DEFAULT_VEHICLE_COST, step=1000)

    total_orders = df["Orders"].sum()
    num_clusters = max(1, int(np.ceil(total_orders / MAX_ORDERS)))
    if len(df) < num_clusters:
        num_clusters = len(df)

    coords = df[["Latitude", "Longitude"]].values
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(coords)
    df["Cluster ID"] = kmeans.labels_

    depot_df = pd.DataFrame(
        {
            "Society ID": ["DEPOT"],
            "Society Name": [DEPOT["Name"]],
            "City": ["Depot"],
            "Drop Point": ["Depot"],
            "Latitude": [DEPOT["Latitude"]],
            "Longitude": [DEPOT["Longitude"]],
            "Orders": [0],
            "Cluster ID": [-1],
        }
    )

    df_all = pd.concat([df, depot_df], ignore_index=True)

    def compute_distance_matrix(locations):
        n = len(locations)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist_matrix[i][j] = np.linalg.norm(locations[i] - locations[j])
        return dist_matrix

    def solve_tsp(dist_matrix):
        n = len(dist_matrix)
        manager = pywrapcp.RoutingIndexManager(n, 1, 0)
        routing = pywrapcp.RoutingModel(manager)
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(dist_matrix[from_node][to_node] * 1e6)
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        search_params = pywrapcp.DefaultRoutingSearchParameters()
        search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search_params.time_limit.seconds = 10
        solution = routing.SolveWithParameters(search_params)
        if solution:
            index = routing.Start(0)
            route = []
            while not routing.IsEnd(index):
                route.append(manager.IndexToNode(index))
                index = solution.Value(routing.NextVar(index))
            route.append(manager.IndexToNode(index))
            return route
        else:
            return None

    optimized_routes = []
    cluster_summaries = []

    for cluster_id in df["Cluster ID"].unique():
        cluster_data = df[df["Cluster ID"] == cluster_id].copy()
        cluster_orders = cluster_data["Orders"].sum()
        if cluster_orders == 0:
            continue
        cluster_locations = cluster_data[["Latitude", "Longitude"]].values
        depot_location = np.array([[DEPOT["Latitude"], DEPOT["Longitude"]]])
        locations = np.vstack([depot_location, cluster_locations])
        dist_matrix = compute_distance_matrix(locations)
        route = solve_tsp(dist_matrix)
        if route is None:
            st.error(f"Could not solve TSP for cluster {cluster_id}")
            continue
        cum_distance = 0
        for i in range(len(route) - 1):
            cum_distance += dist_matrix[route[i]][route[i+1]] * 111  # Approx km
        stop_order = []
        for idx, loc_idx in enumerate(route[1:-1], start=1):
            row = cluster_data.iloc[loc_idx - 1]
            stop_order.append({
                "Cluster ID": cluster_id,
                "Stop Sequence": idx,
                "Society ID": row["Society ID"],
                "Society Name": row["Society Name"],
                "City": row["City"],
                "Drop Point": row["Drop Point"],
                "Latitude": row["Latitude"],
                "Longitude": row["Longitude"],
                "Orders": row["Orders"],
            })
        cost_per_order = min(vehicle_cost / MAX_ORDERS, vehicle_cost / cluster_orders if cluster_orders else vehicle_cost)
        cluster_summaries.append({
            "Cluster ID": cluster_id,
            "Total Orders": cluster_orders,
            "Total Distance (km)": round(cum_distance, 2),
            "Cost per Order (₹)": round(cost_per_order, 2),
            "Number of Stops": len(cluster_data),
        })
        optimized_routes.extend(stop_order)

    if not optimized_routes:
        st.warning("No routes optimized. Please check data.")
        st.stop()

    df_routes = pd.DataFrame(optimized_routes)
    df_summary = pd.DataFrame(cluster_summaries)

    st.sidebar.header("Filters")
    city_filter = st.sidebar.multiselect("City", options=df_routes["City"].unique())
    society_filter = st.sidebar.multiselect("Society Name", options=df_routes["Society Name"].unique())
    drop_filter = st.sidebar.multiselect("Drop Point", options=df_routes["Drop Point"].unique())

    df_filtered = df_routes
    if city_filter:
        df_filtered = df_filtered[df_filtered["City"].isin(city_filter)]
    if society_filter:
        df_filtered = df_filtered[df_filtered["Society Name"].isin(society_filter)]
    if drop_filter:
        df_filtered = df_filtered[df_filtered["Drop Point"].isin(drop_filter)]

    st.subheader("Route Summary")
    st.dataframe(df_summary)

    st.subheader("Optimized Routes Map")
    m = folium.Map(location=[DEPOT["Latitude"], DEPOT["Longitude"]], zoom_start=10)
    folium.Marker([DEPOT["Latitude"], DEPOT["Longitude"]],
                  popup=f"<b>{DEPOT['Name']}</b><br>Depot",
                  icon=folium.Icon(color="red", icon="home")).add_to(m)

    for cid, group in df_filtered.groupby("Cluster ID"):
        points = [(DEPOT["Latitude"], DEPOT["Longitude"])]
        for _, row in group.iterrows():
            folium.Marker(
                location=[row["Latitude"], row["Longitude"]],
                popup=(f"<b>Stop Seq:</b> {row['Stop Sequence']}<br>"
                       f"<b>Society:</b> {row['Society Name']}<br>"
                       f"<b>City:</b> {row['City']}<br>"
                       f"<b>Drop Point:</b> {row['Drop Point']}<br>"
                       f"<b>Orders:</b> {row['Orders']}"),
                tooltip=f"Seq: {row['Stop Sequence']}",
                icon=folium.DivIcon(html=f'<div style="font-size: 12pt; color: blue;">{row["Stop Sequence"]}</div>')
            ).add_to(m)
            points.append((row["Latitude"], row["Longitude"]))
        points.append((DEPOT["Latitude"], DEPOT["Longitude"]))
        folium.PolyLine(points, color="blue", weight=3, opacity=0.6).add_to(m)

    st_folium(m, width=900, height=600)

    st.markdown("---")
    st.download_button("Download Optimized Routes CSV", data=df_routes.to_csv(index=False).encode("utf-8"), file_name="optimized_routes.csv")

else:
    st.info("Upload a CSV file with the required columns to start optimization.")

st.sidebar.markdown(
    """
    ### Sample CSV Format
    | Society ID | Society Name | City    | Drop Point | Latitude | Longitude | Orders |
    |------------|--------------|---------|------------|----------|-----------|--------|
    | S1         | Green Park   | Bangalore | Drop1     | 13.04    | 77.66     | 30     |
    | S2         | Blue Valley  | Bangalore | Drop2     | 13.05    | 77.67     | 50     |
    """
)
