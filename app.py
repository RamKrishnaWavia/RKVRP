import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from io import BytesIO

# Constants
VEHICLE_CAPACITY = 200
VEHICLE_COST_PER_MONTH = 35000
WORKING_DAYS_PER_MONTH = 30
VEHICLE_COST_PER_DAY = VEHICLE_COST_PER_MONTH / WORKING_DAYS_PER_MONTH

# Haversine distance function
def haversine(lat1, lon1, lat2, lon2):
    from math import radians, cos, sin, asin, sqrt
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    km = 6371 * c
    return km

# Distance matrix builder
def build_distance_matrix(locations):
    size = len(locations)
    matrix = {}
    for from_node in range(size):
        matrix[from_node] = {}
        for to_node in range(size):
            if from_node == to_node:
                matrix[from_node][to_node] = 0
            else:
                dist = haversine(
                    locations[from_node][0], locations[from_node][1],
                    locations[to_node][0], locations[to_node][1]
                )
                matrix[from_node][to_node] = int(dist * 1000)  # meters
    return matrix

# TSP Optimization using OR-Tools
def optimize_route(locations):
    tsp_size = len(locations)
    if tsp_size <= 1:
        return list(range(tsp_size)), 0

    num_routes = 1
    depot = 0

    distance_matrix = build_distance_matrix(locations)

    routing = pywrapcp.RoutingModel(tsp_size, num_routes, depot)
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    def distance_callback(from_index, to_index):
        return distance_matrix[from_index][to_index]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        index = routing.Start(0)
        route = []
        route_distance = 0
        while not routing.IsEnd(index):
            route.append(routing.IndexToNode(index))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
        route.append(routing.IndexToNode(index))
        return route, route_distance / 1000
    else:
        return list(range(tsp_size)), 0

# Main app
st.title("🚚 Milk Delivery Route Optimizer")

uploaded_file = st.file_uploader("Upload CSV with drop points", type=["csv"])
vehicle_cost_input = st.number_input("Enter monthly vehicle cost (₹)", value=35000, step=1000)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    required_cols = ['Society ID', 'Society Name', 'City', 'Drop Point', 'Latitude', 'Longitude', 'Orders']
    if not all(col in df.columns for col in required_cols):
        st.error(f"Missing required columns: {required_cols}")
    else:
        df['Orders'] = pd.to_numeric(df['Orders'], errors='coerce')
        invalid_orders = df[df['Orders'].isnull()]
        if not invalid_orders.empty:
            st.warning("⚠️ Some 'Orders' values are missing or non-numeric. Please check your CSV.")
            st.dataframe(invalid_orders)
        else:
            df['Orders'] = df['Orders'].astype(int)

            # Auto cluster calculation
            total_orders = df['Orders'].sum()
            num_clusters = int(np.ceil(total_orders / VEHICLE_CAPACITY))
            coords = df[['Latitude', 'Longitude']]
            kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(coords)
            df['Cluster ID'] = kmeans.labels_

            total_cpo = 0
            st.header("📍 Route Maps & Cost per Order Summary")

            for cluster_id in df['Cluster ID'].unique():
                cluster_df = df[df['Cluster ID'] == cluster_id].reset_index(drop=True)
                locations = list(zip(cluster_df['Latitude'], cluster_df['Longitude']))
                depot = (12.9345, 77.7746)  # Soukya Road
                locations.insert(0, depot)
                route, dist_km = optimize_route(locations)

                cluster_df['Sequence'] = route[1:-1]
                total_orders_cluster = cluster_df['Orders'].sum()
                cost_per_order = round((vehicle_cost_input / WORKING_DAYS_PER_MONTH) / total_orders_cluster, 2)
                total_cpo += cost_per_order

                st.subheader(f"Cluster {cluster_id} 🚛 (Orders: {total_orders_cluster}, CPO: ₹{cost_per_order})")

                m = folium.Map(location=depot, zoom_start=11)
                marker_cluster = MarkerCluster().add_to(m)

                for i, row in cluster_df.iterrows():
                    folium.Marker(
                        location=[row['Latitude'], row['Longitude']],
                        popup=f"{row['Society Name']} ({row['Orders']} orders)<br>Sequence: {row['Sequence']}",
                        tooltip=f"#{row['Sequence']}: {row['Society Name']}"
                    ).add_to(marker_cluster)

                st_data = st_folium(m, width=700, height=400)

            st.success(f"✅ Total estimated cost per order across all clusters: ₹{round(total_cpo / num_clusters, 2)}")
