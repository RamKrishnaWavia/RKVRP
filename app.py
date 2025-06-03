import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import math

# Constants
DEPOT_NAME = "Soukya Road"
DEPOT_LAT = 12.9976
DEPOT_LON = 77.7635
MAX_ORDERS_PER_VEHICLE = 200
DEFAULT_VEHICLE_COST = 35000

# Load Template
TEMPLATE = pd.DataFrame({
    'Society ID': ['BLR001'],
    'Society Name': ['Green Heights'],
    'City': ['Bangalore'],
    'Drop Point': ['Warehouse A'],
    'Latitude': [12.923],
    'Longitude': [77.614],
    'Orders': [120]
})

# Helper functions

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def compute_distance_matrix(locations):
    size = len(locations)
    dist_matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            dist_matrix[i][j] = haversine(*locations[i], *locations[j])
    return dist_matrix

def create_data_model(cluster_df):
    locations = [(DEPOT_LAT, DEPOT_LON)] + list(zip(cluster_df['Latitude'], cluster_df['Longitude']))
    demands = [0] + list(cluster_df['Orders'])
    distance_matrix = compute_distance_matrix(locations)
    return {
        'distance_matrix': distance_matrix,
        'demands': demands,
        'vehicle_capacities': [MAX_ORDERS_PER_VEHICLE],
        'num_vehicles': 1,
        'depot': 0
    }

def optimize_route(cluster_df):
    data = create_data_model(cluster_df)
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        return int(data['distance_matrix'][manager.IndexToNode(from_index)][manager.IndexToNode(to_index)] * 1000)

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    def demand_callback(from_index):
        return data['demands'][manager.IndexToNode(from_index)]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index, 0, data['vehicle_capacities'], True, 'Capacity')

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        index = routing.Start(0)
        route = []
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route.append(node_index)
            index = solution.Value(routing.NextVar(index))
        route.append(manager.IndexToNode(index))

        ordered_df = cluster_df.iloc[[i-1 for i in route[1:-1]]].copy()
        ordered_df.insert(0, 'Route Sequence', range(1, len(ordered_df) + 1))
        total_distance = sum(data['distance_matrix'][route[i]][route[i+1]] for i in range(len(route)-1))
        return ordered_df, total_distance
    else:
        return pd.DataFrame(), 0

# Streamlit App
st.set_page_config(layout="wide")
st.title("Optimized Route Planning - Cost Effective Delivery")

uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
st.download_button("Download Template", TEMPLATE.to_csv(index=False).encode(), "route_template.csv")
vehicle_cost = st.number_input("Monthly Cost Per Vehicle (₹)", value=DEFAULT_VEHICLE_COST)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    required_cols = ['Society ID', 'Society Name', 'City', 'Drop Point', 'Latitude', 'Longitude', 'Orders']
    if not all(col in df.columns for col in required_cols):
        st.error(f"Missing required columns: {required_cols}")
    else:
        df = df.dropna(subset=required_cols)
        df = df[df['Orders'] > 0]
        df = df.reset_index(drop=True)

        num_clusters = int(np.ceil(df['Orders'].sum() / MAX_ORDERS_PER_VEHICLE))
        n_samples = df.shape[0]
        num_clusters = min(num_clusters, n_samples)  # Adjust to avoid more clusters than data points

    if n_samples == 0:
        st.error("No data available to cluster.")
        return

    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(df[['Latitude', 'Longitude']])
        df['Cluster ID'] = kmeans.labels_

        route_dfs = []
        map_obj = folium.Map(location=[DEPOT_LAT, DEPOT_LON], zoom_start=12)
        cluster_group = MarkerCluster().add_to(map_obj)

        for cluster_id in df['Cluster ID'].unique():
            cluster_data = df[df['Cluster ID'] == cluster_id].copy()
            optimized_df, dist = optimize_route(cluster_data)
            if not optimized_df.empty:
                cpo = (vehicle_cost / 30) / optimized_df['Orders'].sum()
                optimized_df['Cluster ID'] = cluster_id
                optimized_df['Distance (km)'] = dist
                optimized_df['CPO (₹)'] = round(cpo, 2)
                route_dfs.append(optimized_df)

                # Add to map
                for _, row in optimized_df.iterrows():
                    folium.Marker(
                        location=[row['Latitude'], row['Longitude']],
                        popup=f"{row['Route Sequence']}. {row['Society Name']} (CPO ₹{row['CPO (₹)']})",
                        tooltip=row['Society ID']
                    ).add_to(cluster_group)

        if route_dfs:
            final_df = pd.concat(route_dfs, ignore_index=True)
            st.dataframe(final_df)
            st.download_button("Download Optimized Routes", final_df.to_csv(index=False).encode(), "optimized_routes.csv")
            st_folium(map_obj, width=1000)
        else:
            st.warning("Could not optimize any routes. Please check data.")
