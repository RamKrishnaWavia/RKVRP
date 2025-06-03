import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import math
import io

st.set_page_config(page_title="Milk Delivery Route Optimizer", layout="wide")

# Constants
DEFAULT_VEHICLE_COST = 35000
VEHICLE_CAPACITY = 200
TARGET_CPO = 4.0
DEPOT_NAME = "Soukya Road"

# Helper function to compute haversine distance
def haversine_distance(coord1, coord2):
    R = 6371  # Earth radius in km
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

# Distance matrix
def create_distance_matrix(locations):
    size = len(locations)
    matrix = {}
    for from_node in range(size):
        matrix[from_node] = {}
        for to_node in range(size):
            matrix[from_node][to_node] = haversine_distance(locations[from_node], locations[to_node])
    return matrix

# OR-Tools routing function
def optimize_cluster_routes(df):
    if len(df) == 0:
        return [], 0

    depot = df[df['Drop Point'] == DEPOT_NAME].iloc[0]
    df = pd.concat([pd.DataFrame([depot]), df[df['Drop Point'] != DEPOT_NAME]])

    locations = list(zip(df['Latitude'], df['Longitude']))
    distance_matrix = create_distance_matrix(locations)
    orders = df['Orders'].tolist()

    manager = pywrapcp.RoutingIndexManager(len(distance_matrix), 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(distance_matrix[from_node][to_node] * 1000)  # in meters

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return int(orders[from_node])

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        [VEHICLE_CAPACITY],  # vehicle maximum capacity
        True,
        'Capacity')

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.time_limit.FromSeconds(10)

    solution = routing.SolveWithParameters(search_parameters)

    if solution is None:
        return [], 0

    index = routing.Start(0)
    route = []
    total_distance_km = 0

    while not routing.IsEnd(index):
        node_index = manager.IndexToNode(index)
        route.append({
            'Society ID': df.iloc[node_index]['Society ID'],
            'Society Name': df.iloc[node_index]['Society Name'],
            'Drop Point': df.iloc[node_index]['Drop Point'],
            'Latitude': df.iloc[node_index]['Latitude'],
            'Longitude': df.iloc[node_index]['Longitude'],
            'Orders': df.iloc[node_index]['Orders']
        })
        next_index = solution.Value(routing.NextVar(index))
        total_distance_km += distance_matrix[node_index][manager.IndexToNode(next_index)]
        index = next_index

    return route, round(total_distance_km, 2)

# Streamlit App
st.title("Milk Delivery Route Optimization")
st.markdown("Upload your delivery data and get optimal routes to reduce cost per order.")

sample_template = pd.DataFrame({
    'Society ID': ["S001", "S002"],
    'Society Name': ["Alpha", "Beta"],
    'City': ["Bangalore", "Bangalore"],
    'Drop Point': ["Soukya Road", "Drop Point 1"],
    'Latitude': [12.987, 12.989],
    'Longitude': [77.678, 77.679],
    'Orders': [50, 100]
})

st.download_button("Download Input Template", data=sample_template.to_csv(index=False), file_name="input_template.csv")

uploaded_file = st.file_uploader("Upload Delivery Data CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    required_columns = ['Society ID', 'Society Name', 'City', 'Drop Point', 'Latitude', 'Longitude', 'Orders']
    if not all(col in df.columns for col in required_columns):
        st.error(f"Missing required columns: {required_columns}")
    else:
        vehicle_cost = st.number_input("Enter Vehicle Monthly Cost (₹)", value=DEFAULT_VEHICLE_COST)

        num_clusters = math.ceil(df['Orders'].sum() / VEHICLE_CAPACITY)
        kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(df[['Latitude', 'Longitude']])
        df['Cluster ID'] = kmeans.labels_

        all_routes = []
        summary_data = []

        for cluster_id in sorted(df['Cluster ID'].unique()):
            cluster_df = df[df['Cluster ID'] == cluster_id]
            route, dist_km = optimize_cluster_routes(cluster_df)
            if route:
                total_orders = cluster_df['Orders'].sum()
                cpo = (vehicle_cost / 30) / total_orders
                for i, stop in enumerate(route):
                    stop['Sequence'] = i + 1
                    stop['Cluster ID'] = cluster_id
                    stop['CPO'] = round(cpo, 2)
                    stop['Distance (km)'] = dist_km
                all_routes.extend(route)
                summary_data.append({
                    'Cluster ID': cluster_id,
                    'Total Orders': total_orders,
                    'Distance (km)': dist_km,
                    'CPO': round(cpo, 2)
                })

        if all_routes:
            st.subheader("Optimized Routes Map")
            m = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=12)
            for stop in all_routes:
                folium.Marker(
                    location=[stop['Latitude'], stop['Longitude']],
                    popup=f"{stop['Sequence']}. {stop['Society Name']} (Orders: {stop['Orders']}, CPO: ₹{stop['CPO']})",
                    tooltip=f"Cluster {stop['Cluster ID']}"
                ).add_to(m)
            st_data = st_folium(m, width=1000, height=600)

            result_df = pd.DataFrame(all_routes)
            st.subheader("Route Summary")
            st.dataframe(pd.DataFrame(summary_data))
            st.download_button("Download Optimized Routes", data=result_df.to_csv(index=False), file_name="optimized_routes.csv")
