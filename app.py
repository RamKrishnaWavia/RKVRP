import streamlit as st
import pandas as pd
import numpy as np
import math
from sklearn.cluster import KMeans
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import pydeck as pdk

st.set_page_config(page_title="Milk Delivery Route Optimizer", layout="wide")

VEHICLE_CAPACITY = 200
DEFAULT_VEHICLE_COST = 35000
DEPOT_NAME = "Soukya Road"


def load_template():
    template = pd.DataFrame({
        "Society ID": ["S001", "S002"],
        "Society Name": ["Green Apartments", "Blue Residency"],
        "City": ["Bangalore", "Bangalore"],
        "Drop Point": ["DP1", "DP2"],
        "Latitude": [13.0476, 13.0587],
        "Longitude": [77.5935, 77.6011],
        "Orders": [120, 90]
    })
    return template


def compute_euclidean_distance_matrix(locations):
    distances = {}
    for from_counter, from_node in enumerate(locations):
        distances[from_counter] = {}
        for to_counter, to_node in enumerate(locations):
            if from_counter == to_counter:
                distances[from_counter][to_counter] = 0
            else:
                distances[from_counter][to_counter] = np.linalg.norm(np.array(from_node) - np.array(to_node))
    return distances


def create_data_model(cluster_df, depot_index=0):
    locations = [(cluster_df.iloc[depot_index]['Latitude'], cluster_df.iloc[depot_index]['Longitude'])] + \
                [(row['Latitude'], row['Longitude']) for idx, row in cluster_df.iterrows() if idx != depot_index]

    demands = [0] + [row['Orders'] for idx, row in cluster_df.iterrows() if idx != depot_index]
    data = {
        'distance_matrix': compute_euclidean_distance_matrix(locations),
        'demands': demands,
        'vehicle_capacities': [VEHICLE_CAPACITY],
        'num_vehicles': 1,
        'depot': 0
    }
    return data


def optimize_route(cluster_df):
    data = create_data_model(cluster_df)
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(data['distance_matrix'][from_node][to_node] * 1000)

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(demand_callback_index, 0, data['vehicle_capacities'], True, 'Capacity')

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.time_limit.seconds = 10
    search_parameters.log_search = False
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    solution = routing.SolveWithParameters(search_parameters)
    if not solution:
        return None, None

    index = routing.Start(0)
    route = []
    dist_m = 0
    while not routing.IsEnd(index):
        node_index = manager.IndexToNode(index)
        route.append(node_index)
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        dist_m += routing.GetArcCostForVehicle(previous_index, index, 0)
    route.append(manager.IndexToNode(index))

    return route, dist_m / 1000


def prepare_routes(df, vehicle_cost):
    total_orders = df['Orders'].sum()
    num_clusters = math.ceil(total_orders / VEHICLE_CAPACITY)
    num_clusters = min(num_clusters, len(df))
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    coords = df[['Latitude', 'Longitude']]
    df['Cluster ID'] = kmeans.fit_predict(coords)

    all_routes, route_costs = [], []
    for cluster_id in sorted(df['Cluster ID'].unique()):
        cluster_df = df[df['Cluster ID'] == cluster_id].reset_index(drop=True)
        depot_row = pd.DataFrame([{
            'Society ID': 'Depot', 'Society Name': DEPOT_NAME, 'City': cluster_df.iloc[0]['City'],
            'Drop Point': 'Depot', 'Latitude': 13.0604, 'Longitude': 77.6089, 'Orders': 0, 'Cluster ID': cluster_id
        }])
        cluster_df = pd.concat([depot_row, cluster_df], ignore_index=True)
        route, dist_km = optimize_route(cluster_df)
        if route is None:
            st.warning(f"No route for cluster {cluster_id}")
            continue

        route_points = []
        for seq, node_idx in enumerate(route):
            row = cluster_df.iloc[node_idx]
            route_points.append({
                'Sequence': seq, 'Society ID': row['Society ID'], 'Society Name': row['Society Name'],
                'City': row['City'], 'Drop Point': row['Drop Point'], 'Latitude': row['Latitude'],
                'Longitude': row['Longitude'], 'Orders': row['Orders']
            })
        route_df = pd.DataFrame(route_points)
        cost_per_order = vehicle_cost / max(cluster_df['Orders'].sum(), 1)
        route_df['Cluster ID'] = cluster_id
        route_df['Route Distance (km)'] = dist_km
        route_df['Cost Per Order (₹)'] = round(cost_per_order, 2)
        all_routes.append(route_df)
        route_costs.append({
            'Cluster ID': cluster_id, 'Distance (km)': round(dist_km, 2),
            'Total Orders': cluster_df['Orders'].sum(), 'Cost Per Order (₹)': round(cost_per_order, 2)
        })

    if all_routes:
        return pd.concat(all_routes, ignore_index=True), pd.DataFrame(route_costs)
    return None, None


def main():
    st.title("Milk Delivery Route Optimization with Dynamic Clustering")
    st.download_button("Download Sample Template", load_template().to_csv(index=False), file_name="template.csv")
    uploaded_file = st.file_uploader("Upload your delivery points CSV", type=["csv"])

    vehicle_cost = st.number_input("Monthly vehicle cost (₹)", 10000, 100000, DEFAULT_VEHICLE_COST, 1000)

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        required_cols = ['Society ID', 'Society Name', 'City', 'Drop Point', 'Latitude', 'Longitude', 'Orders']
        if not all(col in df.columns for col in required_cols):
            st.error(f"Missing columns. Required: {required_cols}")
            return

        with st.expander("Filter Data"):
            city_filter = st.multiselect("City", df['City'].unique().tolist(), default=df['City'].unique().tolist())
            df = df[df['City'].isin(city_filter)]

        if st.button("Run Optimization"):
            with st.spinner("Optimizing routes..."):
                routes_df, summary_df = prepare_routes(df, vehicle_cost)
                if routes_df is None:
                    st.error("No routes generated.")
                    return

                st.dataframe(summary_df)
                st.dataframe(routes_df)
                st.download_button("Download Optimized Routes", routes_df.to_csv(index=False), "optimized_routes.csv")

if __name__ == "__main__":
    main()
