import streamlit as st
import pandas as pd
import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from geopy.distance import great_circle

st.title("Milk Delivery Route Optimizer")

# --- Input CSV upload ---
uploaded_file = st.file_uploader("Upload CSV with columns: latitude, longitude, orders", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Input data preview:", df.head())

    # Check for required columns
    if not set(['latitude', 'longitude', 'orders']).issubset(df.columns):
        st.error("CSV must contain 'latitude', 'longitude' and 'orders' columns")
        st.stop()

    # Distribution Center (DC) location (Set your DC latitude and longitude here)
    DC_LATITUDE = 28.7041   # Example: Delhi
    DC_LONGITUDE = 77.1025

    # Add DC as the first location in the data for routing start/end
    locations = [(DC_LATITUDE, DC_LONGITUDE)] + list(zip(df['latitude'], df['longitude']))
    demands = [0] + list(df['orders'])

    # Vehicle capacity parameters
    VEHICLE_CAPACITY = 200
    VEHICLE_COST = 1200  # Rs per vehicle per day

    # Calculate distance matrix using great circle distance in meters
    def create_distance_matrix(locations):
        size = len(locations)
        dist_matrix = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                if i == j:
                    dist_matrix[i][j] = 0
                else:
                    dist_matrix[i][j] = great_circle(locations[i], locations[j]).kilometers * 1000  # in meters
        return dist_matrix.astype(int)

    dist_matrix = create_distance_matrix(locations)

    # Create the routing index manager
    manager = pywrapcp.RoutingIndexManager(len(dist_matrix), 50, 0)  # Max 50 vehicles, depot=0

    # Create Routing Model
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback (distance callback)
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return dist_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Capacity constraint
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return demands[from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        [VEHICLE_CAPACITY]*50,  # vehicle max capacities
        True,  # start cumul to zero
        'Capacity'
    )

    # Setting first solution heuristic (cheapest addition)
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.seconds = 30
    search_parameters.log_search = False

    # Solve the problem
    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        total_orders_all = 0
        total_cost_all = 0
        results = []

        def get_route_details(vehicle_id):
            index = routing.Start(vehicle_id)
            route = []
            route_orders = 0
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                if node_index != 0:
                    route.append(node_index)  # exclude depot in route output
                    route_orders += demands[node_index]
                index = solution.Value(routing.NextVar(index))
            return route, route_orders

        # Collect routes
        for vehicle_id in range(50):
            route, route_orders = get_route_details(vehicle_id)
            if len(route) == 0:
                continue
            cost = VEHICLE_COST
            cost_per_order = cost / route_orders if route_orders > 0 else 0
            total_orders_all += route_orders
            total_cost_all += cost
            results.append({
                'Route ID': vehicle_id + 1,
                'Number of Orders': route_orders,
                'Vehicle Cost (Rs)': cost,
                'Cost per Order (Rs)': round(cost_per_order, 2),
                'Delivery Sequence': route
            })

        result_df = pd.DataFrame(results)
        st.write("Optimized routes summary:")
        st.dataframe(resu
