import streamlit as st
import pandas as pd
import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from geopy.distance import great_circle

st.title("Milk Delivery Route Optimizer with Society Details")

uploaded_file = st.file_uploader(
    "Upload CSV with columns: Society ID,Society Name,City,Drop Point,Latitude,Longitude,Orders", type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    required_cols = ['Society ID', 'Society Name', 'City', 'Drop Point', 'Latitude', 'Longitude', 'Orders']
    if not set(required_cols).issubset(df.columns):
        st.error(f"CSV must contain columns: {required_cols}")
        st.stop()

    st.write("Input data preview:", df.head())

    # Distribution Center (DC) location (Set your DC latitude and longitude here)
    DC_LATITUDE = 12.9716  # example Bangalore
    DC_LONGITUDE = 77.5946

    # Create location list including DC at index 0
    locations = [(DC_LATITUDE, DC_LONGITUDE)] + list(zip(df['Latitude'], df['Longitude']))
    demands = [0] + list(df['Orders'])

    VEHICLE_CAPACITY = 200
    VEHICLE_COST = 1200

    # Distance matrix (in meters)
    def create_distance_matrix(locations):
        size = len(locations)
        dist_matrix = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                if i == j:
                    dist_matrix[i][j] = 0
                else:
                    dist_matrix[i][j] = great_circle(locations[i], locations[j]).kilometers * 1000
        return dist_matrix.astype(int)

    dist_matrix = create_distance_matrix(locations)

    # Routing model setup
    manager = pywrapcp.RoutingIndexManager(len(dist_matrix), 50, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        return dist_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    def demand_callback(from_index):
        return demands[manager.IndexToNode(from_index)]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,
        [VEHICLE_CAPACITY]*50,
        True,
        'Capacity'
    )

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = 30

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        routes_output = []
        total_orders_all = 0
        total_cost_all = 0

        for vehicle_id in range(50):
            index = routing.Start(vehicle_id)
            if routing.IsEnd(solution.Value(routing.NextVar(index))):
                # no nodes in route
                continue

            route_nodes = []
            route_orders = 0
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                if node_index != 0:
                    route_nodes.append(node_index - 1)  # Adjust for DC at 0
                    route_orders += demands[node_index]
                index = solution.Value(routing.NextVar(index))

            vehicle_cost = VEHICLE_COST
            cost_per_order = vehicle_cost / route_orders if route_orders > 0 else 0
            total_orders_all += route_orders
            total_cost_all += vehicle_cost

            # Extract full details for route nodes
            route_details = df.iloc[route_nodes][
                ['Society ID', 'Society Name', 'City', 'Drop Point', 'Latitude', 'Longitude', 'Orders']
            ].copy()
            route_details['Route ID'] = vehicle_id + 1
            route_details['Vehicle Cost (Rs)'] = vehicle_cost
            route_details['Cost per Order (Rs)'] = round(cost_per_order, 2)

            routes_output.append(route_details)

        final_routes_df = pd.concat(routes_output, ignore_index=True)

        st.write("Optimized routes with all society details:")
        st.dataframe(final_routes_df)

        st.write(f"Total vehicles required: {len(routes_output)}")
        st.write(f"Total orders delivered: {total_orders_all}")
        st.write(f"Total delivery cost: Rs {total_cost_all}")
        st.write(f"Average cost per order: Rs {round(total_cost_all/total_orders_all, 2)}")

        csv = final_routes_df.to_csv(index=False).encode()
        st.download_button("Download full optimized routes CSV", csv, "optimized_routes_full.csv", "text/csv")

    else:
        st.error("No solution found!")
else:
    st.info("Please upload the input CSV file.")
