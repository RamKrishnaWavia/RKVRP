import streamlit as st
import pandas as pd
import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from geopy.distance import great_circle

st.title("Milk Delivery Route Optimization with Vehicle Capacity & Cost")

uploaded_file = st.file_uploader("Upload CSV with columns: Society ID,Society Name,City,Drop Point,Latitude,Longitude,Orders", type="csv")

VEHICLE_CAPACITY = 200
VEHICLE_COST_PER_DAY = 1200
MAX_VEHICLES = 50  # max vehicles allowed

def create_distance_matrix(locations):
    size = len(locations)
    dist_matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if i == j:
                dist_matrix[i][j] = 0
            else:
                dist_matrix[i][j] = great_circle(locations[i], locations[j]).kilometers * 1000  # meters
    return dist_matrix.astype(int)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    expected_cols = ['Society ID', 'Society Name', 'City', 'Drop Point', 'Latitude', 'Longitude', 'Orders']
    if not all(col in df.columns for col in expected_cols):
        st.error(f"CSV missing some required columns. Required: {expected_cols}")
        st.stop()

    # Basic validation
    if (df['Orders'] <= 0).any():
        st.error("Orders must be positive integers")
        st.stop()
    if df[['Latitude','Longitude']].isnull().any().any():
        st.error("Latitude and Longitude cannot have missing values")
        st.stop()

    total_orders = df['Orders'].sum()
    st.write(f"Total orders in input: {total_orders}")

    # Calculate minimum vehicles needed by capacity
    min_vehicles_needed = int(np.ceil(total_orders / VEHICLE_CAPACITY))
    st.write(f"Minimum vehicles needed (capacity {VEHICLE_CAPACITY} orders): {min_vehicles_needed}")

    if min_vehicles_needed > MAX_VEHICLES:
        st.error(f"Required vehicles exceed max allowed {MAX_VEHICLES}. Please increase vehicle count or capacity.")
        st.stop()

    # Locations with DC at index 0 (example DC location, replace with your actual DC lat,long)
    DC_LATITUDE = 12.9716
    DC_LONGITUDE = 77.5946

    locations = [(DC_LATITUDE, DC_LONGITUDE)] + list(zip(df['Latitude'], df['Longitude']))
    demands = [0] + df['Orders'].tolist()

    dist_matrix = create_distance_matrix(locations)

    manager = pywrapcp.RoutingIndexManager(len(dist_matrix), MAX_VEHICLES, 0)
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
        [VEHICLE_CAPACITY]*MAX_VEHICLES,
        True,
        'Capacity'
    )

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = 30

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        all_routes = []
        total_cost = 0
        total_orders_delivered = 0

        for vehicle_id in range(MAX_VEHICLES):
            index = routing.Start(vehicle_id)
            if routing.IsEnd(solution.Value(routing.NextVar(index))):
                continue  # no stops for this vehicle

            route_order = 0
            route_nodes = []

            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                if node_index != 0:
                    route_nodes.append(node_index - 1)  # zero-based index for df
                    route_order += demands[node_index]
                index = solution.Value(routing.NextVar(index))

            if route_order == 0:
                continue

            vehicle_cost = VEHICLE_COST_PER_DAY
            cost_per_order = vehicle_cost / route_order

            total_cost += vehicle_cost
            total_orders_delivered += route_order

            route_df = df.iloc[route_nodes].copy()
            route_df['Route ID'] = vehicle_id + 1
            route_df['Vehicle Cost (Rs)'] = vehicle_cost
            route_df['Cost per Order (Rs)'] = round(cost_per_order, 2)

            # Add delivery sequence in route order
            route_df['Delivery Sequence'] = list(range(1, len(route_df)+1))

            all_routes.append(route_df)

        final_df = pd.concat(all_routes, ignore_index=True)

        st.write(f"Total vehicles used: {final_df['Route ID'].nunique()}")
        st.write(f"Total orders delivered: {total_orders_delivered}")
        st.write(f"Total delivery cost: Rs {total_cost}")
        st.write(f"Average cost per order: Rs {round(total_cost / total_orders_delivered, 2)}")

        st.dataframe(final_df[[
            'Route ID','Delivery Sequence','Society ID', 'Society Name', 'City', 'Drop Point', 
            'Latitude', 'Longitude', 'Orders', 'Vehicle Cost (Rs)', 'Cost per Order (Rs)'
        ]])

        csv = final_df.to_csv(index=False).encode()
        st.download_button("Download Optimized Routes CSV", csv, "optimized_routes.csv", "text/csv")
    else:
        st.error("No solution found. Try increasing time limit or reducing number of stops.")
else:
    st.info("Upload your society orders CSV file to start optimization.")
