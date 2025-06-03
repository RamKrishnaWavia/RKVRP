import streamlit as st
import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# Constants
AVERAGE_SPEED_KMPH = 25  # Average speed in gated societies
VEHICLE_CAPACITY = 20
TIME_WINDOW_START = 3.5 * 60 * 60  # 3:30 AM in seconds from midnight
TIME_WINDOW_END = 7 * 60 * 60      # 7:00 AM in seconds

# Haversine distance in km
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

def create_distance_time_matrix(locations):
    size = len(locations)
    dist_matrix = np.zeros((size, size))
    time_matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if i == j:
                dist_matrix[i][j] = 0
                time_matrix[i][j] = 0
            else:
                dist = haversine(locations[i][0], locations[i][1], locations[j][0], locations[j][1])
                dist_matrix[i][j] = dist
                # time in seconds
                time_matrix[i][j] = (dist / AVERAGE_SPEED_KMPH) * 3600
    return dist_matrix, time_matrix

def print_solution(manager, routing, solution, orders, vehicle_count):
    output = []
    total_distance = 0
    total_load = 0
    for vehicle_id in range(vehicle_count):
        index = routing.Start(vehicle_id)
        route_dist = 0
        route_load = 0
        route = []
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route.append(node_index)
            route_load += orders[node_index]
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_dist += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
        # Add depot at end
        route.append(manager.IndexToNode(index))
        total_distance += route_dist
        total_load += route_load
        output.append({
            "vehicle_id": vehicle_id + 1,
            "route": route,
            "orders_delivered": route_load,
            "distance_km": route_dist / 1000.0,  # cost scaled, we'll adjust below
        })
    return output

def main():
    st.title("Milk Delivery Route Optimizer")

    st.markdown("""
    Upload CSV with columns: `latitude`, `longitude`, `orders`
    - Depot location will be added manually in the app
    - Vehicle capacity fixed at 20 orders
    - Delivery window: 3:30 AM to 7:00 AM
    """)

    uploaded_file = st.file_uploader("Upload Delivery Points CSV", type=["csv"])
    
    depot_lat = st.number_input("Depot Latitude", value=12.9716)  # Default Bangalore center
    depot_lon = st.number_input("Depot Longitude", value=77.5946)

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())

        # Validate columns
        if not {'latitude', 'longitude', 'orders'}.issubset(df.columns):
            st.error("CSV must contain 'latitude', 'longitude' and 'orders' columns")
            return

        # Prepare locations list - depot at index 0
        locations = [(depot_lat, depot_lon)] + list(zip(df['latitude'], df['longitude']))
        orders = [0] + df['orders'].tolist()  # depot has zero orders

        # Create distance and time matrix
        dist_matrix_km, time_matrix_sec = create_distance_time_matrix(locations)
        
        # OR-Tools expects integer cost matrix, scale by 1000 to convert to meters and int
        dist_matrix = (dist_matrix_km * 1000).astype(int)
        time_matrix = time_matrix_sec.astype(int)

        # Create routing index manager
        vehicle_count = (sum(orders) // VEHICLE_CAPACITY) + 1
        manager = pywrapcp.RoutingIndexManager(len(dist_matrix), vehicle_count, 0)
        routing = pywrapcp.RoutingModel(manager)

        # Distance callback
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return dist_matrix[from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Add Capacity constraint
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return orders[from_node]

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            [VEHICLE_CAPACITY] * vehicle_count,
            True,  # start cumul to zero
            'Capacity'
        )

        # Add Time Window constraint
        def time_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return time_matrix[from_node][to_node]

        time_callback_index = routing.RegisterTransitCallback(time_callback)
        routing.AddDimension(
            time_callback_index,
            30 * 60,  # allow 30 mins waiting slack
            TIME_WINDOW_END - TIME_WINDOW_START,  # max time per vehicle
            False,
            'Time'
        )
        time_dimension = routing.GetDimensionOrDie('Time')

        # Add time window for depot
        time_dimension.CumulVar(routing.Start(0)).SetRange(TIME_WINDOW_START, TIME_WINDOW_END)
        time_dimension.CumulVar(routing.End(0)).SetRange(TIME_WINDOW_START, TIME_WINDOW_END)

        # Add time windows for delivery locations
        for node in range(1, len(locations)):
            time_dimension.CumulVar(manager.NodeToIndex(node)).SetRange(TIME_WINDOW_START, TIME_WINDOW_END)

        # Setting first solution heuristic
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

        # Solve the problem
        solution = routing.SolveWithParameters(search_parameters)

        if solution:
            # Output routes
            st.success(f"Solution found with {vehicle_count} vehicles")
            routes = print_solution(manager, routing, solution, orders, vehicle_count)
            for r in routes:
                st.markdown(f"**Vehicle {r['vehicle_id']} route:**")
                route_str = " -> ".join(str(node) for node in r['route'])
                st.write(route_str)
                st.write(f"Orders Delivered: {r['orders_delivered']}")
                st.write(f"Route Distance (approx): {r['distance_km']:.2f} km")
                st.write("---")
        else:
            st.error("No solution found.")

if __name__ == '__main__':
    main()
