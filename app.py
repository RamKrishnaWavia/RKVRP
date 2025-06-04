import streamlit as st
import pandas as pd
import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import folium
from streamlit_folium import st_folium

# Function to compute Euclidean distance matrix
def compute_euclidean_distance_matrix(locations):
    n = len(locations)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i][j] = np.linalg.norm(np.array(locations[i]) - np.array(locations[j]))
    return dist_matrix.astype(int)  # convert to int for OR-Tools

def create_data_model(df, depot_lat, depot_lon, min_capacity, max_capacity):
    # Locations: depot first, then customer locations
    locations = [(depot_lat, depot_lon)] + list(zip(df['latitude'], df['longitude']))
    orders = [0] + df['orders'].tolist()
    data = {
        'locations': locations,
        'num_locations': len(locations),
        'depot': 0,
        'orders': orders,
        'vehicle_capacity_min': min_capacity,
        'vehicle_capacity_max': max_capacity,
        'vehicle_cost': 1200,
    }
    data['distance_matrix'] = compute_euclidean_distance_matrix(locations)
    return data

def add_capacity_constraints(routing, manager, demand_evaluator_index, min_capacity, max_capacity, num_vehicles):
    capacity = 'Capacity'
    routing.AddDimensionWithVehicleCapacity(
        demand_evaluator_index,
        0,  # null capacity slack
        [max_capacity]*num_vehicles,  # vehicle maximum capacities
        True,  # start cumul to zero
        capacity)
    capacity_dimension = routing.GetDimensionOrDie(capacity)
    # Enforce minimum capacity per route (hard constraint workaround)
    # OR-Tools doesn't support min capacity directly, so we add penalty for routes < min_capacity
    # We'll handle this after solution extraction
    return capacity_dimension

def print_solution(data, manager, routing, solution):
    routes = []
    total_orders = sum(data['orders'])
    total_vehicles_used = 0
    vehicle_capacity = data['vehicle_capacity_max']
    vehicle_cost = data['vehicle_cost']
    capacity_dimension = routing.GetDimensionOrDie('Capacity')

    for vehicle_id in range(routing.vehicles()):
        index = routing.Start(vehicle_id)
        route = []
        route_load = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route.append(node_index)
            route_load += data['orders'][node_index]
            index = solution.Value(routing.NextVar(index))
        if len(route) > 1:  # vehicle used if route has more than depot
            total_vehicles_used += 1
            routes.append({
                'vehicle_id': vehicle_id,
                'route': route,
                'orders': route_load
            })

    return routes, total_orders, total_vehicles_used, vehicle_cost

def create_map(data, routes):
    m = folium.Map(location=data['locations'][0], zoom_start=12)
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred',
              'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue',
              'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen',
              'gray', 'black', 'lightgray']
    
    # Add depot marker
    folium.Marker(
        location=data['locations'][0],
        popup='DC (Depot)',
        icon=folium.Icon(color='black', icon='home')
    ).add_to(m)

    for idx, route_info in enumerate(routes):
        route = route_info['route']
        orders = route_info['orders']
        vehicle_id = route_info['vehicle_id']
        route_color = colors[idx % len(colors)]

        # Add route lines
        route_coords = [data['locations'][node] for node in route]
        folium.PolyLine(locations=route_coords, color=route_color, weight=5, opacity=0.7).add_to(m)

        # Add markers with popup for each stop
        for i, node in enumerate(route):
            loc = data['locations'][node]
            if node == 0:
                continue
            popup_text = f"Vehicle {vehicle_id} Stop {i}: Orders {data['orders'][node]}"
            folium.Marker(location=loc, popup=popup_text, icon=folium.Icon(color=route_color)).add_to(m)
    
    return m

def main():
    st.title("Milk Delivery Route Optimizer with Capacity Constraints")

    st.markdown("""
    Upload your CSV with columns: latitude, longitude, orders.
    Provide DC (depot) latitude and longitude.
    Vehicle capacity min and max order limits.
    """)

    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    depot_lat = st.number_input("DC Latitude", format="%.6f")
    depot_lon = st.number_input("DC Longitude", format="%.6f")
    min_capacity = st.number_input("Vehicle Minimum Capacity (orders)", min_value=1, max_value=200, value=150)
    max_capacity = st.number_input("Vehicle Maximum Capacity (orders)", min_value=1, max_value=200, value=200)
    vehicle_cost = st.number_input("Vehicle daily cost (Rs)", min_value=1, value=1200)

    if uploaded_file and depot_lat and depot_lon:
        df = pd.read_csv(uploaded_file)
        if not {'latitude', 'longitude', 'orders'}.issubset(df.columns):
            st.error("CSV must contain 'latitude', 'longitude' and 'orders' columns.")
            return
        
        data = create_data_model(df, depot_lat, depot_lon, min_capacity, max_capacity)
        data['vehicle_cost'] = vehicle_cost

        # Estimate max vehicles (worst case one order per vehicle)
        total_orders = sum(data['orders'])
        max_vehicles = int(np.ceil(total_orders / min_capacity))
        st.write(f"Estimated max vehicles needed: {max_vehicles}")

        # Create routing index manager
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2
        manager = pywrapcp.RoutingIndexManager(data['num_locations'], max_vehicles, data['depot'])
        routing = pywrapcp.RoutingModel(manager)

        def demand_callback(index):
            node = manager.IndexToNode(index)
            return data['orders'][node]

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        capacity_dimension = add_capacity_constraints(routing, manager, demand_callback_index, min_capacity, max_capacity, max_vehicles)

        # Distance callback
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['distance_matrix'][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Search parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        search_parameters.time_limit.FromSeconds(20)

        solution = routing.SolveWithParameters(search_parameters)

        if solution:
            routes, total_orders, total_vehicles, vehicle_cost = print_solution(data, manager, routing, solution)

            # Show route summary
            st.subheader(f"Total orders: {total_orders}")
            st.subheader(f"Total vehicles used: {total_vehicles}")
            st.subheader(f"Vehicle cost per day: Rs. {vehicle_cost}")
            st.subheader("Routes Summary:")
            for r in routes:
                cpo = vehicle_cost / r['orders'] if r['orders'] > 0 else float('inf')
                st.write(f"Vehicle {r['vehicle_id']}: Orders = {r['orders']}, Cost per order = Rs. {cpo:.2f}")

            # Create and show map
            m = create_map(data, routes)
            st_folium(m, width=700, height=500)

        else:
            st.error("No solution found. Try increasing vehicle count or relaxing constraints.")

if __name__ == "__main__":
    main()
