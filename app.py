import streamlit as st
import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import folium
from streamlit_folium import st_folium
import io

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
                time_matrix[i][j] = (dist / AVERAGE_SPEED_KMPH) * 3600  # seconds
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
        route.append(manager.IndexToNode(index))
        total_distance += route_dist
        total_load += route_load
        output.append({
            "vehicle_id": vehicle_id + 1,
            "route": route,
            "orders_delivered": route_load,
            "distance_km": route_dist / 1000.0,
        })
    return output

def create_route_map(routes, locations):
    # Center map around depot
    depot_lat, depot_lon = locations[0]
    m = folium.Map(location=[depot_lat, depot_lon], zoom_start=12)

    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightblue', 'beige',
              'darkgreen', 'cadetblue', 'darkpurple', 'pink', 'lightgreen', 'gray', 'black']

    for r in routes:
        vehicle_color = colors[(r['vehicle_id'] - 1) % len(colors)]
        coords = [locations[node] for node in r['route']]
        folium.PolyLine(coords, color=vehicle_color, weight=5, opacity=0.7).add_to(m)
        # Markers with popup orders
        for idx, node in enumerate(r['route']):
            lat, lon = locations[node]
            popup = f"Vehicle {r['vehicle_id']}<br>Stop: {node}<br>Orders: N/A"
            folium.CircleMarker(location=(lat, lon), radius=5, color=vehicle_color, fill=True).add_to(m)

    # Depot marker
    folium.Marker(location=(depot_lat, depot_lon), popup="Depot", icon=folium.Icon(color="black", icon="home")).add_to(m)
    return m

def create_csv_download(routes, df, vehicle_cost):
    # Build CSV rows: vehicle_id, stop_index, lat, lon, orders
    rows = []
    locations = [(st.session_state.depot_lat, st.session_state.depot_lon)] + list(zip(df['latitude'], df['longitude']))
    orders_list = [0] + df['orders'].tolist()

    for r in routes:
        total_orders = r['orders_delivered']
        cost_per_order = vehicle_cost / total_orders if total_orders > 0 else 0
        for idx, node in enumerate(r['route']):
            lat, lon = locations[node]
            order_count = orders_list[node]
            rows.append({
                'vehicle_id': r['vehicle_id'],
                'stop_sequence': idx+1,
                'node_index': node,
                'latitude': lat,
                'longitude': lon,
                'orders': order_count,
                'cost_per_order_vehicle': cost_per_order
            })
    csv_df = pd.DataFrame(rows)
    return csv_df

def main():
    st.title("Milk Delivery Route Optimizer with Maps and Cost")

    st.markdown("""
    Upload CSV with columns: `latitude`, `longitude`, `orders`  
    Vehicle capacity fixed at 20 orders  
    Delivery window: 3:30 AM to 7:00 AM  
    """)
    uploaded_file = st.file_uploader("Upload Delivery Points CSV", type=["csv"])

    depot_lat = st.number_input("Depot Latitude", value=12.9716, key='depot_lat')  # Bangalore default
    depot_lon = st.number_input("Depot Longitude", value=77.5946, key='depot_lon')
    vehicle_cost = st.number_input("Monthly Vehicle Cost (₹)", min_value=0, value=35000, step=1000)

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())

        if not {'latitude', 'longitude', 'orders'}.issubset(df.columns):
            st.error("CSV must contain 'latitude', 'longitude' and 'orders' columns")
            return

        locations = [(depot_lat, depot_lon)] + list(zip(df['latitude'], df['longitude']))
        orders = [0] + df['orders'].tolist()

        dist_matrix_km, time_matrix_sec = create_distance_time_matrix(locations)

        dist_matrix = (dist_matrix_km * 1000).astype(int)
        time_matrix = time_matrix_sec.astype(int)

        vehicle_count = (sum(orders) // VEHICLE_CAPACITY) + 1

        manager = pywrapcp.RoutingIndexManager(len(dist_matrix), vehicle_count, 0)
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            return dist_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        def demand_callback(from_index):
            return orders[manager.IndexToNode(from_index)]

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,
            [VEHICLE_CAPACITY] * vehicle_count,
            True,
            'Capacity'
        )

        def time_callback(from_index, to_index):
            return time_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

        time_callback_index = routing.RegisterTransitCallback(time_callback)
        routing.AddDimension(
            time_callback_index,
            30*60,
            TIME_WINDOW_END - TIME_WINDOW_START,
            False,
            'Time'
        )

        time_dimension = routing.GetDimensionOrDie('Time')
        time_dimension.CumulVar(routing.Start(0)).SetRange(TIME_WINDOW_START, TIME_WINDOW_END)
        time_dimension.CumulVar(routing.End(0)).SetRange(TIME_WINDOW_START, TIME_WINDOW_END)

        for node in range(1, len(locations)):
            time_dimension.CumulVar(manager.NodeToIndex(node)).SetRange(TIME_WINDOW_START, TIME_WINDOW_END)

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

        solution = routing.SolveWithParameters(search_parameters)

        if solution:
            routes = print_solution(manager, routing, solution, orders, vehicle_count)
            st.success(f"Solution found with {vehicle_count} vehicles")

            # Show routes text
            for r in routes:
                st.markdown(f"### Vehicle {r['vehicle_id']} route:")
                route_str = " -> ".join(str(node) for node in r['route'])
                st.write(route_str)
                st.write(f"Orders Delivered: {r['orders_delivered']}")
                st.write(f"Route Distance (approx): {r['distance_km']:.2f} km")
                cost_per_order = vehicle_cost / r['orders_delivered'] if r['orders_delivered'] > 0 else 0
                st.write(f"Cost Per Order (Vehicle Cost ₹{vehicle_cost}): ₹{cost_per_order:.2f}")
                st.write("---")

            # Show Folium Map
            st.markdown("## Route Map")
            route_map = create_route_map(routes, locations)
            st_data = st_folium(route_map, width=700, height=500)

            # CSV Download
            csv_df = create_csv_download(routes, df, vehicle_cost)
            csv_buffer = io.StringIO()
            csv_df.to_csv(csv_buffer, index=False)
            csv_str = csv_buffer.getvalue()
            st.download_button("Download Routes CSV", data=csv_str, file_name="milk_delivery_routes.csv", mime="text/csv")

        else:
            st.error("No solution found.")

if __name__ == '__main__':
    main()
