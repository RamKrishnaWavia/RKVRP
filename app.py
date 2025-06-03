import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from geopy.distance import geodesic
import io

# Constants
VEHICLE_CAPACITY = 20
DEPOT_LAT = 12.9716
DEPOT_LON = 77.5946
MAX_ROUTE_TIME = 210  # 3.5 hours in minutes

# Calculate distance in km between two lat/lon points
def compute_distance_km(loc1, loc2):
    return geodesic(loc1, loc2).km

# Generate distance matrix from locations
def create_distance_matrix(locations):
    matrix = []
    for from_node in locations:
        row = [int(compute_distance_km(from_node, to_node) * 1000) for to_node in locations]
        matrix.append(row)
    return matrix

# Generate time matrix (assuming average speed of 20 km/h)
def create_time_matrix(distance_matrix):
    return [[int(dist / 1000 * 60 / 20) for dist in row] for row in distance_matrix]

# Main Streamlit App
def main():
    st.title("🚚 Milk Delivery Route Optimizer")
    st.markdown("Upload a CSV with delivery points including `latitude`, `longitude`, and `orders`.")

    # Template download
    st.markdown("### 📅 Download CSV Template")
    template_csv = io.StringIO()
    pd.DataFrame(columns=["latitude", "longitude", "orders"]).to_csv(template_csv, index=False)
    st.download_button(
        label="Download Template",
        data=template_csv.getvalue(),
        file_name="delivery_template.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader("Upload Delivery Points CSV", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if not all(col in df.columns for col in ['latitude', 'longitude', 'orders']):
            st.error("CSV must contain columns: latitude, longitude, orders")
            return

        # Repeat points based on number of orders
        repeated_points = df.loc[df.index.repeat(df['orders'])][['latitude', 'longitude']].reset_index(drop=True)
        locations = [(DEPOT_LAT, DEPOT_LON)] + list(zip(repeated_points['latitude'], repeated_points['longitude']))

        distance_matrix = create_distance_matrix(locations)
        time_matrix = create_time_matrix(distance_matrix)

        num_locations = len(locations)
        num_vehicles = int(np.ceil(len(repeated_points) / VEHICLE_CAPACITY))

        manager = pywrapcp.RoutingIndexManager(num_locations, num_vehicles, 0)
        routing = pywrapcp.RoutingModel(manager)

        def time_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return time_matrix[from_node][to_node]

        time_callback_index = routing.RegisterTransitCallback(time_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(time_callback_index)

        routing.AddDimension(
            time_callback_index,
            slack_max=0,
            capacity=MAX_ROUTE_TIME,
            fix_start_cumul_to_zero=True,
            name='Time'
        )

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

        solution = routing.SolveWithParameters(search_parameters)

        if solution:
            st.success("Routes optimized!")
            routes = []
            for vehicle_id in range(num_vehicles):
                index = routing.Start(vehicle_id)
                route = []
                while not routing.IsEnd(index):
                    node_index = manager.IndexToNode(index)
                    if node_index != 0:
                        route.append(locations[node_index])
                    index = solution.Value(routing.NextVar(index))
                routes.append(route)

            for i, route in enumerate(routes):
                st.markdown(f"### Vehicle {i + 1} Route")
                for loc in route:
                    st.write(f"Lat: {loc[0]}, Lon: {loc[1]}")
        else:
            st.error("No feasible solution found.")

if __name__ == '__main__':
    main()
