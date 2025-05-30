import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from geopy.geocoders import Nominatim
import io
import time

# Title and Description
st.set_page_config(page_title="Milk Delivery Optimizer", layout="wide")
st.title("RK - Delivery Route Optimizer")
st.markdown("Optimize early morning deliveries (4 AM to 6:30 AM) to apartments/societies with minimum cost.")

# CSV Template Download
if st.button("📅 Download CSV Template"):
    template = pd.DataFrame({
        "Society ID": [101, 102],
        "Apartment": ["ABC Residency", "Green Heights"],
        "Latitude": [12.935, 12.938],
        "Longitude": [77.614, 77.610],
        "Orders": [10, 20],
        "Start Time": [0, 0],
        "End Time": [150, 150]
    })
    csv = template.to_csv(index=False).encode('utf-8')
    st.download_button("Download Template", data=csv, file_name="milk_delivery_template.csv", mime='text/csv')

# Address Geocoding Helper
def geocode_address(address):
    geolocator = Nominatim(user_agent="milk_optimizer")
    try:
        location = geolocator.geocode(address, timeout=10)
        return (location.latitude, location.longitude) if location else (None, None)
    except:
        return (None, None)

# User Inputs
num_vehicles = st.number_input("Number of Vehicles", min_value=1, value=1, step=1)
vehicle_capacity = st.number_input("Order Capacity per Vehicle", min_value=1, value=50, step=1)
max_orders_per_route = st.number_input("Max Orders per Route per Vehicle", min_value=1, value=50, step=1)

# File Upload
uploaded_file = st.file_uploader("Upload Delivery Data CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if "Latitude" not in df.columns or "Longitude" not in df.columns:
        with st.spinner("Geocoding addresses..."):
            coords = df["Apartment"].apply(geocode_address)
            df["Latitude"] = coords.apply(lambda x: x[0])
            df["Longitude"] = coords.apply(lambda x: x[1])
        if df["Latitude"].isnull().any():
            st.error("Some addresses could not be geocoded. Please check and try again.")
            st.stop()

    st.success("📍 Locations loaded and geocoded successfully.")
    st.dataframe(df)

    # OR-Tools VRP with Time Windows and Capacity
    def create_data_model():
        data = {
            'locations': list(zip(df["Latitude"], df["Longitude"])),
            'num_vehicles': num_vehicles,
            'depot': 0,
            'demands': df["Orders"].tolist(),
            'vehicle_capacities': [vehicle_capacity] * num_vehicles,
            'time_windows': list(zip(df["Start Time"], df["End Time"]))
        }
        return data

    def compute_euclidean_distance_matrix(locations):
        distances = {}
        for from_counter, from_node in enumerate(locations):
            distances[from_counter] = {}
            for to_counter, to_node in enumerate(locations):
                if from_counter == to_counter:
                    distances[from_counter][to_counter] = 0
                else:
                    distances[from_counter][to_counter] = ((from_node[0] - to_node[0])**2 + (from_node[1] - to_node[1])**2)**0.5
        return distances

    data = create_data_model()
    distance_matrix = compute_euclidean_distance_matrix(data['locations'])
    time_matrix = distance_matrix

    manager = pywrapcp.RoutingIndexManager(len(data['locations']), data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(distance_matrix[from_node][to_node] * 100000)

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Capacity constraints
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index, 0, data['vehicle_capacities'], True, 'Capacity')

    # Time windows
    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(time_matrix[from_node][to_node] * 100)
    time_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.AddDimension(
        time_callback_index, 30, 300, False, 'Time')
    time_dimension = routing.GetDimensionOrDie('Time')

    for location_idx, time_window in enumerate(data['time_windows']):
        index = manager.NodeToIndex(location_idx)
        time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])

    # Search Parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = 10

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        st.subheader("📋 Optimized Delivery Routes")
        total_cost_all = 0
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            plan_output = []
            route_distance = 0
            route_orders = 0
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                plan_output.append(node_index)
                next_index = solution.Value(routing.NextVar(index))
                route_distance += distance_matrix[node_index][manager.IndexToNode(next_index)]
                route_orders += data['demands'][node_index]
                index = next_index
            plan_output.append(manager.IndexToNode(index))

            if route_orders > 0:
                m = folium.Map(location=[df["Latitude"].mean(), df["Longitude"].mean()], zoom_start=13)
                for i, stop in enumerate(plan_output):
                    row = df.iloc[stop]
                    folium.Marker(
                        location=[row["Latitude"], row["Longitude"]],
                        popup=f"{i+1}. {row['Apartment']} (ID: {row['Society ID']}, {row['Orders']} orders)",
                        icon=folium.Icon(color='blue' if i > 0 else 'green', icon='home' if i > 0 else 'play')
                    ).add_to(m)
                folium.PolyLine(
                    locations=[(df.iloc[stop]["Latitude"], df.iloc[stop]["Longitude"]) for stop in plan_output],
                    color="red", weight=3
                ).add_to(m)
                st.markdown(f"### 🛻 Vehicle {vehicle_id + 1} Route")
                st_folium(m, height=400, width=900)

                ordered_df = df.iloc[plan_output].reset_index(drop=True)
                ordered_df["Stop"] = range(1, len(ordered_df)+1)
                st.dataframe(ordered_df[["Stop", "Society ID", "Apartment", "Latitude", "Longitude", "Orders"]])

                total_orders = ordered_df["Orders"].sum()
                cost_per_km = 12
                total_km = route_distance * 111
                total_cost = total_km * cost_per_km
                cost_per_order = total_cost / total_orders if total_orders > 0 else 0
                total_cost_all += total_cost

                st.markdown(f"**Distance:** {total_km:.2f} km | **Orders:** {total_orders} | **Cost:** ₹{total_cost:.2f} | ₹{cost_per_order:.2f}/order")

        st.markdown(f"### 💰 Total Delivery Cost Across All Vehicles: ₹{total_cost_all:.2f}")
    else:
        st.error("Optimization failed. Please check the data and constraints.")
