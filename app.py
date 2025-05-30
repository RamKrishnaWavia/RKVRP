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
st.markdown("Optimize early morning deliveries (4 AM to 7 AM) to apartments/societies with minimum cost.")

# CSV Template Download
if st.button("📅 Download CSV Template"):
    template = pd.DataFrame({
        "Society ID": [101, 102],
        "Apartment": ["ABC Residency", "Green Heights"],
        "Latitude": [12.935, 12.938],
        "Longitude": [77.614, 77.610],
        "Orders": [10, 20]
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

# Vehicle constraints input
num_vehicles = st.number_input("Enter number of vehicles", min_value=1, value=1)
vehicle_capacity = st.number_input("Enter max order capacity per vehicle", min_value=1, value=200)
vehicle_cost = st.number_input("Enter cost per vehicle (₹)", min_value=0, value=500)

# File Upload
uploaded_file = st.file_uploader("Upload Delivery Data CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors='coerce')
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors='coerce')
    df = df.dropna(subset=["Latitude", "Longitude"])

    if df.empty:
        st.error("Latitude or Longitude values missing in the data. Please correct and upload again.")
        st.stop()

    st.success("📍 Locations loaded and validated successfully.")
    st.dataframe(df)

    # OR-Tools CVRP setup with vehicle capacity
    def create_data_model():
        data = {}
        data['locations'] = list(zip(df["Latitude"], df["Longitude"]))
        data['num_vehicles'] = num_vehicles
        data['depot'] = 0
        data['demands'] = [0] + df["Orders"].tolist()[1:]
        data['vehicle_capacities'] = [vehicle_capacity] * num_vehicles
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

    manager = pywrapcp.RoutingIndexManager(len(data['locations']), data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(distance_matrix[from_node][to_node] * 100000)

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add capacity constraints
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,
        data['vehicle_capacities'],
        True,
        'Capacity'
    )

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        st.subheader("📋 Optimized Delivery Routes")
        total_km = 0
        total_orders = df["Orders"].sum()
        route_count = 0

        for vehicle_id in range(num_vehicles):
            index = routing.Start(vehicle_id)
            route = []
            route_distance = 0
            vehicle_orders = 0

            if routing.IsEnd(solution.Value(routing.NextVar(index))):
                continue  # skip unused vehicles

            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route.append(node_index)
                next_index = solution.Value(routing.NextVar(index))
                route_distance += distance_matrix[node_index][manager.IndexToNode(next_index)]
                vehicle_orders += df.iloc[node_index]['Orders'] if node_index < len(df) else 0
                index = next_index
            route.append(manager.IndexToNode(index))
            total_km += route_distance * 111  # convert degrees to km
            route_count += 1

            m = folium.Map(location=[df["Latitude"].mean(), df["Longitude"].mean()], zoom_start=13)
            for i, stop in enumerate(route):
                if stop >= len(df):
                    continue
                row = df.iloc[stop]
                folium.Marker(
                    location=[row["Latitude"], row["Longitude"]],
                    popup=f"{i+1}. {row['Apartment']} (ID: {row['Society ID']}, {row['Orders']} orders)",
                    icon=folium.Icon(color='blue' if i > 0 else 'green', icon='home' if i > 0 else 'play')
                ).add_to(m)
            folium.PolyLine(
                locations=[(df.iloc[stop]["Latitude"], df.iloc[stop]["Longitude"]) for stop in route if stop < len(df)],
                color="red", weight=3
            ).add_to(m)
            st.markdown(f"### Vehicle {vehicle_id+1} Route (Orders: {vehicle_orders})")
            st_folium(m, height=400, width=800)

        total_cost = route_count * vehicle_cost
        cost_per_order = total_cost / total_orders if total_orders > 0 else 0

        st.subheader("💰 Cost Summary")
        st.markdown(f"- **Total distance:** {total_km:.2f} km")
        st.markdown(f"- **Total vehicles used:** {route_count}")
        st.markdown(f"- **Total cost:** ₹{total_cost:.2f}")
        st.markdown(f"- **Cost per order:** ₹{cost_per_order:.2f}")

    else:
        st.error("Optimization failed. Please check the data and try again.")
