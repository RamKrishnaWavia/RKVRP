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

# Inputs
num_vehicles = st.number_input("Enter number of vehicles available", min_value=1, value=1)
vehicle_capacity = st.number_input("Maximum order capacity per vehicle", min_value=1, value=200)
vehicle_cost = st.number_input("Cost per km per vehicle (₹)", min_value=1, value=12)

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

    locations = list(zip(df["Latitude"], df["Longitude"]))
    demands = df["Orders"].tolist()

    def compute_euclidean_distance_matrix(locations):
        distances = {}
        for from_counter, from_node in enumerate(locations):
            distances[from_counter] = {}
            for to_counter, to_node in enumerate(locations):
                distances[from_counter][to_counter] = ((from_node[0] - to_node[0])**2 + (from_node[1] - to_node[1])**2)**0.5
        return distances

    distance_matrix = compute_euclidean_distance_matrix(locations)

    manager = pywrapcp.RoutingIndexManager(len(locations), num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(distance_matrix[from_node][to_node] * 100000)

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return demands[from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,
        [vehicle_capacity] * num_vehicles,
        True,
        'Capacity')

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        route_data = []
        for vehicle_id in range(num_vehicles):
            index = routing.Start(vehicle_id)
            route = []
            route_orders = 0
            total_distance = 0
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route.append(node_index)
                route_orders += demands[node_index]
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                total_distance += distance_matrix[manager.IndexToNode(previous_index)][manager.IndexToNode(index)]
            route.append(manager.IndexToNode(index))
            if len(route) > 1:
                route_data.append({
                    "vehicle_id": vehicle_id + 1,
                    "route": route,
                    "orders": route_orders,
                    "distance": total_distance * 111  # degrees to km
                })

        m = folium.Map(location=[df["Latitude"].mean(), df["Longitude"].mean()], zoom_start=13)

        st.subheader("📋 Vehicle-wise Optimized Routes")
        for r in route_data:
            st.markdown(f"### 🚚 Vehicle {r['vehicle_id']} - Orders: {r['orders']} - Distance: {r['distance']:.2f} km")
            ordered_df = df.iloc[r['route']].reset_index(drop=True)
            ordered_df["Stop"] = range(1, len(ordered_df)+1)
            st.dataframe(ordered_df[["Stop", "Society ID", "Apartment", "Latitude", "Longitude", "Orders"]])

            for i, stop in enumerate(r['route']):
                row = df.iloc[stop]
                folium.Marker(
                    location=[row["Latitude"], row["Longitude"]],
                    popup=f"{i+1}. {row['Apartment']} (ID: {row['Society ID']}, {row['Orders']} orders)",
                    icon=folium.Icon(color='blue' if i > 0 else 'green', icon='home' if i > 0 else 'play')
                ).add_to(m)

            folium.PolyLine(
                locations=[(df.iloc[stop]["Latitude"], df.iloc[stop]["Longitude"]) for stop in r['route']],
                color="red", weight=3
            ).add_to(m)

        st_folium(m, height=500, width=900)

        total_cost = sum(r['distance'] * vehicle_cost for r in route_data)
        total_orders = df["Orders"].sum()
        cost_per_order = total_cost / total_orders if total_orders else 0

        st.subheader("💰 Cost Summary")
        st.markdown(f"- **Total vehicles used:** {len(route_data)}")
        st.markdown(f"- **Total orders delivered:** {total_orders}")
        st.markdown(f"- **Total distance:** {sum(r['distance'] for r in route_data):.2f} km")
        st.markdown(f"- **Total cost:** ₹{total_cost:.2f}")
        st.markdown(f"- **Cost per order:** ₹{cost_per_order:.2f}")

        if st.button("📄 Download All Vehicle Routes CSV"):
            export_rows = []
            for r in route_data:
                route_df = df.iloc[r['route']].reset_index(drop=True)
                route_df["Vehicle"] = r['vehicle_id']
                route_df["Stop"] = range(1, len(route_df)+1)
                export_rows.append(route_df)
            full_export = pd.concat(export_rows, ignore_index=True)
            csv_export = full_export.to_csv(index=False).encode("utf-8")
            st.download_button("Download Routes CSV", data=csv_export, file_name="all_routes_optimized.csv", mime="text/csv")
    else:
        st.error("Optimization failed. Please check the data.")
