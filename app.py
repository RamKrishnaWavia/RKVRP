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

    # OR-Tools VRP setup
    def create_data_model():
        data = {
            'locations': list(zip(df["Latitude"], df["Longitude"])),
            'num_vehicles': 1,
            'depot': 0
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

    manager = pywrapcp.RoutingIndexManager(len(data['locations']), data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(distance_matrix[from_node][to_node] * 100000)
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        index = routing.Start(0)
        route = []
        total_distance = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route.append(node_index)
            next_index = solution.Value(routing.NextVar(index))
            total_distance += distance_matrix[node_index][manager.IndexToNode(next_index)]
            index = next_index
        route.append(manager.IndexToNode(index))

        # Map visualization
        m = folium.Map(location=[df["Latitude"].mean(), df["Longitude"].mean()], zoom_start=13)
        for i, stop in enumerate(route):
            row = df.iloc[stop]
            society_id = row.get("Society ID", "N/A")
            orders = row.get("Orders", "N/A")
            folium.Marker(
                location=[row["Latitude"], row["Longitude"]],
                popup=f"{i+1}. {row['Apartment']} (ID: {society_id}, {orders} orders)",
                icon=folium.Icon(color='blue' if i > 0 else 'green', icon='home' if i > 0 else 'play')
            ).add_to(m)
        folium.PolyLine(
            locations=[(df.iloc[stop]["Latitude"], df.iloc[stop]["Longitude"]) for stop in route],
            color="red", weight=3
        ).add_to(m)
        st_folium(m, height=500, width=900)

        # Route summary
        ordered_df = df.iloc[route].reset_index(drop=True)
        ordered_df["Stop"] = range(1, len(ordered_df)+1)
        st.subheader("📋 Optimized Delivery Route")
        st.dataframe(ordered_df[["Stop", "Society ID", "Apartment", "Latitude", "Longitude", "Orders"]])

        # Cost calculation
        total_orders = df["Orders"].sum()
        cost_per_km = 12  # example fixed rate
        total_km = total_distance * 111  # rough conversion: 1 degree ≈ 111 km
        total_cost = total_km * cost_per_km
        cost_per_order = total_cost / total_orders

        st.subheader("💰 Cost Summary")
        st.markdown(f"- **Total distance:** {total_km:.2f} km")
        st.markdown(f"- **Total cost:** ₹{total_cost:.2f}")
        st.markdown(f"- **Cost per order:** ₹{cost_per_order:.2f}")

        # Download optimized CSV
        if st.button("📄 Download Optimized Route CSV"):
            csv_export = ordered_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Route CSV", data=csv_export, file_name="optimized_milk_route.csv", mime="text/csv")
    else:
        st.error("Optimization failed. Please check the data.")
