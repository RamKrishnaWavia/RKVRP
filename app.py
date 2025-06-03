import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import math

# Constants
VEHICLE_CAPACITY = 200  # Orders per vehicle per route
DEFAULT_VEHICLE_COST = 35000  # INR per month per vehicle
WORKING_DAYS = 30
DEPOT_NAME = "Soukya Road"

# Streamlit app UI
st.set_page_config(page_title="Milk Delivery Route Optimizer", layout="wide")
st.title("🚚 RK - Delivery Route Optimization")
st.markdown("**Objective**: Reduce Cost per Order (CPO) using vehicle capacity (200 orders), route optimization, and depot logic.")

# Template CSV download
with st.expander("📥 Download CSV Template"):
    sample = pd.DataFrame({
        'Society ID': ['S1', 'S2', 'S3'],
        'Society Name': ['Heaven Apartments', 'Green Enclave', 'Ocean View'],
        'City': ['Bangalore', 'Bangalore', 'Bangalore'],
        'Drop Point': ['Soukya Road', 'Soukya Road', 'Soukya Road'],
        'Latitude': [12.9456, 12.9478, 12.9500],
        'Longitude': [77.7501, 77.7515, 77.7525],
        'Orders': [100, 120, 150]
    })
    csv = sample.to_csv(index=False).encode("utf-8")
    st.download_button("📄 Download Template", csv, "milk_routes_template.csv", "text/csv")

# File uploader
uploaded_file = st.file_uploader("Upload delivery data (CSV with required columns)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    required_columns = ['Society ID', 'Society Name', 'City', 'Drop Point', 'Latitude', 'Longitude', 'Orders']
    missing = set(required_columns) - set(df.columns)

    if missing:
        st.error(f"Missing required columns: {list(missing)}")
        st.stop()

    df['Latitude'] = df['Latitude'].astype(float)
    df['Longitude'] = df['Longitude'].astype(float)

# Ensure 'Orders' column is numeric and handle non-numeric entries
df['Orders'] = pd.to_numeric(df['Orders'], errors='coerce')
if df['Orders'].isnull().any():
    st.error("⚠️ Some 'Orders' values are missing or non-numeric. Please check your CSV.")
    st.stop()
df['Orders'] = df['Orders'].astype(int)

    # Vehicle cost input
    vehicle_cost = st.number_input("💰 Vehicle Monthly Cost (INR)", min_value=1000, value=DEFAULT_VEHICLE_COST)

    # Compute required clusters dynamically
    total_orders = df['Orders'].sum()
    num_clusters = math.ceil(total_orders / VEHICLE_CAPACITY)
    st.write(f"🚗 Total Orders: {total_orders}, Required Clusters (Routes): {num_clusters}")

    # Run KMeans clustering
    coords = df[['Latitude', 'Longitude']]
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(coords)
    df['Cluster ID'] = kmeans.labels_

    # Depot coordinates (Soukya Road - fixed)
    depot_lat, depot_lng = 12.9456, 77.7501
    depot = (depot_lat, depot_lng)

    # Helper: compute distance matrix
    def compute_distance_matrix(locations):
        matrix = {}
        for from_idx, from_node in enumerate(locations):
            matrix[from_idx] = {}
            for to_idx, to_node in enumerate(locations):
                matrix[from_idx][to_idx] = math.dist(from_node, to_node)
        return matrix

    # Route optimizer
    def optimize_cluster(df_cluster):
        # Include depot at index 0
        locations = [(depot_lat, depot_lng)] + list(zip(df_cluster['Latitude'], df_cluster['Longitude']))
        distance_matrix = compute_distance_matrix(locations)
        order_demands = [0] + df_cluster['Orders'].tolist()

        manager = pywrapcp.RoutingIndexManager(len(distance_matrix), 1, 0)
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(distance_matrix[from_node][to_node] * 1000)  # Convert to meters

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        def demand_callback(from_index):
            return order_demands[manager.IndexToNode(from_index)]

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            [VEHICLE_CAPACITY],
            True,
            'Capacity'
        )

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )

        solution = routing.SolveWithParameters(search_parameters)

        if solution:
            index = routing.Start(0)
            route, dist_m = [], 0
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route.append(node_index)
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                dist_m += routing.GetArcCostForVehicle(previous_index, index, 0)
            route.append(manager.IndexToNode(index))  # End

            return [locations[i] for i in route], dist_m / 1000  # Convert meters to km
        else:
            return [], 0

    # Run optimization for each cluster
    results = []
    for cid in df['Cluster ID'].unique():
        df_cluster = df[df['Cluster ID'] == cid].reset_index(drop=True)
        route_coords, dist_km = optimize_cluster(df_cluster)
        total_orders = df_cluster['Orders'].sum()
        cost_per_order = round((vehicle_cost / WORKING_DAYS) / total_orders, 2) if total_orders else 0
        results.append({
            'Cluster ID': cid,
            'Total Orders': total_orders,
            'Distance (km)': round(dist_km, 2),
            'CPO (INR)': cost_per_order,
            'Route': route_coords
        })

    # Show results
    st.subheader("📊 Route Summary")
    result_df = pd.DataFrame(results)
    st.dataframe(result_df)

    # Download results
    csv_data = result_df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇ Download Route Summary CSV", csv_data, "route_summary.csv", "text/csv")

    # Map
    st.subheader("🗺️ Delivery Route Map")
    map_center = (df['Latitude'].mean(), df['Longitude'].mean())
    route_map = folium.Map(location=map_center, zoom_start=13)
    marker_cluster = MarkerCluster().add_to(route_map)

    for i, row in df.iterrows():
        popup = f"{row['Society Name']} (Orders: {row['Orders']})<br>Cluster: {row['Cluster ID']}"
        folium.Marker(
            location=(row['Latitude'], row['Longitude']),
            popup=popup,
            icon=folium.Icon(color="blue", icon="info-sign")
        ).add_to(marker_cluster)

    # Show CPO in popup route lines
    for res in results:
        folium.PolyLine(
            locations=res['Route'],
            tooltip=f"Cluster {res['Cluster ID']} | CPO: ₹{res['CPO (INR)']}",
            color="green",
            weight=3
        ).add_to(route_map)

    st_folium(route_map, width=1000, height=600)
