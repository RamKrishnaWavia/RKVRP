import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from haversine import haversine
import folium
from streamlit_folium import st_folium

# --------- Constants ---------
VEHICLE_CAPACITY = 200  # max orders per vehicle (used in green cluster logic)
GREEN_MAX_RADIUS_KM = 2

# --------- Helper functions ---------

def haversine_distance_matrix(locations):
    n = len(locations)
    matrix = {}
    for from_node in range(n):
        matrix[from_node] = {}
        for to_node in range(n):
            if from_node == to_node:
                matrix[from_node][to_node] = 0
            else:
                matrix[from_node][to_node] = int(
                    haversine(locations[from_node], locations[to_node]) * 1000
                )  # in meters as integer
    return matrix

def optimize_route(locations):
    """Solve TSP to get shortest route for given locations list [(lat, lon), ...]"""
    tsp_size = len(locations)
    if tsp_size <= 1:
        return list(range(tsp_size)), 0.0

    dist_matrix = haversine_distance_matrix(locations)

    # Create routing model
    routing = pywrapcp.RoutingModel(tsp_size, 1, 0)  # 1 vehicle, depot=0
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    def distance_callback(from_index, to_index):
        return dist_matrix[from_index][to_index]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Solve
    solution = routing.SolveWithParameters(search_parameters)
    if not solution:
        return None, 0.0

    # Get route
    index = routing.Start(0)
    route = []
    route_distance = 0
    while not routing.IsEnd(index):
        route.append(routing.IndexToNode(index))
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
    route.append(routing.IndexToNode(index))

    # route_distance in meters, convert to km
    return route, route_distance / 1000

def form_green_clusters(df, vehicle_capacity=VEHICLE_CAPACITY, max_radius_km=GREEN_MAX_RADIUS_KM):
    unassigned = df.copy()
    green_clusters = []
    cluster_id = 0

    while not unassigned.empty:
        # Start cluster with society having max orders
        seed_idx = unassigned['Orders'].idxmax()
        seed_row = unassigned.loc[seed_idx]
        center = (seed_row['Latitude'], seed_row['Longitude'])

        # Calculate distance of all societies from center
        unassigned['Distance'] = unassigned.apply(
            lambda row: haversine(center, (row['Latitude'], row['Longitude'])), axis=1
        )

        # Filter societies within 2 km radius
        nearby = unassigned[unassigned['Distance'] <= max_radius_km]

        # Sort nearby by orders descending
        nearby = nearby.sort_values(by='Orders', ascending=False)

        # Accumulate orders up to vehicle capacity
        cumulative_orders = 0
        cluster_members = []

        for idx, row in nearby.iterrows():
            if cumulative_orders + row['Orders'] <= vehicle_capacity:
                cumulative_orders += row['Orders']
                cluster_members.append(idx)
            else:
                break

        # If total orders in cluster < 200, do NOT form green cluster, break
        if cumulative_orders < 200:
            break

        cluster_df = unassigned.loc[cluster_members].copy()
        cluster_df['Cluster Type'] = 'Green'
        cluster_df['Cluster ID'] = f'G{cluster_id}'
        green_clusters.append(cluster_df)

        # Remove clustered societies from unassigned
        unassigned = unassigned.drop(cluster_members)
        cluster_id += 1

    return green_clusters, unassigned

def form_blue_clusters(df):
    # All leftover societies form a single blue cluster
    df = df.copy()
    df['Cluster Type'] = 'Blue'
    df['Cluster ID'] = 'B0'
    return [df]

def create_map(cluster_df):
    # Center map around average lat/lon
    center_lat = cluster_df['Latitude'].mean()
    center_lon = cluster_df['Longitude'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

    for _, row in cluster_df.iterrows():
        folium.CircleMarker(
            location=(row['Latitude'], row['Longitude']),
            radius=5,
            popup=(
                f"{row['Society Name']}<br>"
                f"Orders: {row['Orders']}<br>"
                f"City: {row['City']}"
            ),
            color='green' if row['Cluster Type'] == 'Green' else 'blue',
            fill=True,
            fill_opacity=0.7,
        ).add_to(m)
    return m

def validate_and_clean(df):
    required_cols = ['Society ID', 'Society Name', 'City', 'Drop Point', 'Latitude', 'Longitude', 'Orders']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        return None

    # Remove rows with missing or invalid coordinates
    df = df.dropna(subset=['Latitude', 'Longitude', 'Orders'])
    df['Orders'] = pd.to_numeric(df['Orders'], errors='coerce')

    invalid_orders = df[df['Orders'].isna()]
    if not invalid_orders.empty:
        st.warning(f"Found rows with invalid Orders values. They will be removed.")
        st.dataframe(invalid_orders)
        df = df.drop(invalid_orders.index)

    df['Orders'] = df['Orders'].astype(int)
    return df

# -------- Streamlit UI --------

st.title("Milk Delivery Route Optimization with Green & Blue Clusters")

st.markdown("""
Upload CSV with columns:
- Society ID (unique)
- Society Name
- City
- Drop Point
- Latitude (decimal degrees)
- Longitude (decimal degrees)
- Orders (integer)
""")

csv_template = pd.DataFrame({
    'Society ID': ['S1', 'S2'],
    'Society Name': ['Society A', 'Society B'],
    'City': ['CityX', 'CityX'],
    'Drop Point': ['DP1', 'DP2'],
    'Latitude': [12.9716, 12.9750],
    'Longitude': [77.5946, 77.6000],
    'Orders': [150, 80],
})
st.download_button("Download CSV Template", csv_template.to_csv(index=False), "milk_delivery_template.csv")

uploaded_file = st.file_uploader("Upload your data CSV", type=['csv'])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df_clean = validate_and_clean(df)
    if df_clean is not None:
        st.success(f"Data loaded: {len(df_clean)} valid rows.")

        green_clusters, remaining = form_green_clusters(df_clean)

        blue_clusters = []
        if not remaining.empty:
            blue_clusters = form_blue_clusters(remaining)

        all_clusters = green_clusters + blue_clusters

        total_orders = 0
        total_distance = 0.0
        total_cost = 0.0

        st.header("Cluster-wise Routes and Summary")

        for cluster_df in all_clusters:
            cluster_id = cluster_df['Cluster ID'].iloc[0]
            cluster_type = cluster_df['Cluster Type'].iloc[0]
            cluster_orders = cluster_df['Orders'].sum()

            # Optimize route for cluster
            locations = cluster_df[['Latitude', 'Longitude']].values.tolist()
            route, route_distance = optimize_route(locations)
            if route is None:
                st.warning(f"Could not optimize route for Cluster {cluster_id}")
                continue

            cost_per_order = (route_distance * 35) / cluster_orders if cluster_orders > 0 else 0  # ₹35/km cost assumed

            total_orders += cluster_orders
            total_distance += route_distance
            total_cost += cost_per_order * cluster_orders

            st.subheader(f"Cluster {cluster_id} ({cluster_type})")
            st.write(f"Total Orders: {cluster_orders}")
            st.write(f"Route Distance: {route_distance:.2f} km")
            st.write(f"Cost per Order (CPO): ₹{cost_per_order:.2f}")

            # Show map with societies
            fol_map = create_map(cluster_df)
            st_folium(fol_map, width=700, height=450)

        if total_orders > 0:
            st.header("Overall Summary")
            st.write(f"Total Orders: {total_orders}")
            st.write(f"Total Distance: {total_distance:.2f} km")
            st.write(f"Average Cost per Order: ₹{total_cost / total_orders:.2f}")
