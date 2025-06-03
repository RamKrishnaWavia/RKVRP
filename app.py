import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
from sklearn.cluster import KMeans
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import math

st.set_page_config(layout="wide")
st.title("Milk Delivery Route Optimization (CPO < ₹4)")

TEMPLATE_COLUMNS = ['Society ID', 'Society Name', 'City', 'Drop Point', 'Latitude', 'Longitude', 'Orders']

# Constants
MAX_ORDERS_PER_ROUTE = 200
DEFAULT_VEHICLE_COST_PER_MONTH = 35000
DAYS_PER_MONTH = 30
DEPOT_NAME = "Soukya Road"
DEPOT_LAT, DEPOT_LON = 12.996707, 77.818240  # Replace with your actual depot location

# Haversine distance

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# Upload CSV
st.sidebar.header("Upload Society Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

st.sidebar.download_button("Download Template CSV", pd.DataFrame(columns=TEMPLATE_COLUMNS).to_csv(index=False), file_name="template.csv")

vehicle_cost = st.sidebar.number_input("Vehicle Cost per Month (₹)", value=DEFAULT_VEHICLE_COST_PER_MONTH, step=1000)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    missing_cols = [col for col in TEMPLATE_COLUMNS if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        st.stop()

    # Convert source to first row
    depot_df = pd.DataFrame([{ 'Society ID': 'DEPOT', 'Society Name': DEPOT_NAME, 'City': '', 'Drop Point': 'Depot',
                              'Latitude': DEPOT_LAT, 'Longitude': DEPOT_LON, 'Orders': 0 }])
    df = pd.concat([depot_df, df], ignore_index=True)

    # KMeans Clustering based on Orders per route (MAX 200)
    total_orders = df['Orders'].sum()
    num_clusters = math.ceil(total_orders / MAX_ORDERS_PER_ROUTE)
    n_samples = df.shape[0]
    num_clusters = min(num_clusters, n_samples)

    if n_samples == 0:
        st.error("No data available to cluster.")
        st.stop()

    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(df[['Latitude', 'Longitude']])
    df['Cluster ID'] = kmeans.labels_

    # Display filters
    with st.expander("🔍 Filter Options"):
        selected_city = st.selectbox("Filter by City", ['All'] + sorted(df['City'].dropna().unique().tolist()))
        selected_drop = st.selectbox("Filter by Drop Point", ['All'] + sorted(df['Drop Point'].dropna().unique().tolist()))
        selected_society = st.selectbox("Filter by Society Name", ['All'] + sorted(df['Society Name'].dropna().unique().tolist()))

        filtered_df = df.copy()
        if selected_city != 'All':
            filtered_df = filtered_df[filtered_df['City'] == selected_city]
        if selected_drop != 'All':
            filtered_df = filtered_df[filtered_df['Drop Point'] == selected_drop]
        if selected_society != 'All':
            filtered_df = filtered_df[filtered_df['Society Name'] == selected_society]

    def compute_distance_matrix(locations):
        size = len(locations)
        matrix = {}
        for from_idx in range(size):
            matrix[from_idx] = {}
            for to_idx in range(size):
                if from_idx == to_idx:
                    matrix[from_idx][to_idx] = 0
                else:
                    from_node = locations[from_idx]
                    to_node = locations[to_idx]
                    matrix[from_idx][to_idx] = int(haversine(from_node[0], from_node[1], to_node[0], to_node[1]) * 1000)  # in meters
        return matrix

    def optimize_cluster_routes(df_cluster):
        coords = df_cluster[['Latitude', 'Longitude']].values.tolist()
        demands = df_cluster['Orders'].tolist()
        names = df_cluster['Society Name'].tolist()

        distance_matrix = compute_distance_matrix(coords)

        manager = pywrapcp.RoutingIndexManager(len(coords), 1, 0)  # one vehicle per cluster
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return distance_matrix[from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        def demand_callback(from_index):
            return demands[manager.IndexToNode(from_index)]

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(demand_callback_index, 0, [MAX_ORDERS_PER_ROUTE], True, "Capacity")

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

        solution = routing.SolveWithParameters(search_parameters)

        if not solution:
            return [], 0

        route = []
        index = routing.Start(0)
        route_distance = 0
        seq = 1
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route.append((seq, names[node_index], coords[node_index]))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
            seq += 1
        node_index = manager.IndexToNode(index)
        route.append((seq, names[node_index], coords[node_index]))
        return route, route_distance / 1000

    # Plot map + Optimize each cluster
    route_summary = []
    route_map = folium.Map(location=[DEPOT_LAT, DEPOT_LON], zoom_start=11)

    for cluster_id in sorted(df['Cluster ID'].unique()):
        cluster_df = df[df['Cluster ID'] == cluster_id].reset_index(drop=True)
        route, dist_km = optimize_cluster_routes(cluster_df)
        total_orders = cluster_df['Orders'].sum()
        cpo = (vehicle_cost / DAYS_PER_MONTH) / total_orders if total_orders else 0

        route_summary.append({
            "Cluster ID": cluster_id,
            "Total Orders": total_orders,
            "Route Distance (km)": round(dist_km, 2),
            "Cost Per Order (₹)": round(cpo, 2)
        })

        # Add markers and lines
        points = []
        for seq, name, coord in route:
            popup_text = f"{seq}. {name}<br>Cluster: {cluster_id}<br>CPO: ₹{round(cpo,2)}"
            folium.Marker(location=coord, popup=popup_text, tooltip=popup_text).add_to(route_map)
            points.append(coord)

        if len(points) > 1:
            folium.PolyLine(points, color="blue", weight=2.5, opacity=1).add_to(route_map)

    st.subheader("🗺️ Optimized Delivery Routes")
    folium_static(route_map, width=1000)

    st.subheader("📊 Route Summary")
    summary_df = pd.DataFrame(route_summary)
    st.dataframe(summary_df)

    st.download_button("Download Route Summary", summary_df.to_csv(index=False), file_name="route_summary.csv")
else:
    st.info("Please upload a valid CSV file using the provided template.")
