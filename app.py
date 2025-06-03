import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from math import radians, cos, sin, asin, sqrt
import math

st.set_page_config(layout="wide")
st.title("Milk Delivery Route Optimizer")

# Constants
VEHICLE_COST_DEFAULT = 35000
ORDERS_PER_VEHICLE = 200

# Haversine distance calculation
def haversine(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return 6371 * c

# OR-Tools distance callback
def create_distance_matrix(locations):
    size = len(locations)
    dist_matrix = {}
    for from_counter, from_node in enumerate(locations):
        dist_matrix[from_counter] = {}
        for to_counter, to_node in enumerate(locations):
            dist_matrix[from_counter][to_counter] = int(
                haversine(from_node[0], from_node[1], to_node[0], to_node[1]) * 1000
            )
    return dist_matrix

def optimize_route(locations):
    dist_matrix = create_distance_matrix(locations)
    tsp_size = len(dist_matrix)
    num_routes = 1
    depot = 0

    routing = pywrapcp.RoutingModel(tsp_size, num_routes, depot)
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    def distance_callback(from_index, to_index):
        return dist_matrix[from_index][to_index]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        index = routing.Start(0)
        route = []
        while not routing.IsEnd(index):
            route.append(routing.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        route.append(routing.IndexToNode(index))

        distance_km = sum([
            haversine(locations[route[i]][0], locations[route[i]][1],
                      locations[route[i+1]][0], locations[route[i+1]][1])
            for i in range(len(route) - 1)
        ])
        return route, distance_km
    else:
        return [], 0

# Template download
with st.expander("📥 Download CSV Template"):
    template_df = pd.DataFrame({
        'Society ID': ['S001'],
        'Society Name': ['ABC Heights'],
        'City': ['Bangalore'],
        'Drop Point': ['Main Gate'],
        'Latitude': [12.9608],
        'Longitude': [77.6412],
        'Orders': [50]
    })
    st.download_button("Download Template", template_df.to_csv(index=False), "template.csv")

# File uploader
uploaded_file = st.file_uploader("Upload Society Orders CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    required_cols = ['Society ID', 'Society Name', 'City', 'Drop Point', 'Latitude', 'Longitude', 'Orders']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        st.stop()

    # Validate 'Orders' column
    df['Orders_check'] = pd.to_numeric(df['Orders'], errors='coerce')
    invalid_orders_df = df[df['Orders_check'].isna()]

    if not invalid_orders_df.empty:
        st.error("⚠️ Some rows have missing or non-numeric 'Orders' values. Please fix them before proceeding.")
        st.dataframe(invalid_orders_df.drop(columns=['Orders_check']), use_container_width=True)
        st.stop()
    else:
        df['Orders'] = df['Orders_check'].astype(int)
        df.drop(columns=['Orders_check'], inplace=True)

    # Calculate number of clusters
    total_orders = df['Orders'].sum()
    num_clusters = math.ceil(total_orders / ORDERS_PER_VEHICLE)
    st.success(f"Auto-calculated number of clusters: {num_clusters}")

    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(df[['Latitude', 'Longitude']])
    df['Cluster ID'] = kmeans.labels_

    results = []

    for cluster_id, cluster_df in df.groupby('Cluster ID'):
        cluster_df = cluster_df.reset_index(drop=True)

        depot = [12.9840, 77.7490]  # Soukya Road depot
        cluster_locations = [depot] + cluster_df[['Latitude', 'Longitude']].values.tolist() + [depot]

        route, distance_km = optimize_route(cluster_locations)
        total_orders = cluster_df['Orders'].sum()
        cost_per_order = (VEHICLE_COST_DEFAULT / (30 * total_orders)) if total_orders else 0

        cluster_df['Route Sequence'] = range(1, len(cluster_df) + 1)
        cluster_df['Optimized Route'] = route[1:-1]  # remove depot start and end
        cluster_df['Total Distance (km)'] = round(distance_km, 2)
        cluster_df['Cost per Order (INR)'] = round(cost_per_order, 2)

        results.append(cluster_df)

    final_df = pd.concat(results).reset_index(drop=True)

    st.download_button("📥 Download Optimized Routes CSV", final_df.to_csv(index=False), "optimized_routes.csv")

    st.subheader("📍 Route Map")
    m = folium.Map(location=[12.9840, 77.7490], zoom_start=10)

    for cluster_id, group in final_df.groupby('Cluster ID'):
        for _, row in group.iterrows():
            popup_text = f"Cluster: {cluster_id}, Society: {row['Society Name']}<br>CPO: ₹{row['Cost per Order (INR)']}"
            folium.Marker(
                location=[row['Latitude'], row['Longitude']],
                popup=popup_text,
                tooltip=f"Seq: {row['Route Sequence']}"
            ).add_to(m)

    st_folium(m, width=1200, height=600)

    with st.expander("🔍 Filter Routes"):
        selected_city = st.selectbox("Select City", options=["All"] + sorted(df['City'].unique().tolist()))
        selected_society = st.selectbox("Select Society", options=["All"] + sorted(df['Society Name'].unique().tolist()))
        selected_drop = st.selectbox("Select Drop Point", options=["All"] + sorted(df['Drop Point'].unique().tolist()))

        filtered_df = final_df.copy()
        if selected_city != "All":
            filtered_df = filtered_df[filtered_df['City'] == selected_city]
        if selected_society != "All":
            filtered_df = filtered_df[filtered_df['Society Name'] == selected_society]
        if selected_drop != "All":
            filtered_df = filtered_df[filtered_df['Drop Point'] == selected_drop]

        st.dataframe(filtered_df, use_container_width=True)
