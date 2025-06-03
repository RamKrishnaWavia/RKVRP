import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from folium.plugins import MarkerCluster

# Constants
VEHICLE_CAPACITY = 200
VEHICLE_COST_PER_MONTH = 35000
WORKING_DAYS_PER_MONTH = 30
DEPOT_LOCATION = (12.8996, 77.7605)  # Soukya Road

def haversine_distance(lat1, lon1, lat2, lon2):
    from math import radians, sin, cos, sqrt, atan2
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    return R * (2 * atan2(sqrt(a), sqrt(1 - a)))

def calculate_clusters(df):
    total_orders = df['Orders'].sum()
    clusters = int(np.ceil(total_orders / VEHICLE_CAPACITY))
    return clusters

def create_distance_matrix(locations):
    size = len(locations)
    matrix = {}
    for from_idx in range(size):
        matrix[from_idx] = {}
        for to_idx in range(size):
            if from_idx == to_idx:
                matrix[from_idx][to_idx] = 0
            else:
                matrix[from_idx][to_idx] = int(haversine_distance(*locations[from_idx], *locations[to_idx]) * 1000)
    return matrix

def optimize_cluster_routes(cluster_df):
    locations = [(DEPOT_LOCATION[0], DEPOT_LOCATION[1])] + list(zip(cluster_df['Latitude'], cluster_df['Longitude'])) + [DEPOT_LOCATION]
    distance_matrix = create_distance_matrix(locations)
    manager = pywrapcp.RoutingIndexManager(len(distance_matrix), 1, 0, len(distance_matrix) - 1)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    solution = routing.SolveWithParameters(search_parameters)
    if solution:
        index = routing.Start(0)
        route = []
        distance = 0
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route.append(node)
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            distance += routing.GetArcCostForVehicle(previous_index, index, 0)
        route.append(manager.IndexToNode(index))
        return route, distance / 1000
    else:
        return [], 0

def main():
    st.title("Milk Delivery Route Optimizer (CPO < ₹4)")

    uploaded_file = st.file_uploader("Upload CSV with society data", type=['csv'])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        required_cols = ['Society ID', 'Society Name', 'City', 'Drop Point', 'Latitude', 'Longitude', 'Orders']
        if not all(col in df.columns for col in required_cols):
            st.error(f"Missing required columns: {required_cols}")
            return

        df['Orders'] = pd.to_numeric(df['Orders'], errors='coerce')
        if df['Orders'].isnull().any():
            st.error("⚠️ Some 'Orders' values are missing or non-numeric. Please check your CSV.")
            st.stop()
        df['Orders'] = df['Orders'].astype(int)

        num_clusters = calculate_clusters(df)
        kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(df[['Latitude', 'Longitude']])
        df['Cluster ID'] = kmeans.labels_

        summary_data = []
        m = folium.Map(location=DEPOT_LOCATION, zoom_start=11)
        marker_cluster = MarkerCluster().add_to(m)

        for cluster_id in sorted(df['Cluster ID'].unique()):
            cluster_df = df[df['Cluster ID'] == cluster_id]
            route, dist_km = optimize_cluster_routes(cluster_df)

            if not route:
                st.warning(f"Route could not be optimized for cluster {cluster_id}")
                continue

            route_points = [DEPOT_LOCATION] + list(zip(cluster_df.iloc[[i - 1 for i in route[1:-1]]]['Latitude'],
                                                       cluster_df.iloc[[i - 1 for i in route[1:-1]]]['Longitude'])) + [DEPOT_LOCATION]
            folium.PolyLine(route_points, color='blue', weight=2.5, opacity=0.8).add_to(m)

            total_orders = cluster_df['Orders'].sum()
            cpo = (VEHICLE_COST_PER_MONTH / WORKING_DAYS_PER_MONTH) / total_orders

            for i, (lat, lon) in enumerate(route_points[1:-1]):
                society_name = cluster_df.iloc[i]['Society Name']
                drop_point = cluster_df.iloc[i]['Drop Point']
                popup_text = f"Society: {society_name}<br>Drop Point: {drop_point}<br>Cluster: {cluster_id}<br>CPO: ₹{cpo:.2f}"
                folium.Marker([lat, lon], popup=popup_text, tooltip=f"#{i+1} - {society_name}").add_to(marker_cluster)

            summary_data.append({
                "Cluster ID": cluster_id,
                "Total Orders": total_orders,
                "Distance (km)": round(dist_km, 2),
                "CPO (₹)": round(cpo, 2)
            })

        st.subheader("Optimized Route Map")
        st_data = st_folium(m, width=1000, height=600)

        st.subheader("Route Summary")
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df)

        csv = summary_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Summary as CSV", csv, "route_summary.csv", "text/csv")

    else:
        st.markdown("Download CSV template to upload")
        sample = pd.DataFrame({
            'Society ID': ['S001', 'S002'],
            'Society Name': ['Green Meadows', 'Lakeview Residency'],
            'City': ['Bangalore', 'Bangalore'],
            'Drop Point': ['Tower A', 'Tower B'],
            'Latitude': [12.9001, 12.9051],
            'Longitude': [77.7605, 77.7612],
            'Orders': [150, 180]
        })
        st.dataframe(sample)
        csv = sample.to_csv(index=False).encode('utf-8')
        st.download_button("Download Template", csv, "template.csv", "text/csv")

if __name__ == "__main__":
    main()
