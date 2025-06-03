import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from geopy.geocoders import Nominatim

st.set_page_config(page_title="RK - Delivery Route Optimizer", layout="wide")

st.title("RK - Delivery Route Optimizer")
st.markdown("Optimize early morning deliveries (4 AM to 7 AM) with cost-effective routes.\n"
            "Max 200 orders per vehicle per route. Depot: Soukya Road.")

# CSV Template Download
if st.button("📥 Download Sample Input CSV Template"):
    sample_data = pd.DataFrame({
        "Society ID": [101, 102, 103],
        "Society Name": ["ABC Residency", "Green Heights", "Sunshine Apartments"],
        "City": ["Bangalore", "Bangalore", "Chennai"],
        "Drop Point": ["Drop1", "Drop1", "Drop2"],
        "Latitude": [12.935, 12.938, 13.082],
        "Longitude": [77.614, 77.610, 80.270],
        "Orders": [15, 20, 30]
    })
    csv_bytes = sample_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Template CSV",
        data=csv_bytes,
        file_name="milk_delivery_input_template.csv",
        mime="text/csv"
    )

uploaded_file = st.file_uploader("Upload Delivery Data CSV", type=["csv"])

# Input vehicle monthly cost
vehicle_monthly_cost = st.number_input("Enter Vehicle Monthly Cost (₹)", min_value=1000, value=35000, step=500)

MAX_ORDERS_PER_VEHICLE = 200
DEPOT_NAME = "Soukya Road"
DEPOT_COORDS = (12.9554, 77.6479)  # Approx lat/lon for Soukya Road; adjust if needed

def validate_input(df):
    required_cols = ['Society ID', 'Society Name', 'City', 'Drop Point', 'Latitude', 'Longitude', 'Orders']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        return False
    return True

def cluster_data(df):
    # Cluster by KMeans based on max orders per vehicle
    coords = df[['Latitude', 'Longitude']].to_numpy()
    total_orders = df['Orders'].sum()
    num_clusters = max(1, int(np.ceil(total_orders / MAX_ORDERS_PER_VEHICLE)))
    if len(df) < num_clusters:
        num_clusters = len(df)  # Cannot have more clusters than points
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(coords)
    df['Cluster ID'] = kmeans.labels_
    return df, num_clusters

def create_distance_matrix(locations):
    size = len(locations)
    dist_matrix = {}
    for i in range(size):
        dist_matrix[i] = {}
        for j in range(size):
            if i == j:
                dist_matrix[i][j] = 0
            else:
                dist_matrix[i][j] = int(((locations[i][0] - locations[j][0]) ** 2 + (locations[i][1] - locations[j][1]) ** 2) ** 0.5 * 100000)
    return dist_matrix

def optimize_route(df_cluster):
    # Add depot as first point for routing
    solution = routing.SolveWithParameters(search_parameters)
    if solution is None:
    st.error("❌ No feasible solution found. Please adjust vehicle capacity, number of orders, or time windows.")
    return [], 0
           
    depot = DEPOT_COORDS
    locations = [depot] + list(zip(df_cluster['Latitude'], df_cluster['Longitude']))
    demands = [0] + df_cluster['Orders'].tolist()

    # Setup OR Tools data model
    data = {}
    data['distance_matrix'] = create_distance_matrix(locations)
    data['demands'] = demands
    data['vehicle_capacities'] = [MAX_ORDERS_PER_VEHICLE]
    data['num_vehicles'] = 1
    data['depot'] = 0

    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)

    def demand_callback(from_index):
        node = manager.IndexToNode(from_index)
        return data['demands'][node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data['vehicle_capacities'],  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity'
    )

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        route = []
        index = routing.Start(0)
        route_distance = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route.append(node_index)
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
        route.append(manager.IndexToNode(index))  # end point (depot)
        return route, route_distance / 100000  # convert back approx
    else:
        return None, None

def build_map_and_show(df, clusters_routes, clusters_distances, clusters_cost_per_order):
    m = folium.Map(location=DEPOT_COORDS, zoom_start=10)
    # Add depot marker
    folium.Marker(
        location=DEPOT_COORDS,
        popup=f"Depot: {DEPOT_NAME}",
        icon=folium.Icon(color='red', icon='home')
    ).add_to(m)

    for cluster_id, route in clusters_routes.items():
        cluster_df = df[df['Cluster ID'] == cluster_id].reset_index(drop=True)
        route_coords = [DEPOT_COORDS] + list(zip(cluster_df['Latitude'], cluster_df['Longitude'])) + [DEPOT_COORDS]
        # Add markers with sequence and popup info
        for idx, point_idx in enumerate(route[1:-1], start=1):
            row = cluster_df.iloc[point_idx - 1]  # route includes depot as 0th, so offset by -1
            popup_text = (f"Stop {idx}: {row['Society Name']} (ID: {row['Society ID']})<br>"
                          f"City: {row['City']}<br>"
                          f"Drop Point: {row['Drop Point']}<br>"
                          f"Orders: {row['Orders']}<br>"
                          f"Cluster: {cluster_id}<br>"
                          f"CPO: ₹{clusters_cost_per_order[cluster_id]:.2f}")
            folium.Marker(
                location=(row['Latitude'], row['Longitude']),
                popup=popup_text,
                icon=folium.DivIcon(html=f"""<div style="font-size: 12pt; color: blue;">{idx}</div>""")
            ).add_to(m)
        # Draw polyline for route
        folium.PolyLine(locations=route_coords, color='blue', weight=3, opacity=0.7).add_to(m)

    st.subheader("🗺️ Optimized Delivery Routes Map")
    st_folium(m, height=600)

def main():
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if not validate_input(df):
            st.stop()

        # Filter UI
        st.sidebar.header("Filter Routes")
        cities = df['City'].unique().tolist()
        city_filter = st.sidebar.multiselect("Select City", options=cities, default=cities)
        filtered_df = df[df['City'].isin(city_filter)]

        societies = filtered_df['Society Name'].unique().tolist()
        society_filter = st.sidebar.multiselect("Select Society", options=societies, default=societies)
        filtered_df = filtered_df[filtered_df['Society Name'].isin(society_filter)]

        drop_points = filtered_df['Drop Point'].unique().tolist()
        drop_point_filter = st.sidebar.multiselect("Select Drop Point", options=drop_points, default=drop_points)
        filtered_df = filtered_df[filtered_df['Drop Point'].isin(drop_point_filter)]

        # Clustering
        clustered_df, num_clusters = cluster_data(filtered_df)
        st.write(f"Total Orders: {clustered_df['Orders'].sum()}, Number of Clusters (Routes): {num_clusters}")

        clusters_routes = {}
        clusters_distances = {}
        clusters_cost_per_order = {}

        for cluster_id in range(num_clusters):
            cluster_data_df = clustered_df[clustered_df['Cluster ID'] == cluster_id]
            route, dist = optimize_route(cluster_data_df)
            if route is None:
                st.warning(f"Could not find route solution for cluster {cluster_id}")
                continue
            clusters_routes[cluster_id] = route
            clusters_distances[cluster_id] = dist
            total_orders = cluster_data_df['Orders'].sum()
            cost_per_order = vehicle_monthly_cost / total_orders if total_orders > 0 else float('inf')
            clusters_cost_per_order[cluster_id] = cost_per_order

        build_map_and_show(clustered_df, clusters_routes, clusters_distances, clusters_cost_per_order)

        # Show route summary table
        summary_rows = []
        for cluster_id in clusters_routes:
            cluster_df = clustered_df[clustered_df['Cluster ID'] == cluster_id]
            summary_rows.append({
                'Cluster ID': cluster_id,
                'Total Orders': cluster_df['Orders'].sum(),
                'Route Distance (approx km)': round(clusters_distances[cluster_id], 2),
                'Cost per Order (₹)': round(clusters_cost_per_order[cluster_id], 2),
                'Depot': DEPOT_NAME
            })
        summary_df = pd.DataFrame(summary_rows)
        st.subheader("Route Summary")
        st.dataframe(summary_df)

        # Download optimized routes CSV
        csv_buffer = []
        for cluster_id in clusters_routes:
            cluster_df = clustered_df[clustered_df['Cluster ID'] == cluster_id].reset_index(drop=True)
            route = clusters_routes[cluster_id]
            ordered_societies = []
            for idx in route[1:-1]:  # skip depot start/end
                row = cluster_df.iloc[idx-1]  # offset -1 due to depot at 0
                ordered_societies.append(row)
            route_df = pd.DataFrame(ordered_societies)
            route_df['Cluster ID'] = cluster_id
            route_df['Route Sequence'] = range(1, len(route_df)+1)
            csv_buffer.append(route_df)

        if csv_buffer:
            all_routes_df = pd.concat(csv_buffer)
            csv_data = all_routes_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Optimized Routes CSV", data=csv_data, file_name="optimized_routes.csv", mime="text/csv")

if __name__ == "__main__":
    main()
