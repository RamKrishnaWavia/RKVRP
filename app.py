import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
from haversine import haversine
from ortools.constraint_solver import routing_enums_pb2, pywrapcp

# Constants
VEHICLE_CAPACITY = 200
VEHICLE_COST = 35000  # Monthly cost in INR
ORDERS_PER_DAY = 30  # Assume 30 days/month for daily routes

# Distance Matrix Helper
def compute_distance_matrix(locations):
    size = len(locations)
    matrix = [[0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            if i != j:
                matrix[i][j] = int(haversine(locations[i], locations[j]) * 1000)  # in meters
    return matrix

# Route Optimizer
def optimize_route(locations):
    manager = pywrapcp.RoutingIndexManager(len(locations), 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    distance_matrix = compute_distance_matrix(locations)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    solution = routing.SolveWithParameters(search_parameters)

    route = []
    distance_m = 0
    if solution:
        index = routing.Start(0)
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            prev_index = index
            index = solution.Value(routing.NextVar(index))
            distance_m += routing.GetArcCostForVehicle(prev_index, index, 0)
        route.append(manager.IndexToNode(index))

    return route, distance_m / 1000  # in km

# Upload & Template Section
st.title("Milk Delivery Route Optimizer")

with st.expander("📥 Download CSV Template"):
    sample_df = pd.DataFrame({
        'Society ID': ['S1', 'S2'],
        'Society Name': ['Green Apartments', 'Blue Towers'],
        'City': ['Bangalore', 'Bangalore'],
        'Drop Point': ['Gate 1', 'Main Entrance'],
        'Latitude': [12.935, 12.936],
        'Longitude': [77.614, 77.616],
        'Orders': [150, 120],
    })
    st.download_button("Download Template", sample_df.to_csv(index=False), "template.csv")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    required_cols = ['Society ID', 'Society Name', 'City', 'Drop Point', 'Latitude', 'Longitude', 'Orders']
    if not all(col in df.columns for col in required_cols):
        st.error(f"Missing required columns: {set(required_cols) - set(df.columns)}")
    else:
        df['Orders'] = pd.to_numeric(df['Orders'], errors='coerce')
        invalid_orders = df[df['Orders'].isna() | (df['Orders'] <= 0)]

        if not invalid_orders.empty:
            st.warning("⚠️ Some 'Orders' values are missing or invalid. Please correct them.")
            st.dataframe(invalid_orders)
        else:
            # Auto Cluster Assignment
            df['Latitude'] = df['Latitude'].astype(float)
            df['Longitude'] = df['Longitude'].astype(float)
            total_orders = df['Orders'].sum()
            num_clusters = max(1, int(np.ceil(total_orders / VEHICLE_CAPACITY)))
            kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(df[['Latitude', 'Longitude']])
            df['Cluster'] = kmeans.labels_

            total_cost = 0
            total_orders_all = 0

            for cluster_id in sorted(df['Cluster'].unique()):
                cluster_df = df[df['Cluster'] == cluster_id].reset_index(drop=True)
                locations = cluster_df[['Latitude', 'Longitude']].values.tolist()
                route, distance_km = optimize_route(locations)

                cluster_df['Route Order'] = [route.index(i) for i in range(len(cluster_df))]
                cluster_df = cluster_df.sort_values('Route Order')

                order_sum = cluster_df['Orders'].sum()
                cluster_cost = VEHICLE_COST / ORDERS_PER_DAY
                cost_per_order = cluster_cost / order_sum
                total_cost += cluster_cost
                total_orders_all += order_sum

                st.subheader(f"🧭 Cluster {cluster_id + 1}")
                st.markdown(f"**Total Orders:** {order_sum}  |  **Distance:** {distance_km:.2f} km  |  **Cost/Order:** ₹{cost_per_order:.2f}")
                st.dataframe(cluster_df[['Society Name', 'Orders', 'Route Order']])

                # Map
                cluster_map = folium.Map(location=locations[0], zoom_start=13)
                for i, (lat, lon) in enumerate(locations):
                    folium.Marker([lat, lon], tooltip=f"{cluster_df.loc[i, 'Society Name']}, Orders: {cluster_df.loc[i, 'Orders']}").add_to(cluster_map)
                st_folium(cluster_map, width=700)

            st.success(f"✅ Total Orders: {total_orders_all} | Total Cost: ₹{total_cost:.2f} | Overall Cost/Order: ₹{total_cost/total_orders_all:.2f}")
