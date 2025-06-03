import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from geopy.geocoders import Nominatim
from sklearn.cluster import KMeans
import numpy as np
import io
import math

# Streamlit page setup
st.set_page_config(page_title="RK - Delivery Route Optimizer", layout="wide")
st.title("RK - Delivery Route Optimizer")
st.markdown("Optimize early morning deliveries (4 AM to 7 AM) with minimal cost per order and route efficiency.")

# User input: vehicle cost
vehicle_monthly_cost = st.number_input("Enter vehicle cost per month (₹)", min_value=1000, value=35000, step=1000)

# CSV Template Download
if st.button("📅 Download CSV Template"):
    template = pd.DataFrame({
        "Society ID": [101, 101, 102, 102],
        "Society Name": ["ABC Residency", "ABC Residency", "Green Heights", "Green Heights"],
        "Drop Point": ["Block A", "Block B", "Tower 1", "Tower 2"],
        "Latitude": [12.9351, 12.9353, 12.9381, 12.9383],
        "Longitude": [77.6139, 77.6140, 77.6098, 77.6101],
        "Orders": [30, 40, 50, 60]
    })
    csv = template.to_csv(index=False).encode('utf-8')
    st.download_button("Download Template", data=csv, file_name="delivery_template.csv", mime='text/csv')

# File upload
uploaded_file = st.file_uploader("Upload Delivery Data CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    required_columns = {"Society ID", "Society Name", "Drop Point", "Latitude", "Longitude", "Orders"}
    if not required_columns.issubset(df.columns):
        st.error("Uploaded CSV does not contain all required columns.")
        st.stop()

    st.success("✅ Data loaded successfully")
    st.dataframe(df)

    # Append depot at the beginning and end
    depot_location = (12.9350, 77.6100)  # Soukya Road coordinates

    locations = [(depot_location[0], depot_location[1])]
    locations += list(zip(df["Latitude"], df["Longitude"]))

    orders = [0] + df["Orders"].tolist()  # 0 orders at depot

    # Cluster based on capacity (200 orders per vehicle)
    max_orders_per_vehicle = 200
    order_array = np.array(orders[1:])
    total_orders = order_array.sum()
    num_clusters = math.ceil(total_orders / max_orders_per_vehicle)

    coords = np.array(list(zip(df["Latitude"], df["Longitude"])))
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(coords)
    df["Cluster"] = kmeans.labels_

    all_routes = []
    all_costs = []
    total_distance = 0
    total_orders_all = 0
    route_map = folium.Map(location=[df["Latitude"].mean(), df["Longitude"].mean()], zoom_start=13)

    for cluster_id in sorted(df["Cluster"].unique()):
        cluster_df = df[df["Cluster"] == cluster_id].reset_index(drop=True)

        # Rebuild location list (depot + cluster)
        cluster_locations = [depot_location] + list(zip(cluster_df["Latitude"], cluster_df["Longitude"]))
        cluster_orders = [0] + cluster_df["Orders"].tolist()

        # OR-Tools setup
        def create_data_model():
            return {
                'locations': cluster_locations,
                'num_vehicles': 1,
                'depot': 0,
                'demands': cluster_orders,
                'vehicle_capacities': [max_orders_per_vehicle]
            }

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

        def demand_callback(from_index):
            return data['demands'][manager.IndexToNode(from_index)]

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            data['vehicle_capacities'],
            True,
            'Capacity'
        )

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

        solution = routing.SolveWithParameters(search_parameters)

        if solution:
            index = routing.Start(0)
            route = []
            route_distance = 0
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route.append(node_index)
                next_index = solution.Value(routing.NextVar(index))
                route_distance += distance_matrix[node_index][manager.IndexToNode(next_index)]
                index = next_index
            route.append(manager.IndexToNode(index))  # return to depot

            route_df = cluster_df.iloc[[i-1 for i in route if i != 0]].copy()
            route_df.insert(0, "Stop", list(range(1, len(route_df)+1)))
            route_df.insert(0, "Route", f"Vehicle {cluster_id+1}")

            for i, stop in enumerate(route):
                loc = data['locations'][stop]
                popup = "Depot" if stop == 0 else f"{i}. {cluster_df.iloc[stop-1]['Drop Point']} (Orders: {cluster_df.iloc[stop-1]['Orders']})"
                folium.Marker(
                    location=loc,
                    popup=popup,
                    icon=folium.Icon(color='green' if stop == 0 else 'blue', icon='home')
                ).add_to(route_map)

            folium.PolyLine(
                locations=[data['locations'][i] for i in route],
                color="red", weight=3, tooltip=f"Route {cluster_id+1}"
            ).add_to(route_map)

            all_routes.append(route_df)
            all_costs.append(route_distance)
            total_distance += route_distance
            total_orders_all += sum(cluster_df["Orders"])
        else:
            st.warning(f"No solution found for cluster {cluster_id+1}")

    st.subheader("🗺 Route Map")
    st_folium(route_map, height=500, width=1000)

    if all_routes:
        result_df = pd.concat(all_routes, ignore_index=True)
        st.subheader("📋 Optimized Routes")
        st.dataframe(result_df)

        # Summary
        st.subheader("💰 Cost Summary")
        total_km = total_distance * 111
        cost_per_day_per_vehicle = vehicle_monthly_cost / 30
        total_cost = len(all_routes) * cost_per_day_per_vehicle
        cpo = total_cost / total_orders_all if total_orders_all else 0

        st.markdown(f"- **Total vehicles used:** {len(all_routes)}")
        st.markdown(f"- **Total distance (km):** {total_km:.2f}")
        st.markdown(f"- **Total cost (₹):** ₹{total_cost:.2f}")
        st.markdown(f"- **Cost per order (₹):** ₹{cpo:.2f}")

        # Download optimized route
        if st.button("📄 Download Optimized Route CSV"):
            csv_export = result_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Route CSV", data=csv_export, file_name="optimized_routes.csv", mime="text/csv")
