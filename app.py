import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from geopy.geocoders import Nominatim
from sklearn.cluster import KMeans
import math
import numpy as np

# Streamlit UI setup
st.set_page_config(page_title="RK - Delivery Route Optimizer", layout="wide")
st.title("RK - Delivery Route Optimizer")
st.markdown("Optimize delivery routes with K-Means clustering and cost-effective routing under ₹4 per order.")

# Constants
VEHICLE_CAPACITY = 200
VEHICLE_COST_PER_MONTH = 35000
DAYS_IN_MONTH = 30
VEHICLE_COST_PER_DAY = VEHICLE_COST_PER_MONTH / DAYS_IN_MONTH
TARGET_COST_PER_ORDER = 4
SOURCE_NAME = "Soukya Road"
SOURCE_LAT = 12.9387
SOURCE_LON = 77.7837

# CSV Template Download
if st.button("📅 Download CSV Template"):
    template = pd.DataFrame({
        "Society ID": [101, 102],
        "Apartment": ["ABC Residency", "Green Heights"],
        "Latitude": [12.935, 12.938],
        "Longitude": [77.614, 77.610],
        "Orders": [120, 150]
    })
    csv = template.to_csv(index=False).encode('utf-8')
    st.download_button("Download Template", data=csv, file_name="milk_delivery_template.csv", mime='text/csv')

# File Upload
uploaded_file = st.file_uploader("Upload Delivery Data CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = df.dropna(subset=["Latitude", "Longitude", "Orders"])

    total_orders = df["Orders"].sum()
    num_clusters = math.ceil(total_orders / VEHICLE_CAPACITY)

    coords = df[["Latitude", "Longitude"]]
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(coords)
    df['Cluster'] = kmeans.labels_

    st.success(f"🚚 Total Orders: {total_orders} | Estimated Routes (Clusters): {num_clusters}")

    route_summary = []
    for cluster_id in sorted(df['Cluster'].unique()):
        cluster_df = df[df['Cluster'] == cluster_id].reset_index(drop=True)
        cluster_df = pd.concat([
            pd.DataFrame({
                "Society ID": ["SRC"],
                "Apartment": [SOURCE_NAME],
                "Latitude": [SOURCE_LAT],
                "Longitude": [SOURCE_LON],
                "Orders": [0]
            }),
            cluster_df
        ], ignore_index=True)

        def create_data_model():
            return {
                'locations': list(zip(cluster_df['Latitude'], cluster_df['Longitude'])),
                'num_vehicles': 1,
                'depot': 0
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

            cluster_df = cluster_df.iloc[route].reset_index(drop=True)
            cluster_df['Stop'] = range(1, len(cluster_df) + 1)

            # Map
            m = folium.Map(location=[cluster_df['Latitude'].mean(), cluster_df['Longitude'].mean()], zoom_start=13)
            for i, row in cluster_df.iterrows():
                folium.Marker(
                    location=[row["Latitude"], row["Longitude"]],
                    popup=f"{row['Stop']}. {row['Apartment']} (ID: {row['Society ID']}, {row['Orders']} orders)",
                    icon=folium.Icon(color='blue' if i > 0 else 'green', icon='home' if i > 0 else 'play')
                ).add_to(m)

            folium.PolyLine(
                locations=list(zip(cluster_df['Latitude'], cluster_df['Longitude'])),
                color="red", weight=3
            ).add_to(m)
            st.subheader(f"📍 Route {cluster_id + 1}")
            st_folium(m, height=400, width=800)

            # Cost calculation
            route_orders = cluster_df['Orders'].sum()
            total_km = total_distance * 111
            cost_per_order = VEHICLE_COST_PER_DAY / route_orders if route_orders else 0

            st.markdown(f"- Total Orders: **{route_orders}**")
            st.markdown(f"- Distance: **{total_km:.2f} km**")
            st.markdown(f"- Cost per Order: **₹{cost_per_order:.2f}**")

            st.dataframe(cluster_df[["Stop", "Society ID", "Apartment", "Latitude", "Longitude", "Orders"]])

            csv_export = cluster_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label=f"📄 Download Route {cluster_id + 1} CSV",
                data=csv_export,
                file_name=f"route_{cluster_id + 1}.csv",
                mime="text/csv"
            )

            route_summary.append({
                "Route": cluster_id + 1,
                "Orders": route_orders,
                "Distance (km)": round(total_km, 2),
                "Cost per Order (₹)": round(cost_per_order, 2)
            })

    if route_summary:
        st.subheader("📊 Route Summary")
        st.dataframe(pd.DataFrame(route_summary))
        avg_cost = np.mean([r['Cost per Order (₹)'] for r in route_summary])
        st.success(f"✅ Average Cost per Order across routes: ₹{avg_cost:.2f} (Target: < ₹{TARGET_COST_PER_ORDER})")
