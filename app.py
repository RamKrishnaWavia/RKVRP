import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np
from geopy.distance import geodesic
import io

st.set_page_config(page_title="RK - Delivery Route Optimizer", layout="wide")
st.title("RK - Delivery Route Optimizer")
st.markdown("Optimize your milk delivery routes from Soukya Road with minimum cost per order and efficient sequencing.")

# User Inputs
vehicle_cost = st.number_input("Enter monthly cost per vehicle (₹)", value=35000)

# CSV Template
if st.button("📄 Download CSV Template"):
    template = pd.DataFrame({
        "Society ID": [0, 1, 2],
        "Apartment": ["Depot - Soukya Road", "ABC Residency", "Green Heights"],
        "Latitude": [12.935, 12.938, 12.942],
        "Longitude": [77.614, 77.610, 77.617],
        "Orders": [0, 100, 120]  # Depot has 0 orders
    })
    csv = template.to_csv(index=False).encode('utf-8')
    st.download_button("Download Template", data=csv, file_name="milk_delivery_template.csv", mime='text/csv')

# Upload File
uploaded_file = st.file_uploader("Upload Delivery Data CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Validate Columns
    required_cols = ["Society ID", "Apartment", "Latitude", "Longitude", "Orders"]
    if not all(col in df.columns for col in required_cols):
        st.error("Missing one or more required columns in uploaded CSV.")
        st.stop()

    depot = df.iloc[0]  # Soukya Road
    customers = df.iloc[1:].reset_index(drop=True)

    # KMeans Clustering (based on orders)
    total_orders = customers["Orders"].sum()
    num_clusters = int(np.ceil(total_orders / 200))

    coords = customers[["Latitude", "Longitude"]].values
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(coords, sample_weight=customers["Orders"])
    customers["Cluster"] = kmeans.labels_

    route_outputs = []
    full_routes_csv = []
    vehicle_number = 1

    for cluster_id in range(num_clusters):
        cluster_df = customers[customers["Cluster"] == cluster_id].copy().reset_index(drop=True)

        # Insert depot at start and end
        cluster_df = pd.concat([pd.DataFrame([depot]), cluster_df, pd.DataFrame([depot])], ignore_index=True)

        # Create distance matrix using geopy
        def compute_distance_matrix(df):
            dist = {}
            for i in range(len(df)):
                dist[i] = {}
                for j in range(len(df)):
                    if i == j:
                        dist[i][j] = 0
                    else:
                        dist[i][j] = geodesic((df.iloc[i]["Latitude"], df.iloc[i]["Longitude"]),
                                              (df.iloc[j]["Latitude"], df.iloc[j]["Longitude"])).km
            return dist

        distance_matrix = compute_distance_matrix(cluster_df)

        manager = pywrapcp.RoutingIndexManager(len(distance_matrix), 1, 0)
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(distance_matrix[from_node][to_node] * 1000)  # meters

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

        solution = routing.SolveWithParameters(search_parameters)

        if solution:
            index = routing.Start(0)
            sequence = []
            total_distance = 0
            while not routing.IsEnd(index):
                sequence.append(manager.IndexToNode(index))
                prev_index = index
                index = solution.Value(routing.NextVar(index))
                total_distance += distance_matrix[manager.IndexToNode(prev_index)][manager.IndexToNode(index)]
            sequence.append(manager.IndexToNode(index))

            # Route Data
            ordered_df = cluster_df.iloc[sequence].reset_index(drop=True)
            ordered_df["Stop"] = range(1, len(ordered_df)+1)
            ordered_df["Vehicle"] = vehicle_number

            # Map
            m = folium.Map(location=[ordered_df["Latitude"].mean(), ordered_df["Longitude"].mean()], zoom_start=13)
            for i in range(len(ordered_df)-1):
                row = ordered_df.iloc[i]
                next_row = ordered_df.iloc[i+1]
                folium.Marker(
                    location=[row["Latitude"], row["Longitude"]],
                    popup=f"{row['Stop']}. {row['Apartment']} (ID: {row['Society ID']}, {row['Orders']} orders)",
                    icon=folium.Icon(color='blue' if i > 0 else 'green', icon='home')
                ).add_to(m)
                folium.PolyLine([
                    (row["Latitude"], row["Longitude"]),
                    (next_row["Latitude"], next_row["Longitude"])
                ], color='red', weight=3, tooltip=f"{distance_matrix[sequence[i]][sequence[i+1]]:.2f} km").add_to(m)
            st.subheader(f"🛣️ Route {vehicle_number} (Vehicle #{vehicle_number})")
            st_folium(m, height=500, width=900)

            # Costing
            total_orders = ordered_df["Orders"].sum()
            cost_per_order = vehicle_cost / total_orders if total_orders else 0

            st.markdown(f"**Total Distance:** {total_distance:.2f} km")
            st.markdown(f"**Total Orders:** {total_orders}")
            st.markdown(f"**Cost per Order:** ₹{cost_per_order:.2f}")

            st.dataframe(ordered_df[["Stop", "Society ID", "Apartment", "Latitude", "Longitude", "Orders"]])

            route_outputs.append({
                "Route": vehicle_number,
                "Orders": total_orders,
                "Distance (km)": round(total_distance, 2),
                "Cost per Order (₹)": round(cost_per_order, 2)
            })

            full_routes_csv.append(ordered_df)
            vehicle_number += 1

    # Route Summary
    st.subheader("📦 Route Summary")
    route_summary = pd.DataFrame(route_outputs)
    st.dataframe(route_summary)

    # Download Final Optimized Routes
    final_df = pd.concat(full_routes_csv, ignore_index=True)
    csv_final = final_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download All Optimized Routes CSV", data=csv_final, file_name="optimized_routes.csv", mime="text/csv")
