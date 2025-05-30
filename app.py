import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import math
import io

# Page config
st.set_page_config(page_title="RK - Delivery Route Optimizer", layout="wide")
st.title("RK - Delivery Route Optimizer")
st.markdown("Optimize early morning deliveries (4 AM to 7 AM) with minimum cost. One vehicle per route, max 200 orders/vehicle. Target: Cost per order < ₹4.")

# CSV Template Download
if st.button("📄 Download CSV Template"):
    template = pd.DataFrame({
        "Society ID": [101, 102],
        "Apartment": ["ABC Residency", "Green Heights"],
        "Latitude": [12.935, 12.938],
        "Longitude": [77.614, 77.610],
        "Orders": [120, 100]
    })
    csv = template.to_csv(index=False).encode("utf-8")
    st.download_button("Download Template", data=csv, file_name="delivery_template.csv", mime="text/csv")

# Vehicle cost input
vehicle_monthly_cost = st.number_input("Enter Monthly Cost per Vehicle (₹)", value=35000, step=500)

# File Upload
uploaded_file = st.file_uploader("Upload Delivery Data CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    required_cols = ["Society ID", "Apartment", "Latitude", "Longitude", "Orders"]
    if not all(col in df.columns for col in required_cols):
        st.error("CSV must contain: Society ID, Apartment, Latitude, Longitude, Orders")
        st.stop()

    st.success("✅ Data uploaded successfully")
    st.dataframe(df)

    locations = list(zip(df.Latitude, df.Longitude))
    demands = df.Orders.tolist()

    def compute_euclidean_distance_matrix(locations):
        distances = {}
        for from_counter, from_node in enumerate(locations):
            distances[from_counter] = {}
            for to_counter, to_node in enumerate(locations):
                distances[from_counter][to_counter] = math.hypot(
                    from_node[0] - to_node[0], from_node[1] - to_node[1])
        return distances

    distance_matrix = compute_euclidean_distance_matrix(locations)

    # Split into routes with max 200 orders per vehicle
    sorted_df = df.sort_values("Orders", ascending=False).reset_index(drop=True)
    routes = []
    current_route = []
    current_orders = 0

    for i, row in sorted_df.iterrows():
        if current_orders + row["Orders"] > 200 and current_route:
            routes.append(current_route)
            current_route = []
            current_orders = 0
        current_route.append(i)
        current_orders += row["Orders"]
    if current_route:
        routes.append(current_route)

    total_cost = 0
    total_orders = df["Orders"].sum()
    all_routes_df = []

    for route_idx, route_indices in enumerate(routes):
        route_df = df.loc[route_indices].reset_index(drop=True)
        route_locs = list(zip(route_df.Latitude, route_df.Longitude))
        route_dm = compute_euclidean_distance_matrix(route_locs)

        manager = pywrapcp.RoutingIndexManager(len(route_locs), 1, 0)
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_idx, to_idx):
            from_node = manager.IndexToNode(from_idx)
            to_node = manager.IndexToNode(to_idx)
            return int(route_dm[from_node][to_node] * 100000)

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        solution = routing.SolveWithParameters(search_parameters)

        if not solution:
            st.error(f"Route {route_idx+1} optimization failed")
            continue

        index = routing.Start(0)
        route_order = []
        total_route_distance = 0

        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_order.append(node_index)
            next_index = solution.Value(routing.NextVar(index))
            total_route_distance += route_dm[node_index][manager.IndexToNode(next_index)]
            index = next_index
        route_order.append(manager.IndexToNode(index))

        m = folium.Map(location=[route_df.Latitude.mean(), route_df.Longitude.mean()], zoom_start=13)
        cumulative_distance = 0

        for i, stop in enumerate(route_order[:-1]):
            row = route_df.iloc[stop]
            next_stop = route_order[i+1]
            distance_to_next = route_dm[stop][next_stop] * 111
            cumulative_distance += distance_to_next
            folium.Marker(
                location=[row.Latitude, row.Longitude],
                popup=f"Stop {i+1}: {row['Apartment']} (Orders: {row['Orders']})\nDist to next: {distance_to_next:.2f} km",
                icon=folium.Icon(color='blue' if i > 0 else 'green')
            ).add_to(m)

        folium.PolyLine(
            locations=[(route_df.iloc[stop].Latitude, route_df.iloc[stop].Longitude) for stop in route_order],
            color="red", weight=3
        ).add_to(m)

        st.subheader(f"🛻 Route {route_idx+1} Map")
        st_folium(m, height=400, width=900)

        ordered_df = route_df.iloc[route_order].reset_index(drop=True)
        ordered_df["Stop"] = range(1, len(ordered_df)+1)
        all_routes_df.append(ordered_df.assign(Route=f"Route {route_idx+1}"))

        route_cost = vehicle_monthly_cost
        route_orders = ordered_df["Orders"].sum()
        cost_per_order = route_cost / route_orders
        total_cost += route_cost

        st.markdown(f"**Route {route_idx+1} Summary:**")
        st.markdown(f"- Total Orders: {route_orders}")
        st.markdown(f"- Total Distance: {total_route_distance * 111:.2f} km")
        st.markdown(f"- Vehicle Monthly Cost: ₹{vehicle_monthly_cost}")
        st.markdown(f"- Cost per Order: ₹{cost_per_order:.2f} {'✅' if cost_per_order <= 4 else '❌'}")
        st.dataframe(ordered_df[["Stop", "Society ID", "Apartment", "Latitude", "Longitude", "Orders"]])

    # Final Summary
    if all_routes_df:
        full_df = pd.concat(all_routes_df).reset_index(drop=True)
        st.subheader("📦 Final Optimized Routes Summary")
        st.markdown(f"- Total Vehicles Used: {len(routes)}")
        st.markdown(f"- Total Orders: {total_orders}")
        st.markdown(f"- Total Monthly Cost: ₹{total_cost}")
        st.markdown(f"- Overall Cost per Order: ₹{total_cost / total_orders:.2f} {'✅' if total_cost / total_orders <= 4 else '❌'}")

        # Download
        csv_export = full_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Optimized Routes CSV", data=csv_export, file_name="optimized_routes.csv", mime="text/csv")
