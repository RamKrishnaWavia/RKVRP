import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from geopy.geocoders import Nominatim
import math

# --- Constants ---
MAX_ORDERS_PER_ROUTE = 200
VEHICLE_MONTHLY_COST = 35000
COST_TARGET_PER_ORDER = 4

# --- App Layout ---
st.set_page_config(page_title="RK - Delivery Route Optimizer", layout="wide")
st.title("RK - Delivery Route Optimizer")
st.markdown("Optimize deliveries with cost per order under ₹4 using 200-order route constraints.")

# --- CSV Template Download ---
if st.button("📅 Download CSV Template"):
    template = pd.DataFrame({
        "Society ID": [101, 102],
        "Apartment": ["ABC Residency", "Green Heights"],
        "Latitude": [12.935, 12.938],
        "Longitude": [77.614, 77.610],
        "Orders": [150, 180]
    })
    csv = template.to_csv(index=False).encode('utf-8')
    st.download_button("Download Template", data=csv, file_name="milk_delivery_template.csv", mime='text/csv')

# --- File Upload ---
uploaded_file = st.file_uploader("Upload Delivery Data CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = df.dropna(subset=["Latitude", "Longitude"])

    total_orders = df["Orders"].sum()
    total_routes_required = math.ceil(total_orders / MAX_ORDERS_PER_ROUTE)

    # Sort by lat/long to form chunks (basic clustering)
    df = df.sort_values(by=["Latitude", "Longitude"]).reset_index(drop=True)

    st.success(f"📦 Total Orders: {total_orders} → Estimated Routes Required: {total_routes_required}")

    vehicle_cost_per_day = VEHICLE_MONTHLY_COST / 30
    cost_summary = []

    for i in range(total_routes_required):
        chunk = df.iloc[i * MAX_ORDERS_PER_ROUTE:(i + 1) * MAX_ORDERS_PER_ROUTE].copy()
        if chunk.empty:
            continue

        # Insert source: Soukya Road (lat/lon hardcoded)
        source = pd.DataFrame({
            "Society ID": ["SRC"],
            "Apartment": ["Soukya Road"],
            "Latitude": [12.9698],
            "Longitude": [77.7855],
            "Orders": [0]
        })
        route_df = pd.concat([source, chunk], ignore_index=True)

        # Calculate distance matrix
        locations = list(zip(route_df["Latitude"], route_df["Longitude"]))
        def compute_dist_matrix(locations):
            matrix = {}
            for i, from_loc in enumerate(locations):
                matrix[i] = {}
                for j, to_loc in enumerate(locations):
                    if i == j:
                        matrix[i][j] = 0
                    else:
                        dx = from_loc[0] - to_loc[0]
                        dy = from_loc[1] - to_loc[1]
                        matrix[i][j] = int(((dx**2 + dy**2) ** 0.5) * 100000)
            return matrix
        dist_matrix = compute_dist_matrix(locations)

        # OR-Tools VRP Setup
        manager = pywrapcp.RoutingIndexManager(len(locations), 1, 0)
        routing = pywrapcp.RoutingModel(manager)
        def distance_callback(from_index, to_index):
            return dist_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        search_params = pywrapcp.DefaultRoutingSearchParameters()
        search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

        solution = routing.SolveWithParameters(search_params)

        if solution:
            index = routing.Start(0)
            sequence, stops = [], []
            total_distance = 0
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                sequence.append(node)
                stops.append((route_df.iloc[node]["Latitude"], route_df.iloc[node]["Longitude"]))
                next_index = solution.Value(routing.NextVar(index))
                total_distance += dist_matrix[node][manager.IndexToNode(next_index)]
                index = next_index
            sequence.append(manager.IndexToNode(index))
            stops.append((route_df.iloc[manager.IndexToNode(index)]["Latitude"], route_df.iloc[manager.IndexToNode(index)]["Longitude"]))

            total_km = total_distance / 100000 * 111
            route_orders = route_df["Orders"].sum()
            route_cost = vehicle_cost_per_day
            cost_per_order = route_cost / route_orders if route_orders else 0

            m = folium.Map(location=stops[0], zoom_start=13)
            for j, stop in enumerate(sequence):
                r = route_df.iloc[stop]
                folium.Marker(
                    location=[r["Latitude"], r["Longitude"]],
                    popup=f"{j+1}. {r['Apartment']} (ID: {r['Society ID']}, {r['Orders']} orders)",
                    icon=folium.Icon(color='blue' if j > 0 else 'green')
                ).add_to(m)
                if j < len(sequence) - 1:
                    folium.PolyLine(
                        locations=[stops[j], stops[j+1]],
                        color="red", weight=3, popup=f"Distance: {((dist_matrix[sequence[j]][sequence[j+1]] / 100000) * 111):.2f} km"
                    ).add_to(m)
            st.subheader(f"🛣️ Route {i+1} Map")
            st_folium(m, height=500, width=900)

            route_df["Stop"] = [j + 1 for j in range(len(route_df))]
            st.dataframe(route_df[["Stop", "Society ID", "Apartment", "Latitude", "Longitude", "Orders"]])

            st.markdown(f"**Total Distance:** {total_km:.2f} km")
            st.markdown(f"**Cost per Order:** ₹{cost_per_order:.2f}")
            cost_summary.append({
                "Route": i + 1,
                "Orders": route_orders,
                "Distance (km)": total_km,
                "Cost/Order": cost_per_order
            })

            csv_export = route_df.to_csv(index=False).encode("utf-8")
            st.download_button(f"📄 Download Route {i+1} CSV", data=csv_export, file_name=f"optimized_route_{i+1}.csv", mime="text/csv")

    # --- Summary Table ---
    if cost_summary:
        summary_df = pd.DataFrame(cost_summary)
        st.subheader("📊 Cost Summary Across All Routes")
        st.dataframe(summary_df)

        all_costs_ok = summary_df["Cost/Order"].max() <= COST_TARGET_PER_ORDER
        st.success("✅ All routes optimized with cost/order ≤ ₹4") if all_costs_ok else st.warning("⚠️ Some routes exceed ₹4/order.")
