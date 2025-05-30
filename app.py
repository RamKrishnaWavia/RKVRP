import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import math

# Page setup
st.set_page_config(page_title="RK - Delivery Route Optimizer", layout="wide")
st.title("RK - Delivery Route Optimizer")
st.markdown("Optimize early morning deliveries (4 AM to 7 AM) with cost-efficient routing.")

# CSV Template Download
if st.button("📅 Download CSV Template"):
    template = pd.DataFrame({
        "Society ID": [101, 102],
        "Apartment": ["ABC Residency", "Green Heights"],
        "Latitude": [12.935, 12.938],
        "Longitude": [77.614, 77.610],
        "Orders": [210, 300]
    })
    csv = template.to_csv(index=False).encode('utf-8')
    st.download_button("Download Template", data=csv, file_name="milk_delivery_template.csv", mime='text/csv')

# User input for vehicle cost
vehicle_monthly_cost = st.number_input("Enter Vehicle Monthly Cost (₹)", value=35000, step=1000)
daily_vehicle_cost = vehicle_monthly_cost / 30

# Upload CSV
uploaded_file = st.file_uploader("Upload Delivery Data CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    required_cols = ["Society ID", "Apartment", "Latitude", "Longitude", "Orders"]
    if not all(col in df.columns for col in required_cols):
        st.error(f"Uploaded file must contain columns: {', '.join(required_cols)}")
        st.stop()

    locations = list(zip(df["Latitude"], df["Longitude"]))
    num_locations = len(locations)
    max_orders_per_route = 200

    # Distance matrix calculation
    def compute_distance_matrix(locations):
        distances = [[0] * len(locations) for _ in locations]
        for i in range(len(locations)):
            for j in range(len(locations)):
                if i != j:
                    dx = locations[i][0] - locations[j][0]
                    dy = locations[i][1] - locations[j][1]
                    distances[i][j] = math.sqrt(dx * dx + dy * dy)
        return distances

    distance_matrix = compute_distance_matrix(locations)

    # Create multiple routes based on max orders
    df = df.sort_values("Orders", ascending=False).reset_index(drop=True)
    routes = []
    used = [False] * len(df)

    for _ in range(len(df)):
        current_route = []
        current_orders = 0
        for idx, row in df.iterrows():
            if not used[idx] and current_orders + row["Orders"] <= max_orders_per_route:
                current_route.append(idx)
                current_orders += row["Orders"]
                used[idx] = True
        if current_route:
            routes.append(current_route)

    st.subheader("📦 Route Summary")
    total_vehicles_used = len(routes)
    all_routes_data = []

    for route_num, route_indices in enumerate(routes):
        st.markdown(f"### 🚚 Route {route_num + 1}")
        route_df = df.loc[route_indices].reset_index(drop=True)

        # Simple Nearest Neighbor ordering for the route
        ordered = [0]
        visited = [False] * len(route_df)
        visited[0] = True
        for _ in range(1, len(route_df)):
            last = ordered[-1]
            nearest = None
            min_dist = float('inf')
            for j in range(len(route_df)):
                if not visited[j]:
                    dist = math.sqrt((route_df.loc[last, "Latitude"] - route_df.loc[j, "Latitude"])**2 +
                                     (route_df.loc[last, "Longitude"] - route_df.loc[j, "Longitude"])**2)
                    if dist < min_dist:
                        min_dist = dist
                        nearest = j
            ordered.append(nearest)
            visited[nearest] = True

        ordered_df = route_df.loc[ordered].reset_index(drop=True)
        ordered_df["Stop"] = ordered_df.index + 1
        ordered_df["Distance from Previous (km)"] = 0.0
        total_km = 0.0

        for i in range(1, len(ordered_df)):
            prev = ordered_df.loc[i - 1]
            curr = ordered_df.loc[i]
            dx = prev["Latitude"] - curr["Latitude"]
            dy = prev["Longitude"] - curr["Longitude"]
            dist_km = math.sqrt(dx * dx + dy * dy) * 111
            ordered_df.at[i, "Distance from Previous (km)"] = dist_km
            total_km += dist_km

        total_orders = ordered_df["Orders"].sum()
        route_cost = daily_vehicle_cost
        cost_per_order = route_cost / total_orders

        st.dataframe(ordered_df[["Stop", "Society ID", "Apartment", "Orders", "Distance from Previous (km)"]])

        st.markdown(f"- **Total Orders:** {total_orders}")
        st.markdown(f"- **Total Distance:** {total_km:.2f} km")
        st.markdown(f"- **Vehicle Cost for Route:** ₹{route_cost:.2f}")
        st.markdown(f"- **Cost per Order:** ₹{cost_per_order:.2f}")

        if cost_per_order > 4:
            st.error(f"⚠️ Cost per order exceeds ₹4 on Route {route_num + 1}!")
        if total_orders < 200:
            st.warning(f"⚠️ Less than 200 orders in Route {route_num + 1}.")

        # Draw map
        m = folium.Map(location=[ordered_df["Latitude"].mean(), ordered_df["Longitude"].mean()], zoom_start=13)
        for i, row in ordered_df.iterrows():
            popup = f"Stop {i+1}: {row['Apartment']} (ID: {row['Society ID']}, Orders: {row['Orders']}, Dist: {row['Distance from Previous (km)']:.2f} km)"
            folium.Marker(
                location=[row["Latitude"], row["Longitude"]],
                popup=popup,
                icon=folium.Icon(color='blue' if i > 0 else 'green')
            ).add_to(m)
        folium.PolyLine(
            locations=[(row["Latitude"], row["Longitude"]) for _, row in ordered_df.iterrows()],
            color="red", weight=3
        ).add_to(m)
        st_folium(m, height=450, width=900)

        all_routes_data.append(ordered_df)

    st.success(f"✅ Optimization complete! Total Vehicles Used: {total_vehicles_used}")
