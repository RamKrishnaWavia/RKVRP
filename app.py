import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from geopy.geocoders import Nominatim

# Constants
DEPOT_ADDRESS = "Soukya Road, Bangalore"
MAX_ROUTE_DURATION = 3 * 60 * 60  # 3 hours max route time (4 AM to 7 AM)

st.set_page_config(page_title="RK - Delivery Route Optimizer", layout="wide")
st.title("RK - Delivery Route Optimizer")

st.markdown(
    """
    Optimize milk deliveries with:
    - Depot fixed at Soukya Road
    - Minimum 200 orders per route
    - Target cost per order ≤ ₹4
    - Editable vehicle monthly cost (default ₹35,000)
    """
)

# Download CSV template
if st.button("Download CSV Template"):
    template = pd.DataFrame({
        "Society ID": [101, 102],
        "Society Name": ["ABC Residency", "Green Heights"],
        "Latitude": [12.935, 12.938],
        "Longitude": [77.614, 77.610],
        "Orders": [10, 20]
    })
    st.download_button("Download Template CSV", data=template.to_csv(index=False), file_name="milk_delivery_template.csv", mime='text/csv')

# User inputs for cost and capacity
vehicle_monthly_cost = st.number_input("Enter vehicle monthly cost (₹)", value=35000, step=1000)
max_orders_per_vehicle = st.number_input("Enter minimum orders per vehicle route (min 200)", value=200, min_value=200)
target_cost_per_order = 4.0  # fixed target

# Geocode depot location once
@st.cache_data(show_spinner=False)
def geocode_address(address):
    geolocator = Nominatim(user_agent="milk_optimizer")
    location = geolocator.geocode(address)
    if location:
        return (location.latitude, location.longitude)
    else:
        return None

depot_location = geocode_address(DEPOT_ADDRESS)
if depot_location is None:
    st.error(f"Could not geocode depot address: {DEPOT_ADDRESS}")
    st.stop()

uploaded_file = st.file_uploader("Upload delivery CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Check required columns
    required_cols = {"Society ID", "Society Name", "Latitude", "Longitude", "Orders"}
    if not required_cols.issubset(df.columns):
        st.error(f"CSV must contain columns: {required_cols}")
        st.stop()

    # Add depot row at start
    depot_row = pd.DataFrame([{
        "Society ID": 0,
        "Society Name": "Depot (Soukya Road)",
        "Latitude": depot_location[0],
        "Longitude": depot_location[1],
        "Orders": 0
    }])
    df = pd.concat([depot_row, df], ignore_index=True)

    # Prepare data model for OR-Tools
    locations = list(zip(df["Latitude"], df["Longitude"]))
    orders = df["Orders"].tolist()

    # Distance matrix (Euclidean)
    def compute_distance_matrix(locations):
        size = len(locations)
        matrix = []
        for from_idx in range(size):
            row = []
            for to_idx in range(size):
                if from_idx == to_idx:
                    row.append(0)
                else:
                    dist = ((locations[from_idx][0] - locations[to_idx][0]) ** 2 + (locations[from_idx][1] - locations[to_idx][1]) ** 2) ** 0.5
                    row.append(dist)
            matrix.append(row)
        return matrix

    distance_matrix = compute_distance_matrix(locations)

    # Create routing model
    manager = pywrapcp.RoutingIndexManager(len(locations), len(locations)-1, 0)  # Max vehicles = total points-1, depot=0
    routing = pywrapcp.RoutingModel(manager)

    # Distance callback
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        # convert distance degrees to meters roughly by multiplying by 111000
        return int(distance_matrix[from_node][to_node] * 111000)

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add capacity constraint (orders)
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return orders[from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        [max_orders_per_vehicle] * (len(locations) - 1),  # vehicle capacities (max orders per vehicle)
        True,  # start cumul to zero
        "Capacity"
    )

    # Add time window constraints (4 AM to 7 AM = 3 hours, converted to seconds)
    time_per_meter = 0.5  # example: 0.5 sec per meter driving time (adjust as needed)

    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        dist = distance_matrix[from_node][to_node] * 111000  # meters
        return int(dist * time_per_meter)

    time_callback_index = routing.RegisterTransitCallback(time_callback)

    routing.AddDimension(
        time_callback_index,
        30 * 60,  # 30 min slack
        MAX_ROUTE_DURATION,
        False,
        "Time"
    )
    time_dimension = routing.GetDimensionOrDie("Time")
    # Set time window for depot and deliveries
    depot_start = 0
    depot_end = MAX_ROUTE_DURATION
    for location_idx in range(len(locations)):
        index = manager.NodeToIndex(location_idx)
        time_dimension.CumulVar(index).SetRange(depot_start, depot_end)

    # Setting first solution heuristic
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.time_limit.seconds = 30

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        routes = []
        total_cost = 0
        vehicle_count = 0

        m = folium.Map(location=depot_location, zoom_start=12)
        folium.Marker(location=depot_location, popup="Depot (Soukya Road)", icon=folium.Icon(color='green', icon='home')).add_to(m)

        # Extract routes
        for vehicle_id in range(len(locations)-1):
            index = routing.Start(vehicle_id)
            if routing.IsEnd(solution.Value(routing.NextVar(index))):
                # No route for this vehicle
                continue
            route = []
            route_distance = 0
            route_orders = 0

            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route.append(node_index)
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                if not routing.IsEnd(index):
                    route_distance += distance_matrix[node_index][manager.IndexToNode(index)] * 111  # km approx

            # Calculate route cost and cost per order
            route_orders = sum(orders[i] for i in route)
            if route_orders < max_orders_per_vehicle:
                continue  # enforce min orders per route

            vehicle_count += 1
            route_cost = vehicle_monthly_cost  # per vehicle monthly cost
            cost_per_order = route_cost / route_orders

            # Only keep routes with cost per order <= target
            if cost_per_order > target_cost_per_order:
                continue

            total_cost += route_cost

            # Add markers and polyline on map
            route_coords = []
            for i, stop_idx in enumerate(route):
                lat, lon = locations[stop_idx]
                route_coords.append((lat, lon))
                popup_text = f"Stop {i+1}: {df.at[stop_idx, 'Society Name']}<br>Orders: {orders[stop_idx]}"
                if i > 0:
                    prev_lat, prev_lon = locations[route[i-1]]
                    dist_km = ((lat - prev_lat) ** 2 + (lon - prev_lon) ** 2) ** 0.5 * 111
                    popup_text += f"<br>Distance from prev stop: {dist_km:.2f} km"
                folium.Marker(
                    location=(lat, lon),
                    popup=popup_text,
                    icon=folium.Icon(color='blue' if stop_idx != 0 else 'green', icon='truck' if stop_idx != 0 else 'home')
                ).add_to(m)

            folium.PolyLine(route_coords, color="red", weight=3).add_to(m)

            routes.append({
                "Vehicle": vehicle_id + 1,
                "Route Stops": [df.at[i, "Society Name"] for i in route],
                "Orders": route_orders,
                "Distance (km)": round(route_distance, 2),
                "Cost (₹)": route_cost,
                "Cost per Order (₹)": round(cost_per_order, 2)
            })

        st.subheader("Optimized Routes Map")
        st_folium(m, height=600)

        if not routes:
            st.warning("No feasible routes found with given constraints.")
        else:
            st.subheader("Route Summary")
            routes_df = pd.DataFrame(routes)
            st.dataframe(routes_df)

            st.subheader("Cost Summary")
            st.markdown(f"- **Vehicles used:** {vehicle_count}")
            st.markdown(f"- **Total monthly cost:** ₹{total_cost}")
            st.markdown(f"- **Target cost per order:** ₹{target_cost_per_order}")

            # Download combined route data CSV
            detailed_rows = []
            for r in routes:
                vehicle_num = r["Vehicle"]
                stops = r["Route Stops"]
                for seq, stop in enumerate(stops, 1):
                    detailed_rows.append({"Vehicle": vehicle_num, "Stop Sequence": seq, "Society Name": stop})

            detailed_df = pd.DataFrame(detailed_rows)
            csv_bytes = detailed_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Detailed Route CSV", data=csv_bytes, file_name="optimized_routes.csv", mime='text/csv')

    else:
        st.error("No solution found within time limit. Try adjusting constraints or input data.")
