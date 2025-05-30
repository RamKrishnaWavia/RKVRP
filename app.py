import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
from geopy.geocoders import Nominatim

# Fixed depot coordinates (Soukya Road)
DEPOT_LAT = 13.0833   # Replace with actual Soukya Road lat
DEPOT_LON = 77.6253   # Replace with actual Soukya Road lon

st.title("RK - Delivery Route Optimizer")

# User inputs vehicle monthly cost and max orders per vehicle (default 200)
vehicle_monthly_cost = st.number_input("Enter monthly cost per vehicle (₹)", value=35000, min_value=1000)
max_orders_per_vehicle = st.number_input("Enter max orders per vehicle (minimum 200)", value=200, min_value=200)

st.markdown("### Download CSV Template")
template = pd.DataFrame({
    "Society ID": [101, 102],
    "Society Name": ["ABC Residency", "Green Heights"],
    "Latitude": [12.935, 12.938],
    "Longitude": [77.614, 77.610],
    "Orders": [10, 20]
})
csv_template = template.to_csv(index=False).encode('utf-8')
st.download_button("Download Template", data=csv_template, file_name="milk_delivery_template.csv", mime='text/csv')

uploaded_file = st.file_uploader("Upload delivery data CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Add depot as first row
    depot_row = pd.DataFrame([{
        "Society ID": 0,
        "Society Name": "Soukya Road Depot",
        "Latitude": DEPOT_LAT,
        "Longitude": DEPOT_LON,
        "Orders": 0
    }])
    df = pd.concat([depot_row, df], ignore_index=True)

    # Prepare data for OR-tools
    locations = list(zip(df["Latitude"], df["Longitude"]))
    demands = df["Orders"].tolist()

    def compute_euclidean_distance_matrix(locations):
        distances = {}
        for i, from_node in enumerate(locations):
            distances[i] = {}
            for j, to_node in enumerate(locations):
                distances[i][j] = int(
                    (((from_node[0] - to_node[0])**2 + (from_node[1] - to_node[1])**2)**0.5) * 111000
                )  # meters approx
        return distances

    distance_matrix = compute_euclidean_distance_matrix(locations)

    # Setup OR-Tools data model
    data = {}
    data['distance_matrix'] = distance_matrix
    data['demands'] = demands
    data['vehicle_capacities'] = [max_orders_per_vehicle] * len(df)  # large max, will reduce later
    data['num_vehicles'] = len(df)  # max possible (one vehicle per location)
    data['depot'] = 0

    manager = pywrapcp.RoutingIndexManager(len(distance_matrix), data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add demand constraint
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data['vehicle_capacities'],
        True,  # start cumul to zero
        "Capacity"
    )

    # Setting first solution heuristic
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        used_vehicles = 0
        routes = []
        total_distance_all = 0
        total_orders_all = sum(demands)

        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            route_orders = 0
            if routing.IsEnd(solution.Value(routing.NextVar(index))):
                # vehicle not used
                continue
            used_vehicles += 1
            route = []
            route_distance = 0
            prev_index = index
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route.append(node_index)
                next_index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(prev_index, next_index, vehicle_id)
                prev_index = next_index
                index = next_index
            route.append(manager.IndexToNode(index))  # depot end
            route_orders = sum(demands[n] for n in route)
            cost = vehicle_monthly_cost  # monthly cost per vehicle for now
            cost_per_order = cost / route_orders if route_orders > 0 else 0

            # Filter routes not satisfying order or cost constraints
            if route_orders < max_orders_per_vehicle or cost_per_order > 4:
                continue

            total_distance_all += route_distance
            routes.append({
                "vehicle_id": vehicle_id + 1,
                "route": route,
                "orders": route_orders,
                "distance_km": route_distance / 1000,
                "cost": cost,
                "cost_per_order": cost_per_order
            })

        if not routes:
            st.error("No feasible routes found with the given constraints (min 200 orders and cost/order ≤ ₹4). Try adjusting parameters.")
        else:
            st.success(f"Optimized {len(routes)} routes with total orders: {total_orders_all}")
            for r in routes:
                st.subheader(f"Route for Vehicle {r['vehicle_id']}")
                route_df = df.iloc[r['route']][["Society ID", "Society Name", "Latitude", "Longitude", "Orders"]].copy()
                route_df["Stop Sequence"] = range(1, len(route_df) + 1)
                st.dataframe(route_df)

                # Map Visualization
                m = folium.Map(location=[DEPOT_LAT, DEPOT_LON], zoom_start=12)
                for i, stop in enumerate(r['route']):
                    loc = (df.loc[stop, "Latitude"], df.loc[stop, "Longitude"])
                    popup_text = f"Stop {i+1}: {df.loc[stop, 'Society Name']}<br>Orders: {df.loc[stop, 'Orders']}"
                    folium.Marker(loc, popup=popup_text,
                                  icon=folium.Icon(color="green" if i == 0 else "blue")).add_to(m)
                folium.PolyLine([ (df.loc[stop, "Latitude"], df.loc[stop, "Longitude"]) for stop in r['route']],
                                color="red", weight=3).add_to(m)
                st_folium(m, height=400)

                st.markdown(f"**Route distance:** {r['distance_km']:.2f} km")
                st.markdown(f"**Vehicle monthly cost:** ₹{r['cost']}")
                st.markdown(f"**Cost per order:** ₹{r['cost_per_order']:.2f}")

            # Download CSV with all routes flattened
            all_routes = []
            for r in routes:
                route_df = df.iloc[r['route']][["Society ID", "Society Name", "Latitude", "Longitude", "Orders"]].copy()
                route_df["Vehicle ID"] = r["vehicle_id"]
                route_df["Stop Sequence"] = range(1, len(route_df) + 1)
                all_routes.append(route_df)
            all_routes_df = pd.concat(all_routes, ignore_index=True)

            csv_out = all_routes_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Optimized Routes CSV", data=csv_out, file_name="optimized_routes.csv", mime="text/csv")
    else:
        st.error("No solution found. Please check input data and constraints.")
