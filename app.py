
import streamlit as st
import pandas as pd
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Milk Delivery Route Optimizer", layout="wide")

st.title("🥛 Milk Delivery Route Optimizer")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of Uploaded Data")
    st.dataframe(df)

    num_vehicles = st.number_input("Number of Vehicles", min_value=1, max_value=20, value=2)
    vehicle_capacity = st.number_input("Vehicle Capacity (litres)", min_value=1, value=50)

    def parse_time(tstr):
        h, m = map(int, tstr.split(":"))
        return h * 60 + m

    locations = df[["latitude", "longitude"]].values.tolist()
    demands = df["order_qty"].tolist()
    time_windows = [(parse_time(row["start_time"]), parse_time(row["end_time"])) for _, row in df.iterrows()]

    def compute_euclidean_distance_matrix(locations):
        matrix = []
        for from_node in locations:
            row = []
            for to_node in locations:
                row.append(((from_node[0] - to_node[0])**2 + (from_node[1] - to_node[1])**2)**0.5 * 1000)
            matrix.append(row)
        return matrix

    distance_matrix = compute_euclidean_distance_matrix(locations)

    manager = pywrapcp.RoutingIndexManager(len(distance_matrix), num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        return int(distance_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)])

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    def demand_callback(from_index):
        return demands[manager.IndexToNode(from_index)]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,
        [vehicle_capacity] * num_vehicles,
        True,
        "Capacity"
    )

    def time_callback(from_index, to_index):
        return int(distance_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)] / 10)

    time_callback_index = routing.RegisterTransitCallback(time_callback)
    horizon = 150
    routing.AddDimension(
        time_callback_index,
        10,
        horizon,
        False,
        "Time"
    )
    time_dimension = routing.GetDimensionOrDie("Time")

    for i, (open_time, close_time) in enumerate(time_windows):
        index = manager.NodeToIndex(i)
        time_dimension.CumulVar(index).SetRange(open_time, close_time)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        st.success("✅ Routes optimized successfully!")
        route_data = []
        m = folium.Map(location=locations[0], zoom_start=13)

        colors = ["red", "blue", "green", "purple", "orange", "darkred", "lightblue", "gray"]

        for vehicle_id in range(num_vehicles):
            index = routing.Start(vehicle_id)
            route = []
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                time_var = time_dimension.CumulVar(index)
                time_min = solution.Min(time_var)
                route.append((node_index, time_min))
                index = solution.Value(routing.NextVar(index))
            node_index = manager.IndexToNode(index)
            time_var = time_dimension.CumulVar(index)
            time_min = solution.Min(time_var)
            route.append((node_index, time_min))
            vehicle_route = []
            for (node_id, arrival) in route:
                lat, lon = locations[node_id]
                folium.Marker(
                    location=[lat, lon],
                    popup=f"{df.iloc[node_id]['apartment_name']} (Arrives at {arrival // 60}:{arrival % 60:02})",
                    icon=folium.Icon(color=colors[vehicle_id % len(colors)])
                ).add_to(m)
                vehicle_route.append(df.iloc[node_id]["apartment_name"])
            route_data.append({"Vehicle": vehicle_id + 1, "Route": " → ".join(vehicle_route)})

        st.subheader("📌 Optimized Routes")
        st.dataframe(pd.DataFrame(route_data))

        st.subheader("🗺️ Route Map")
        st_folium(m, width=1000, height=600)

    else:
        st.error("❌ No solution found. Try adjusting constraints.")
