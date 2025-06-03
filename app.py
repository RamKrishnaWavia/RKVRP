import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from sklearn.cluster import KMeans
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

def haversine_distance(lat1, lon1, lat2, lon2):
    from math import radians, cos, sin, asin, sqrt
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return R * c

def compute_euclidean_distance_matrix(locations):
    size = len(locations)
    return {
        from_counter: {
            to_counter: int(haversine_distance(locations[from_counter][0], locations[from_counter][1],
                                              locations[to_counter][0], locations[to_counter][1]) * 1000)
            for to_counter in range(size)} for from_counter in range(size)}

def optimize_route(locations):
    tsp_size = len(locations)
    num_routes = 1
    depot = 0

    if tsp_size < 2:
        return list(range(tsp_size)), 0

    routing = pywrapcp.RoutingModel(tsp_size, num_routes, depot)
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    distance_matrix = compute_euclidean_distance_matrix(locations)
    def distance_callback(from_index, to_index):
        return distance_matrix[from_index][to_index]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        index = routing.Start(0)
        route = []
        route_distance = 0
        while not routing.IsEnd(index):
            route.append(routing.IndexToNode(index))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
        route.append(routing.IndexToNode(index))
        return route, route_distance / 1000
    else:
        return [], 0

def display_invalid_orders(df):
    invalid_rows = df[pd.to_numeric(df['Orders'], errors='coerce').isna()]
    if not invalid_rows.empty:
        st.error("⚠️ Some 'Orders' values are missing or non-numeric. Please check your CSV.")
        st.dataframe(invalid_rows)
        return True
    return False

st.set_page_config(page_title="Milk Route Optimizer", layout="wide")
st.title("📦 Milk Delivery Route Optimizer with OR-Tools")

vehicle_cost = st.sidebar.number_input("Vehicle Monthly Cost (₹)", value=35000)
capacity = st.sidebar.number_input("Max Orders per Vehicle", value=200)

st.markdown("### 📤 Upload Delivery CSV")
uploaded_file = st.file_uploader("Upload CSV file with Society ID, Society Name, City, Drop Point, Latitude, Longitude, Orders", type=["csv"])

sample_csv = pd.DataFrame({
    'Society ID': ['S1', 'S2'],
    'Society Name': ['Apt A', 'Apt B'],
    'City': ['Bangalore', 'Bangalore'],
    'Drop Point': ['Gate 1', 'Gate 2'],
    'Latitude': [12.935, 12.937],
    'Longitude': [77.61, 77.62],
    'Orders': [150, 100]
})
st.download_button("📄 Download Sample Template", sample_csv.to_csv(index=False), file_name="sample_template.csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    required_columns = ['Society ID', 'Society Name', 'City', 'Drop Point', 'Latitude', 'Longitude', 'Orders']
    if not all(col in df.columns for col in required_columns):
        st.error(f"Missing required columns: {required_columns}")
    else:
        if display_invalid_orders(df):
            st.stop()

        df['Orders'] = pd.to_numeric(df['Orders'])

        total_orders = df['Orders'].sum()
        num_clusters = max(1, int(np.ceil(total_orders / capacity)))
        kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(df[['Latitude', 'Longitude']])
        df['Cluster ID'] = kmeans.labels_

        st.subheader("📊 Cluster Summary")
        summaries = []
        for cluster_id in sorted(df['Cluster ID'].unique()):
            cluster_df = df[df['Cluster ID'] == cluster_id]
            cluster_orders = cluster_df['Orders'].sum()
            locations = [(row['Latitude'], row['Longitude']) for _, row in cluster_df.iterrows()]
            depot = (12.934, 77.610)  # Soukya Road
            locations.insert(0, depot)

            route, distance_km = optimize_route(locations)
            cpo = round(vehicle_cost / cluster_orders, 2) if cluster_orders > 0 else 0

            summaries.append({
                'Cluster ID': cluster_id,
                'Total Orders': cluster_orders,
                'Distance (km)': round(distance_km, 2),
                'CPO (₹)': cpo
            })

        st.dataframe(pd.DataFrame(summaries))

        st.subheader("🗺️ Cluster Routes Map")
        layers = []
        colors = [(255, 0, 0), (0, 0, 255), (0, 128, 0), (255, 140, 0), (128, 0, 128)]
        for i, cluster_id in enumerate(df['Cluster ID'].unique()):
            cluster_df = df[df['Cluster ID'] == cluster_id]
            color = colors[i % len(colors)]
            layer = pdk.Layer(
                'ScatterplotLayer',
                data=cluster_df,
                get_position='[Longitude, Latitude]',
                get_color=color,
                get_radius=100,
                pickable=True
            )
            layers.append(layer)

        st.pydeck_chart(pdk.Deck(
            map_style='mapbox://styles/mapbox/light-v9',
            initial_view_state=pdk.ViewState(
                latitude=df['Latitude'].mean(),
                longitude=df['Longitude'].mean(),
                zoom=11,
                pitch=0
            ),
            layers=layers,
            tooltip={"text": "{Society Name}\n{Drop Point}\nOrders: {Orders}"}
        ))
