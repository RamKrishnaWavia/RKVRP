import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from geopy.distance import geodesic
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import pydeck as pdk

st.set_page_config(page_title="Milk Delivery Route Optimizer", layout="wide")
st.title("Milk Delivery Route Optimizer")

REQUIRED_COLUMNS = ['Society ID', 'Society Name', 'City', 'Drop Point', 'Latitude', 'Longitude', 'Orders']

def validate_input(df):
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        return False
    return True

def highlight_invalid_orders(val):
    try:
        int_val = int(val)
        return ''
    except:
        return 'background-color: red'

def create_distance_matrix(locations):
    matrix = []
    for i in locations:
        row = []
        for j in locations:
            row.append(geodesic(i, j).km)
        matrix.append(row)
    return matrix

def optimize_route(locations):
    tsp_size = len(locations)
    if tsp_size < 2:
        return [], 0

    manager = pywrapcp.RoutingIndexManager(tsp_size, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    distance_matrix = create_distance_matrix(locations)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(distance_matrix[from_node][to_node] * 1000)

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    solution = routing.SolveWithParameters(search_parameters)
    if not solution:
        return [], 0

    index = routing.Start(0)
    route = []
    route_distance = 0
    while not routing.IsEnd(index):
        node_index = manager.IndexToNode(index)
        route.append(node_index)
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)

    route.append(manager.IndexToNode(index))
    return route, route_distance / 1000

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if validate_input(df):
        df['Orders'] = pd.to_numeric(df['Orders'], errors='coerce')
        invalid_orders = df[df['Orders'].isna()]
        if not invalid_orders.empty:
            st.warning("⚠️ Some 'Orders' values are missing or non-numeric. Please fix highlighted rows.")
            st.dataframe(invalid_orders.style.applymap(highlight_invalid_orders, subset=['Orders']))
        else:
            total_orders = df['Orders'].sum()
            vehicle_capacity = 200
            num_clusters = int(np.ceil(total_orders / vehicle_capacity))

            kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(df[['Latitude', 'Longitude']])
            df['Cluster ID'] = kmeans.labels_

            all_routes = []
            total_cost = 0

            for cluster_id in df['Cluster ID'].unique():
                cluster_df = df[df['Cluster ID'] == cluster_id].reset_index(drop=True)
                locations = list(zip(cluster_df['Latitude'], cluster_df['Longitude']))
                route, dist_km = optimize_route(locations)
                cost = 35000 / max(cluster_df['Orders'].sum(), 1)
                total_cost += cost
                all_routes.append({
                    "cluster": cluster_id,
                    "route": route,
                    "distance_km": dist_km,
                    "cost_per_order": round(cost, 2)
                })

            st.success("✅ Optimization Complete")
            st.metric("Total Estimated Cost per Order (All Clusters)", f"₹{round(total_cost / len(all_routes), 2)}")

            for cluster in all_routes:
                st.subheader(f"Cluster {cluster['cluster']}")
                st.write(f"Route Distance: {cluster['distance_km']:.2f} km")
                st.write(f"Cost per Order: ₹{cluster['cost_per_order']}")

            df_display = df.copy()
            st.dataframe(df_display)

            st.download_button("Download Clustered Data", df.to_csv(index=False), "clustered_routes.csv", "text/csv")
    else:
        st.info("📄 Please upload a valid CSV file with required columns.")
else:
    st.download_button("📥 Download Template CSV", data=pd.DataFrame(columns=REQUIRED_COLUMNS).to_csv(index=False), file_name="template.csv")
