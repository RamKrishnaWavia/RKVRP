import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from haversine import haversine, Unit
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import folium
from streamlit_folium import st_folium

# Constants
VEHICLE_COST = 35000  # ₹ per month
VEHICLE_ORDERS_TARGET = 200
CPO_TARGET = 4.0  # ₹ per order
MAX_RADIUS_KM_GREEN = 2
MIN_ORDERS_GREEN = 200

st.set_page_config(page_title="Route Optimizer", layout="wide")
st.title("Milk Delivery Route Optimizer")

st.markdown("""
Upload your CSV file containing delivery locations. The optimizer will:
- Automatically cluster societies into **Green** (≥200 orders, 2km radius) and **Blue** (proximity-based)
- Optimize routes using Google OR-Tools
- Show route maps, cost per order, and cluster-level stats
""")

TEMPLATE_COLUMNS = ['Society ID', 'Society Name', 'City', 'Drop Point', 'Latitude', 'Longitude', 'Orders']

with st.expander("📥 Download Input Template"):
    st.download_button("Download CSV Template", data=pd.DataFrame(columns=TEMPLATE_COLUMNS).to_csv(index=False), file_name="delivery_template.csv")

uploaded_file = st.file_uploader("Upload your delivery data CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Validate columns
    missing_cols = set(TEMPLATE_COLUMNS) - set(df.columns)
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
    else:
        # Clean and convert Orders column
        df['Orders'] = pd.to_numeric(df['Orders'], errors='coerce')
        invalid_rows = df[df['Orders'].isna()]

        if not invalid_rows.empty:
            st.warning("⚠️ Some 'Orders' values are missing or non-numeric. These rows are shown below:")
            st.dataframe(invalid_rows)
        
        df = df.dropna(subset=['Orders'])
        df['Orders'] = df['Orders'].astype(int)

        # GREEN CLUSTERS (radius ≤ 2 km and orders ≥ 200)
        def form_green_clusters(df):
            green_clusters = []
            used_ids = set()
            cluster_id = 0

            for idx, row in df.iterrows():
                if row['Society ID'] in used_ids:
                    continue
                center = (row['Latitude'], row['Longitude'])
                neighbors = []
                total_orders = 0

                for jdx, other in df.iterrows():
                    if other['Society ID'] in used_ids:
                        continue
                    point = (other['Latitude'], other['Longitude'])
                    distance = haversine(center, point, unit=Unit.KILOMETERS)
                    if distance <= MAX_RADIUS_KM_GREEN:
                        neighbors.append(other['Society ID'])
                        total_orders += other['Orders']

                if total_orders >= MIN_ORDERS_GREEN:
                    df.loc[df['Society ID'].isin(neighbors), 'Cluster Type'] = 'Green'
                    df.loc[df['Society ID'].isin(neighbors), 'Cluster ID'] = f"G{cluster_id}"
                    used_ids.update(neighbors)
                    cluster_id += 1

            return df

        df['Cluster Type'] = 'Unassigned'
        df['Cluster ID'] = None
        df = form_green_clusters(df)

        # BLUE CLUSTERS (remaining unassigned using KMeans)
        blue_df = df[df['Cluster Type'] == 'Unassigned'].copy()
        if not blue_df.empty:
            num_blue_clusters = max(1, len(blue_df) // 10)
            kmeans = KMeans(n_clusters=num_blue_clusters, random_state=42).fit(blue_df[['Latitude', 'Longitude']])
            for i, label in enumerate(kmeans.labels_):
                df.loc[blue_df.index[i], 'Cluster Type'] = 'Blue'
                df.loc[blue_df.index[i], 'Cluster ID'] = f"B{label}"

        # ROUTE OPTIMIZATION
        def create_distance_matrix(locations):
            size = len(locations)
            matrix = {}
            for from_idx in range(size):
                matrix[from_idx] = {}
                for to_idx in range(size):
                    dist = haversine(locations[from_idx], locations[to_idx], unit=Unit.KILOMETERS)
                    matrix[from_idx][to_idx] = int(dist * 1000)  # in meters
            return matrix

        def optimize_route(locations):
            tsp_size = len(locations)
            if tsp_size <= 1:
                return [0], 0.0
            num_routes = 1
            depot = 0

            dist_matrix = create_distance_matrix(locations)

            manager = pywrapcp.RoutingIndexManager(tsp_size, num_routes, depot)
            routing = pywrapcp.RoutingModel(manager)

            def distance_callback(from_index, to_index):
                return dist_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

            transit_callback_index = routing.RegisterTransitCallback(distance_callback)
            routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

            search_parameters = pywrapcp.DefaultRoutingSearchParameters()
            search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

            solution = routing.SolveWithParameters(search_parameters)
            route = []
            if solution:
                index = routing.Start(0)
                while not routing.IsEnd(index):
                    route.append(manager.IndexToNode(index))
                    index = solution.Value(routing.NextVar(index))
                route.append(manager.IndexToNode(index))
            
            distance_km = sum(
                haversine(locations[route[i]], locations[route[i+1]], unit=Unit.KILOMETERS)
                for i in range(len(route)-1)
            )
            return route, round(distance_km, 2)

        # VISUALIZE AND CALCULATE PER CLUSTER
        cluster_ids = df['Cluster ID'].dropna().unique()
        for cluster_id in cluster_ids:
            cluster_df = df[df['Cluster ID'] == cluster_id]
            locations = cluster_df[['Latitude', 'Longitude']].values.tolist()
            route, distance_km = optimize_route(locations)

            total_orders = cluster_df['Orders'].sum()
            cpo = VEHICLE_COST / max(total_orders, 1)

            st.subheader(f"📦 Cluster {cluster_id} ({cluster_df['Cluster Type'].iloc[0]})")
            st.write(f"Total Orders: {total_orders}, Route Distance: {distance_km} km, Cost Per Order: ₹{cpo:.2f}")

            # MAP
            route_df = cluster_df.iloc[route]
            m = folium.Map(location=locations[0], zoom_start=14)
            for i, row in route_df.iterrows():
                folium.Marker([row['Latitude'], row['Longitude']], tooltip=row['Society Name']).add_to(m)
            folium.PolyLine(locations).add_to(m)
            st_folium(m, width=700, height=500)

        st.success("✅ Route optimization completed for all clusters.")
