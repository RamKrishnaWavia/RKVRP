import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
from sklearn.cluster import KMeans
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from haversine import haversine, Unit
import base64

# Configuration
VEHICLE_CAPACITY = 200
VEHICLE_MONTHLY_COST = 35000
DAYS_IN_MONTH = 30

# Function to generate template CSV
def generate_template():
    data = {
        'Society ID': ['S1', 'S2'],
        'Society Name': ['Society A', 'Society B'],
        'City': ['CityX', 'CityY'],
        'Drop Point': ['Point A', 'Point B'],
        'Latitude': [12.9716, 12.2958],
        'Longitude': [77.5946, 76.6394],
        'Orders': [120, 80]
    }
    df = pd.DataFrame(data)
    return df

# Haversine distance matrix
def create_distance_matrix(locations):
    dist_matrix = np.zeros((len(locations), len(locations)))
    for i, loc1 in enumerate(locations):
        for j, loc2 in enumerate(locations):
            if i != j:
                dist_matrix[i][j] = haversine(loc1, loc2, unit=Unit.KILOMETERS)
    return dist_matrix

# Optimize route using OR-Tools
def optimize_route(locations):
    tsp_size = len(locations)
    num_routes = 1
    depot = 0
    manager = pywrapcp.RoutingIndexManager(tsp_size, num_routes, depot)
    routing = pywrapcp.RoutingModel(manager)

    dist_matrix = create_distance_matrix(locations)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(dist_matrix[from_node][to_node] * 1000)

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    solution = routing.SolveWithParameters(search_params)

    if solution:
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
    else:
        return [], 0

# Main App
st.title("Milk Delivery Route Optimizer")

with st.expander("📥 Download Template CSV"):
    template_df = generate_template()
    csv = template_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="milk_delivery_template.csv">Download Template File</a>'
    st.markdown(href, unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Delivery Data CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    required_cols = ['Society ID', 'Society Name', 'City', 'Drop Point', 'Latitude', 'Longitude', 'Orders']
    if not all(col in df.columns for col in required_cols):
        st.error(f"Missing required columns: {required_cols}")
    else:
        df['Invalid Orders'] = df['Orders'].apply(lambda x: pd.isna(x) or not str(x).isdigit())
        if df['Invalid Orders'].any():
            st.warning("⚠️ Some 'Orders' values are missing or non-numeric. Please check highlighted rows.")
            st.dataframe(df[df['Invalid Orders']])
        else:
            df['Orders'] = df['Orders'].astype(int)
            df['Cluster ID'] = -1

            # Calculate number of clusters based on order volume and vehicle capacity
            total_orders = df['Orders'].sum()
            num_clusters = max(1, int(np.ceil(total_orders / VEHICLE_CAPACITY)))
            coords = df[['Latitude', 'Longitude']].values
            kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(coords)
            df['Cluster ID'] = kmeans.labels_

            # Map for visualization
            m = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=11)
            cluster_costs = {}

            for cluster_id in sorted(df['Cluster ID'].unique()):
                cluster_df = df[df['Cluster ID'] == cluster_id].copy()
                cluster_locations = list(zip(cluster_df['Latitude'], cluster_df['Longitude']))
                route, distance_km = optimize_route(cluster_locations)

                route_societies = cluster_df.iloc[route]
                total_orders = route_societies['Orders'].sum()
                cost_per_order = (VEHICLE_MONTHLY_COST / DAYS_IN_MONTH) / max(total_orders, 1)
                cluster_costs[cluster_id] = round(cost_per_order, 2)

                for i, row in route_societies.iterrows():
                    popup = (f"Society: {row['Society Name']}<br>City: {row['City']}<br>"
                             f"Drop Point: {row['Drop Point']}<br>Cluster ID: {cluster_id}<br>"
                             f"CPO: ₹{round(cost_per_order, 2)}")
                    folium.CircleMarker(
                        location=(row['Latitude'], row['Longitude']),
                        radius=5,
                        color='blue',
                        fill=True,
                        fill_color='blue',
                        popup=popup
                    ).add_to(m)

            st.subheader("🗺️ Cluster Map with Cost Per Order")
            folium_static(m)

            st.subheader("📊 Cost Per Order by Cluster")
            cost_df = pd.DataFrame([{'Cluster ID': k, 'Cost Per Order (₹)': v} for k, v in cluster_costs.items()])
            st.dataframe(cost_df)

            st.subheader("🔍 Filter Data")
            city = st.selectbox("City", options=["All"] + sorted(df['City'].unique().tolist()))
            society = st.selectbox("Society Name", options=["All"] + sorted(df['Society Name'].unique().tolist()))
            drop_point = st.selectbox("Drop Point", options=["All"] + sorted(df['Drop Point'].unique().tolist()))

            filtered_df = df.copy()
            if city != "All":
                filtered_df = filtered_df[filtered_df['City'] == city]
            if society != "All":
                filtered_df = filtered_df[filtered_df['Society Name'] == society]
            if drop_point != "All":
                filtered_df = filtered_df[filtered_df['Drop Point'] == drop_point]

            st.dataframe(filtered_df.drop(columns=['Invalid Orders']))
