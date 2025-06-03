import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from sklearn.cluster import KMeans
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from geopy.distance import geodesic

st.set_page_config(layout="wide")
st.title("Milk Delivery Route Optimizer 🚚🥛")

TEMPLATE_COLUMNS = ['Society ID', 'Society Name', 'City', 'Drop Point', 'Latitude', 'Longitude', 'Orders']

with st.expander("📥 Download CSV Template"):
    st.download_button(
        label="Download Template",
        data=pd.DataFrame(columns=TEMPLATE_COLUMNS).to_csv(index=False),
        file_name='route_input_template.csv',
        mime='text/csv'
    )

def load_data():
    uploaded_file = st.file_uploader("Upload Order CSV", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        missing_cols = [col for col in TEMPLATE_COLUMNS if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            return None

        df['Invalid Orders'] = False
        df['Orders'] = pd.to_numeric(df['Orders'], errors='coerce')
        df['Invalid Orders'] = df['Orders'].isna()

        if df['Invalid Orders'].any():
            st.warning("⚠️ Some 'Orders' values are missing or non-numeric. Please check highlighted rows below.")
            st.dataframe(df[df['Invalid Orders']])
            return None

        df['Orders'] = df['Orders'].astype(int)
        return df
    return None

def haversine_distance(coord1, coord2):
    return geodesic(coord1, coord2).km

def create_distance_matrix(locations):
    size = len(locations)
    matrix = {}
    for from_idx in range(size):
        matrix[from_idx] = {}
        for to_idx in range(size):
            if from_idx == to_idx:
                matrix[from_idx][to_idx] = 0
            else:
                coord1 = (locations[from_idx]['lat'], locations[from_idx]['lon'])
                coord2 = (locations[to_idx]['lat'], locations[to_idx]['lon'])
                matrix[from_idx][to_idx] = int(haversine_distance(coord1, coord2) * 1000)  # meters
    return matrix

def optimize_route(locations):
    tsp_size = len(locations)
    if tsp_size < 2:
        return [], 0

    num_routes = 1
    depot = 0

    distance_matrix = create_distance_matrix(locations)

    manager = pywrapcp.RoutingIndexManager(tsp_size, num_routes, depot)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    solution = routing.SolveWithParameters(search_parameters)
    if solution:
        route = []
        index = routing.Start(0)
        route_distance = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route.append(node_index)
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
        route.append(manager.IndexToNode(index))
        return route, route_distance / 1000  # km
    else:
        return [], 0

def assign_clusters(df, capacity=200):
    total_orders = df['Orders'].sum()
    num_clusters = max(1, int(np.ceil(total_orders / capacity)))
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    coords = df[['Latitude', 'Longitude']]
    df['Cluster ID'] = kmeans.fit_predict(coords)
    return df

def main():
    df = load_data()
    if df is None:
        return

    vehicle_cost = st.number_input("Enter Monthly Vehicle Cost (₹)", value=35000, min_value=0)

    df = assign_clusters(df)

    results = []
    for cluster_id in sorted(df['Cluster ID'].unique()):
        cluster_df = df[df['Cluster ID'] == cluster_id].reset_index(drop=True)
        cluster_locations = [
            {'lat': row['Latitude'], 'lon': row['Longitude'], 'name': row['Society Name'], 'orders': row['Orders']}
            for _, row in cluster_df.iterrows()
        ]
        if len(cluster_locations) < 2:
            continue
        route, distance_km = optimize_route(cluster_locations)
        total_orders = cluster_df['Orders'].sum()
        cost_per_order = round(vehicle_cost / total_orders, 2) if total_orders else 0

        for i, idx in enumerate(route):
            cluster_df.loc[idx, 'Route Sequence'] = i + 1
        cluster_df['Route Distance (km)'] = distance_km
        cluster_df['Cost per Order (₹)'] = cost_per_order
        results.append(cluster_df)

    if results:
        final_df = pd.concat(results).sort_values(by=['Cluster ID', 'Route Sequence'])
        st.success("✅ Optimized routes generated!")
        st.dataframe(final_df)

        st.download_button("📤 Download Optimized Routes", final_df.to_csv(index=False), "optimized_routes.csv")

        selected_city = st.selectbox("Filter by City", ["All"] + sorted(final_df['City'].unique()))
        if selected_city != "All":
            final_df = final_df[final_df['City'] == selected_city]

        chart_df = final_df.dropna(subset=['Latitude', 'Longitude'])
        st.pydeck_chart(pdk.Deck(
            initial_view_state=pdk.ViewState(
                latitude=chart_df['Latitude'].mean(),
                longitude=chart_df['Longitude'].mean(),
                zoom=11,
                pitch=0
            ),
            layers=[
                pdk.Layer(
                    'ScatterplotLayer',
                    data=chart_df,
                    get_position='[Longitude, Latitude]',
                    get_fill_color='[200, 30, 0, 160]',
                    get_radius=100,
                    pickable=True,
                )
            ],
            tooltip={
                "html": "<b>{Society Name}</b><br/>Cluster: {Cluster ID}<br/>Route Seq: {Route Sequence}<br/>CPO: ₹{Cost per Order (₹)}",
                "style": {"color": "white"}
            }
        ))

if __name__ == '__main__':
    main()
