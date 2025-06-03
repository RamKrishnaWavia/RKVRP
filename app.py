import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from haversine import haversine, Unit
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import folium
from streamlit_folium import st_folium

st.set_page_config(layout="wide")

# Constants
VEHICLE_CAPACITY = 200  # orders per vehicle max capacity
VEHICLE_MONTHLY_COST = 35000  # ₹
DELIVERY_DAYS_PER_MONTH = 30
MAX_GREEN_CLUSTER_RADIUS_KM = 2

def haversine_distance_matrix(locations):
    size = len(locations)
    dist_matrix = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            if i == j:
                dist_matrix[i][j] = 0
            else:
                dist_matrix[i][j] = int(haversine(locations[i], locations[j], unit=Unit.METERS))
    return dist_matrix

def optimize_route(locations):
    tsp_size = len(locations)
    if tsp_size <= 1:
        return list(range(tsp_size)), 0.0

    dist_matrix = haversine_distance_matrix(locations)
    manager = pywrapcp.RoutingIndexManager(tsp_size, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return dist_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    solution = routing.SolveWithParameters(search_parameters)
    if not solution:
        return None, 0.0

    index = routing.Start(0)
    route = []
    route_distance = 0
    while not routing.IsEnd(index):
        route.append(manager.IndexToNode(index))
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
    route.append(manager.IndexToNode(index))

    return route, route_distance / 1000  # meters to km

def cluster_data(df):
    # Convert lat-lon to radians for haversine DBSCAN
    coords = np.radians(df[['Latitude', 'Longitude']])
    # DBSCAN eps in radians (2 km for green cluster)
    eps_rad = MAX_GREEN_CLUSTER_RADIUS_KM / 6371.0088  
    db = DBSCAN(eps=eps_rad, min_samples=1, algorithm='ball_tree', metric='haversine').fit(coords)

    df['DBSCAN_Cluster'] = db.labels_
    clusters = []

    # For each DBSCAN cluster, check total orders to separate green vs blue
    for cluster_id in sorted(df['DBSCAN_Cluster'].unique()):
        cluster_df = df[df['DBSCAN_Cluster'] == cluster_id]
        total_orders = cluster_df['Orders'].sum()
        # Green cluster: total orders <= VEHICLE_CAPACITY and radius <= 2km
        center_lat = cluster_df['Latitude'].mean()
        center_lon = cluster_df['Longitude'].mean()

        max_dist_km = cluster_df.apply(
            lambda row: haversine((center_lat, center_lon), (row['Latitude'], row['Longitude'])), axis=1).max()

        if total_orders <= VEHICLE_CAPACITY and max_dist_km <= MAX_GREEN_CLUSTER_RADIUS_KM:
            cluster_type = 'Green'
        else:
            cluster_type = 'Blue'
        
        clusters.append({
            'cluster_id': cluster_id,
            'type': cluster_type,
            'data': cluster_df,
            'total_orders': total_orders,
            'center': (center_lat, center_lon),
            'max_dist_km': max_dist_km
        })
    return clusters

def main():
    st.title("Optimized Multi-Cluster Route Planner (Green & Blue Clusters)")

    uploaded_file = st.file_uploader("Upload CSV with columns: Society ID, Society Name, City, Drop Point, Latitude, Longitude, Orders", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        required_cols = ['Society ID', 'Society Name', 'City', 'Drop Point', 'Latitude', 'Longitude', 'Orders']
        if not all(col in df.columns for col in required_cols):
            st.error(f"CSV must contain these columns: {required_cols}")
            return

        # Validate Orders column numeric & non-null
        df['Orders'] = pd.to_numeric(df['Orders'], errors='coerce')
        invalid_orders = df[df['Orders'].isna()]
        if not invalid_orders.empty:
            st.warning("Some rows have invalid or missing 'Orders'. Highlighting:")
            st.dataframe(invalid_orders.style.apply(lambda x: ['background-color: pink']*len(x), axis=1))
            return

        clusters = cluster_data(df)

        st.markdown(f"### Total Clusters Found: {len(clusters)}")
        total_orders_all = df['Orders'].sum()
        st.markdown(f"### Total Orders: {total_orders_all}")

        overall_map = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=11)
        cluster_results = []

        for c in clusters:
            st.markdown(f"---\n### Cluster {c['cluster_id']} ({c['type']})")
            st.markdown(f"- Total Orders: {c['total_orders']}")
            st.markdown(f"- Max Distance from Center: {c['max_dist_km']:.2f} km")

            cluster_df = c['data'].reset_index(drop=True)

            locs = cluster_df[['Latitude', 'Longitude']].values.tolist()
            route, dist_km = optimize_route(locs)
            if route is None:
                st.warning("Route optimization failed for this cluster.")
                continue

            cpo = (VEHICLE_MONTHLY_COST / DELIVERY_DAYS_PER_MONTH) / c['total_orders'] if c['total_orders'] > 0 else 0

            st.markdown(f"- Optimized Route Distance: {dist_km:.2f} km")
            st.markdown(f"- Cost Per Order (CPO): ₹{cpo:.2f}")

            # Show route sequence
            route_ordered = cluster_df.iloc[route]
            st.dataframe(route_ordered[['Society ID', 'Society Name', 'Orders', 'Latitude', 'Longitude']])

            # Map cluster with route lines
            m = folium.Map(location=c['center'], zoom_start=13)
            folium.Marker(location=c['center'], tooltip=f"Cluster {c['cluster_id']} Center").add_to(m)

            # Draw points
            for idx, row in cluster_df.iterrows():
                folium.CircleMarker(
                    location=(row['Latitude'], row['Longitude']),
                    radius=5,
                    tooltip=f"{row['Society Name']} ({row['Orders']} orders)",
                    color='green' if c['type'] == 'Green' else 'blue',
                    fill=True,
                    fill_opacity=0.7,
                ).add_to(m)

            # Draw route lines
            route_points = [(cluster_df.iloc[i]['Latitude'], cluster_df.iloc[i]['Longitude']) for i in route]
            folium.PolyLine(route_points, color='red', weight=3, opacity=0.8).add_to(m)

            st_folium(m, width=700, height=450)

            cluster_results.append({
                'cluster_id': c['cluster_id'],
                'type': c['type'],
                'total_orders': c['total_orders'],
                'route_distance_km': dist_km,
                'cost_per_order': cpo,
            })

        # Summary table
        summary_df = pd.DataFrame(cluster_results)
        st.markdown("### Cluster Summary")
        st.dataframe(summary_df)

if __name__ == "__main__":
    main()
