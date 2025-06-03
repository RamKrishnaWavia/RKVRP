import streamlit as st
import pandas as pd
import numpy as np
import math
from sklearn.cluster import KMeans
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from haversine import haversine
import pydeck as pdk

# Constants
DEFAULT_VEHICLE_COST = 1200
MIN_ORDERS_GREEN = 200

st.set_page_config(layout="wide")
st.title("Optimized Route Clustering with Cost Per Order")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
min_orders_threshold = st.number_input("Minimum Orders for Green Clusters", value=200, step=10)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    required_columns = ['Society ID', 'Society Name', 'City', 'Drop Point', 'Latitude', 'Longitude', 'Orders']

    if not all(col in df.columns for col in required_columns):
        st.error(f"Missing required columns: {set(required_columns) - set(df.columns)}")
    else:
        df['Orders'] = pd.to_numeric(df['Orders'], errors='coerce')
        invalid_orders_df = df[df['Orders'].isna()]
        if not invalid_orders_df.empty:
            st.warning("⚠️ Some 'Orders' values are missing or non-numeric. Highlighted below:")
            st.dataframe(invalid_orders_df)
            df = df.dropna(subset=['Orders'])

        # Clustering
        def form_clusters(df, min_orders):
            coords = df[['Latitude', 'Longitude']].values
            clusters = []
            used = set()

            for idx, row in df.iterrows():
                if idx in used:
                    continue
                center = (row['Latitude'], row['Longitude'])
                cluster = [idx]
                total_orders = row['Orders']
                
                for jdx, row2 in df.iterrows():
                    if jdx != idx and jdx not in used:
                        dist = haversine(center, (row2['Latitude'], row2['Longitude']))
                        if dist <= 1.5:
                            cluster.append(jdx)
                            total_orders += row2['Orders']

                for i in cluster:
                    used.add(i)
                clusters.append((cluster, total_orders))

            green, blue = [], []
            for i, (cluster, total_orders) in enumerate(clusters):
                cluster_type = "Green" if total_orders >= min_orders else "Blue"
                for idx in cluster:
                    df.at[idx, 'Cluster ID'] = i
                    df.at[idx, 'Cluster Type'] = cluster_type

            return df

        df = form_clusters(df.copy(), min_orders_threshold)

        # Routing function
        def compute_distance_matrix(locations):
            dist_matrix = np.zeros((len(locations), len(locations)))
            for i, origin in enumerate(locations):
                for j, dest in enumerate(locations):
                    if i != j:
                        dist_matrix[i][j] = haversine(origin, dest)
            return dist_matrix

        def optimize_route(locations):
            tsp_size = len(locations)
            if tsp_size < 2:
                return list(range(tsp_size)), 0

            distance_matrix = compute_distance_matrix(locations)
            manager = pywrapcp.RoutingIndexManager(tsp_size, 1, 0)
            routing = pywrapcp.RoutingModel(manager)

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
                return list(range(tsp_size)), 0

            index = routing.Start(0)
            route = []
            route_distance = 0
            while not routing.IsEnd(index):
                route.append(manager.IndexToNode(index))
                prev_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(prev_index, index, 0)

            return route, route_distance / 1000

        summary = []
        for cluster_id in df['Cluster ID'].unique():
            cluster_df = df[df['Cluster ID'] == cluster_id]
            locations = cluster_df[['Latitude', 'Longitude']].values.tolist()
            route, distance_km = optimize_route(locations)
            total_orders = cluster_df['Orders'].sum()
            vehicle_cost = DEFAULT_VEHICLE_COST
            cpo = vehicle_cost / total_orders if total_orders else 0
            df.loc[cluster_df.index, 'Route Order'] = route
            df.loc[cluster_df.index, 'Distance KM'] = distance_km
            df.loc[cluster_df.index, 'CPO'] = cpo

            summary.append({
                'Cluster ID': cluster_id,
                'Type': cluster_df['Cluster Type'].iloc[0],
                'Orders': total_orders,
                'Distance (KM)': round(distance_km, 2),
                'CPO': round(cpo, 2)
            })

        st.subheader("Cluster-wise Summary")
        st.dataframe(pd.DataFrame(summary))

        # Map
        st.subheader("Cluster Map")
        color_map = {'Green': [0, 255, 0], 'Blue': [0, 0, 255]}
        df['color'] = df['Cluster Type'].apply(lambda x: color_map[x])

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=df,
            get_position='[Longitude, Latitude]',
            get_radius=100,
            get_fill_color='color',
            pickable=True,
        )

        tooltip = {
            "html": "<b>Society:</b> {Society Name} <br/> <b>Cluster:</b> {Cluster ID} ({Cluster Type}) <br/> <b>Orders:</b> {Orders} <br/> <b>CPO:</b> ₹{CPO:.2f}",
            "style": {"backgroundColor": "steelblue", "color": "white"}
        }

        st.pydeck_chart(pdk.Deck(
            initial_view_state=pdk.ViewState(
                latitude=df['Latitude'].mean(),
                longitude=df['Longitude'].mean(),
                zoom=11,
            ),
            layers=[layer],
            tooltip=tooltip
        ))

        st.download_button("Download Output CSV", data=df.to_csv(index=False), file_name="clustered_output.csv")

        st.markdown("---")
        st.markdown("### Sample Template")
        st.dataframe(pd.DataFrame(columns=required_columns))
