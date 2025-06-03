import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from haversine import haversine, Unit
import pydeck as pdk

st.set_page_config(layout="wide")
st.title("🗺️ Optimized Milk Delivery Route Planner")

# Vehicle cost input
vehicle_cost = st.sidebar.number_input("Vehicle Monthly Cost (₹)", min_value=1000, value=35000, step=1000)

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

required_columns = ['Society ID', 'Society Name', 'City', 'Drop Point', 'Latitude', 'Longitude', 'Orders']

def optimize_route(locations):
    tsp_size = len(locations)
    if tsp_size <= 1:
        return [], 0.0

    dist_matrix = [[0]*tsp_size for _ in range(tsp_size)]
    for i in range(tsp_size):
        for j in range(tsp_size):
            if i != j:
                dist_matrix[i][j] = haversine(locations[i], locations[j], unit=Unit.KILOMETERS)

    num_routes = 1
    depot = 0

    routing = pywrapcp.RoutingModel(tsp_size, num_routes, depot)
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    def distance_callback(from_index, to_index):
        return int(dist_matrix[from_index][to_index] * 1000)  # in meters

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    solution = routing.SolveWithParameters(search_parameters)
    if not solution:
        return [], 0.0

    route = []
    index = routing.Start(0)
    route_distance = 0
    while not routing.IsEnd(index):
        route.append(index)
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance += dist_matrix[previous_index][index]
    route.append(index)

    return route, route_distance

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
    else:
        # Check for invalid order values
        invalid_orders = df[~df['Orders'].apply(lambda x: str(x).isdigit())]
        if not invalid_orders.empty:
            st.warning("⚠️ Some 'Orders' values are missing or non-numeric. Please check your CSV.")
            st.dataframe(invalid_orders)
        else:
            df['Orders'] = df['Orders'].astype(int)
            df = df.dropna(subset=['Latitude', 'Longitude'])

            # Determine number of clusters dynamically
            orders_per_vehicle = 200
            total_orders = df['Orders'].sum()
            num_clusters = max(1, int(np.ceil(total_orders / orders_per_vehicle)))

            kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(df[['Latitude', 'Longitude']])
            df['Cluster ID'] = kmeans.labels_

            # Sidebar filters
            with st.sidebar:
                selected_city = st.selectbox("Select City", options=["All"] + sorted(df['City'].unique()))
                selected_society = st.selectbox("Select Society", options=["All"] + sorted(df['Society Name'].unique()))
                selected_drop_point = st.selectbox("Select Drop Point", options=["All"] + sorted(df['Drop Point'].unique()))

            # Filter data
            filtered_df = df.copy()
            if selected_city != "All":
                filtered_df = filtered_df[filtered_df['City'] == selected_city]
            if selected_society != "All":
                filtered_df = filtered_df[filtered_df['Society Name'] == selected_society]
            if selected_drop_point != "All":
                filtered_df = filtered_df[filtered_df['Drop Point'] == selected_drop_point]

            st.subheader("📦 Cluster Summary with Routes & CPO")

            for cluster_id in sorted(filtered_df['Cluster ID'].unique()):
                cluster_df = filtered_df[filtered_df['Cluster ID'] == cluster_id]
                if len(cluster_df) <= 1:
                    continue

                st.markdown(f"### 🟢 Cluster {cluster_id}")
                total_orders = cluster_df['Orders'].sum()
                route, distance_km = optimize_route(cluster_df[['Latitude', 'Longitude']].values.tolist())
                cpo = vehicle_cost / total_orders if total_orders else float('inf')

                st.write(f"Total Orders: {total_orders}")
                st.write(f"Cost per Order (CPO): ₹{cpo:.2f}")

                st.pydeck_chart(pdk.Deck(
                    initial_view_state=pdk.ViewState(
                        latitude=cluster_df['Latitude'].mean(),
                        longitude=cluster_df['Longitude'].mean(),
                        zoom=11,
                        pitch=0,
                    ),
                    layers=[
                        pdk.Layer(
                            'ScatterplotLayer',
                            data=cluster_df,
                            get_position='[Longitude, Latitude]',
                            get_color='[200, 30, 0, 160]',
                            get_radius=80,
                        )
                    ],
                    tooltip={"text": "Society: {Society Name}\nOrders: {Orders}\nCPO: ₹" + f"{cpo:.2f}"}
                ))

            st.success("✅ Optimization and Visualization Complete!")

else:
    with open("sample_template.csv", "w") as f:
        f.write("Society ID,Society Name,City,Drop Point,Latitude,Longitude,Orders\n")
        f.write("S001,ABC Residency,Bangalore,Point A,12.9716,77.5946,50\n")
        f.write("S002,XYZ Enclave,Bangalore,Point B,12.9352,77.6142,80\n")
        f.write("S003,Green Ville,Bangalore,Point A,12.9611,77.6387,60\n")

    st.download_button(
        label="📥 Download Sample Template",
        data=open("sample_template.csv").read(),
        file_name="sample_template.csv",
        mime="text/csv"
    )
