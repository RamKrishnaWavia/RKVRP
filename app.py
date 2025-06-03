import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

st.set_page_config(page_title="RK - Delivery Route Optimizer", layout="wide")
st.title("RK - Delivery Route Optimizer")
st.markdown("Upload delivery data and optimize routes based on max 200 orders per vehicle and cost constraints.")

# CSV Template Download
if st.button("📅 Download CSV Template"):
    template = pd.DataFrame({
        "City": ["Bangalore", "Bangalore"],
        "Apartment": ["ABC Residency", "Green Heights"],
        "Drop Point": ["Soukya Road", "Soukya Road"],
        "Latitude": [12.935, 12.938],
        "Longitude": [77.614, 77.610],
        "Orders": [50, 120]
    })
    csv = template.to_csv(index=False).encode('utf-8')
    st.download_button("Download Template CSV", data=csv, file_name="milk_delivery_template.csv", mime='text/csv')

# Upload CSV
uploaded_file = st.file_uploader("Upload Delivery Data CSV with City, Apartment, Drop Point, Latitude, Longitude, Orders", type=["csv"])
if not uploaded_file:
    st.info("Please upload a CSV file to proceed.")
    st.stop()

df = pd.read_csv(uploaded_file)

required_cols = ["City", "Apartment", "Drop Point", "Latitude", "Longitude", "Orders"]
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    st.error(f"Missing required columns in upload: {missing_cols}")
    st.stop()

# Sidebar Filters
st.sidebar.header("Filter Delivery Points")

city_options = sorted(df["City"].unique())
selected_cities = st.sidebar.multiselect("Select City(s)", options=city_options, default=city_options)

society_options = sorted(df["Apartment"].unique())
selected_societies = st.sidebar.multiselect("Select Society Name(s)", options=society_options, default=society_options)

drop_point_options = sorted(df["Drop Point"].unique())
selected_drop_points = st.sidebar.multiselect("Select Drop Point(s)", options=drop_point_options, default=drop_point_options)

filtered_df = df[
    (df["City"].isin(selected_cities)) &
    (df["Apartment"].isin(selected_societies)) &
    (df["Drop Point"].isin(selected_drop_points))
].reset_index(drop=True)

if filtered_df.empty:
    st.warning("No data points match the selected filters.")
    st.stop()

st.write(f"Filtered {len(filtered_df)} delivery points.")

# User inputs
vehicle_monthly_cost = st.number_input("Enter vehicle monthly cost (₹)", min_value=1000, value=35000, step=1000)
max_orders_per_vehicle = 200

# Depot location fixed: Soukya Road
# Check if depot row exists in filtered_df Drop Point column
depot_rows = filtered_df[filtered_df["Drop Point"].str.lower() == "souqya road"]
# If depot row missing or typo, let’s get coordinates from input or set fixed coordinates:
depot_lat, depot_lon = None, None
if not depot_rows.empty:
    depot_lat = depot_rows.iloc[0]["Latitude"]
    depot_lon = depot_rows.iloc[0]["Longitude"]
else:
    # Default Soukya Road coords (can be updated)
    depot_lat, depot_lon = 13.0100, 77.6900

# Add depot as first point in data (if not present) to ensure start/end
depot_point = pd.DataFrame({
    "City": ["Depot"],
    "Apartment": ["Depot"],
    "Drop Point": ["Soukya Road"],
    "Latitude": [depot_lat],
    "Longitude": [depot_lon],
    "Orders": [0]
})

if "Soukya Road" not in filtered_df["Drop Point"].values:
    filtered_df = pd.concat([depot_point, filtered_df], ignore_index=True)
else:
    # Move depot to first row if present
    depot_idx = filtered_df[filtered_df["Drop Point"].str.lower() == "souqya road"].index[0]
    filtered_df = pd.concat([filtered_df.loc[[depot_idx]], filtered_df.drop(depot_idx)]).reset_index(drop=True)

# Clustering based on orders to split into routes <= 200 orders
coords = filtered_df[["Latitude", "Longitude"]].values
orders = filtered_df["Orders"].values

# Calculate number of clusters based on total orders and max orders per vehicle
total_orders = orders.sum()
num_clusters = max(1, int(total_orders / max_orders_per_vehicle) + (1 if total_orders % max_orders_per_vehicle else 0))

# KMeans clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
filtered_df['Cluster'] = kmeans.fit_predict(coords)

# For each cluster, check sum orders and split if > 200 (optional: can be improved)

# Function to create distance matrix including depot as start/end
def compute_distance_matrix(locations):
    distances = {}
    for i, from_node in enumerate(locations):
        distances[i] = {}
        for j, to_node in enumerate(locations):
            if i == j:
                distances[i][j] = 0
            else:
                # Euclidean approx distance
                dist = ((from_node[0]-to_node[0])**2 + (from_node[1]-to_node[1])**2)**0.5
                distances[i][j] = dist
    return distances

def optimize_route(cluster_df):
    locations = cluster_df[["Latitude", "Longitude"]].values
    orders_list = cluster_df["Orders"].tolist()

    data = {
        "distance_matrix": compute_distance_matrix(locations),
        "num_vehicles": 1,
        "depot": 0
    }

    manager = pywrapcp.RoutingIndexManager(len(locations), data["num_vehicles"], data["depot"])
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(data["distance_matrix"][from_node][to_node] * 100000)

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add orders capacity constraint
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return orders_list[from_node]
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        [max_orders_per_vehicle],  # vehicle max capacity
        True,  # start cumul to zero
        "Capacity"
    )

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    solution = routing.SolveWithParameters(search_parameters)

    if not solution:
        st.error("No solution found for one of the clusters!")
        return None, None, None

    route = []
    index = routing.Start(0)
    route_distance = 0
    while not routing.IsEnd(index):
        node_index = manager.IndexToNode(index)
        route.append(node_index)
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance += data["distance_matrix"][manager.IndexToNode(previous_index)][manager.IndexToNode(index)]
    route.append(manager.IndexToNode(index))

    return route, route_distance, orders_list

# Run optimization cluster wise
optimized_routes = []
route_id = 1
for c in range(num_clusters):
    cluster_data = filtered_df[filtered_df["Cluster"] == c].reset_index(drop=True)
    if len(cluster_data) < 2:
        # single point cluster, skip or add as is
        optimized_routes.append((route_id, cluster_data, [0, 1], 0))
        route_id += 1
        continue
    route, dist, orders_list = optimize_route(cluster_data)
    if route:
        route_df = cluster_data.iloc[route].copy()
        route_df["Route_Stop"] = range(len(route_df))
        route_df["Route_ID"] = route_id
        optimized_routes.append((route_id, route_df, route, dist))
        route_id += 1

# Combine all routes data
all_routes_df = pd.concat([r[1] for r in optimized_routes], ignore_index=True)

# Cost calculations
# Total km approx = route distance * 111 (approx deg to km)
all_routes_df['Distance_km'] = 0.0

route_distance_map = {}
for r_id, _, _, dist in optimized_routes:
    route_distance_map[r_id] = dist * 111

for idx, row in all_routes_df.iterrows():
    all_routes_df.at[idx, 'Distance_km'] = route_distance_map[row['Route_ID']] / len(all_routes_df[all_routes_df['Route_ID'] == row['Route_ID']])

# Cost per route and cost per order
route_orders_sum = all_routes_df.groupby("Route_ID")["Orders"].sum().to_dict()
route_cost_map = {}
route_cpo_map = {}
for r_id in route_orders_sum:
    cost = vehicle_monthly_cost
    orders_count = route_orders_sum[r_id]
    cpo = cost / orders_count if orders_count else 0
    route_cost_map[r_id] = cost
    route_cpo_map[r_id] = cpo

# Show map with routes and popup info
m = folium.Map(location=[depot_lat, depot_lon], zoom_start=12)

colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred',
          'beige', 'darkblue', 'darkgreen', 'cadetblue', 'pink', 'lightblue', 
          'lightgreen', 'gray', 'black', 'lightgray']

for r_id, route_df, route_list, dist in optimized_routes:
    color = colors[(r_id - 1) % len(colors)]
    points = route_df[["Latitude", "Longitude"]].values.tolist()
    folium.PolyLine(points, color=color, weight=4, opacity=0.7, popup=f"Route {r_id} Distance: {dist*111:.2f} km, CPO: ₹{route_cpo_map[r_id]:.2f}").add_to(m)
    for idx, row in route_df.iterrows():
        popup_text = (f"Route: {r_id}<br>Stop Seq: {row['Route_Stop']}<br>"
                      f"Society: {row['Apartment']}<br>Orders: {row['Orders']}<br>"
                      f"Distance on route (approx): {route_distance_map[r_id]/len(route_df):.2f} km<br>"
                      f"Cost Per Order: ₹{route_cpo_map[r_id]:.2f}")
        folium.Marker(location=[row["Latitude"], row["Longitude"]],
                      popup=popup_text,
                      tooltip=f"Stop {row['Route_Stop']} - {row['Apartment']}",
                      icon=folium.Icon(color=color)).add_to(m)

st.subheader("Optimized Routes Map")
st_data = st_folium(m, width=900, height=600)

# Show summary table
st.subheader("Route Summary")
summary_df = pd.DataFrame({
    "Route ID": list(route_orders_sum.keys()),
    "Total Orders": list(route_orders_sum.values()),
    "Route Distance (km)": [route_distance_map[r] * 111 if r in route_distance_map else 0 for r in route_orders_sum.keys()],
    "Vehicle Cost (₹)": [route_cost_map[r] for r in route_orders_sum.keys()],
    "Cost Per Order (₹)": [route_cpo_map[r] for r in route_orders_sum.keys()]
})
st.dataframe(summary_df)

# Download optimized routes CSV
csv_opt = all_routes_df.to_csv(index=False).encode('utf-8')
st.download_button("Download Optimized Routes CSV", data=csv_opt, file_name="optimized_routes.csv", mime='text/csv')
