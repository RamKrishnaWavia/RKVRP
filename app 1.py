import pandas as pd
from geopy.distance import geodesic
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import numpy as np

# Your depot location
depot = (13.0379, 77.6609)  # Soukya Road Bangalore (replace with exact lat, lon)

# Sample df: Replace this with your actual dataframe with columns:
# 'SocietyID', 'Society Name', 'Latitude', 'Longitude', 'SocietyOrders', 'ClusterID'
df = pd.read_csv('your_societies_with_clusters.csv')

# Vehicle capacity and minimum orders per cluster to assign vehicle
VEHICLE_CAPACITY = 500
MIN_ORDERS = 200

def create_distance_matrix(locations):
    size = len(locations)
    dist_matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if i == j:
                dist_matrix[i][j] = 0
            else:
                dist_matrix[i][j] = geodesic(locations[i], locations[j]).km * 1000
    return dist_matrix.astype(int)

def optimize_route(cluster_df, depot_location, vehicle_capacity=VEHICLE_CAPACITY):
    locations = [depot_location] + list(zip(cluster_df['Latitude'], cluster_df['Longitude']))
    demands = [0] + list(cluster_df['SocietyOrders'])

    dist_matrix = create_distance_matrix(locations)

    manager = pywrapcp.RoutingIndexManager(len(dist_matrix), 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return dist_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return demands[from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index, 0, [vehicle_capacity], True, 'Capacity')

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    solution = routing.SolveWithParameters(search_parameters)
    if not solution:
        return None, None, None

    route = []
    index = routing.Start(0)
    route_distance = 0
    route_load = 0
    while not routing.IsEnd(index):
        node_index = manager.IndexToNode(index)
        route.append(node_index)
        route_load += demands[node_index]
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
    route.append(manager.IndexToNode(index))

    return route, route_distance, route_load

# Summary results
summary = []

cluster_ids = df['ClusterID'].unique()

for cid in sorted(cluster_ids):
    cluster_data = df[df['ClusterID'] == cid].reset_index(drop=True)
    total_orders = cluster_data['SocietyOrders'].sum()

    # Check min orders condition
    if total_orders < MIN_ORDERS:
        vehicles = 0
        route_distance_km = 0
        route_stops = []
    else:
        vehicles = 1
        route, dist_m, load = optimize_route(cluster_data, depot)
        route_distance_km = dist_m / 1000 if dist_m else 0

        # Prepare readable stops
        route_stops = []
        if route:
            for idx in route:
                if idx == 0:
                    route_stops.append('Depot')
                else:
                    soc = cluster_data.loc[idx-1]
                    route_stops.append(f"{soc['Society Name']} (Orders: {soc['SocietyOrders']})")

    summary.append({
        'ClusterID': cid,
        'TotalOrders': total_orders,
        'Vehicles': vehicles,
        'RouteDistance_km': round(route_distance_km, 2),
        'RouteStops': route_stops
    })

# Print summary
for item in summary:
    print(f"\nCluster {item['ClusterID']}:")
    print(f" Total Orders: {item['TotalOrders']}")
    print(f" Vehicles needed: {item['Vehicles']}")
    print(f" Route Distance (km): {item['RouteDistance_km']}")
    if item['Vehicles'] > 0:
        print(" Route Stops:")
        for stop in item['RouteStops']:
            print(f"  - {stop}")
    else:
        print(" No vehicle assigned due to low orders.")

