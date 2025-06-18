import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from haversine import haversine, Unit
from io import BytesIO

st.set_page_config(layout="wide")

# --- UTILITY & ALGORITHM FUNCTIONS ---

def validate_columns(df):
    """Validate if the dataframe contains all required columns."""
    required_cols = {'Society ID', 'Society Name', 'Latitude', 'Longitude', 'Orders', 'Hub ID'}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        return f"File is missing required columns: {', '.join(missing_cols)}"
    # Data type validation
    for col in ['Latitude', 'Longitude', 'Orders', 'Hub ID']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    if df.isnull().values.any():
        return "File contains non-numeric or empty values in required numeric columns. Please check."
    return None

def get_delivery_sequence(points, depot_coord):
    """
    Calculates the delivery sequence and total distance for a cluster using Nearest Neighbor heuristic.
    The route starts at the depot, visits all points, and returns to the depot.
    
    Args:
        points (list of dicts): Each dict must have 'Society ID', 'Latitude', 'Longitude'.
        depot_coord (tuple): (latitude, longitude) of the depot.
    
    Returns:
        tuple: (list of society IDs in order, total distance in km)
    """
    if not points:
        return [], 0.0

    point_coords = {p['Society ID']: (p['Latitude'], p['Longitude']) for p in points}
    
    current_coord = depot_coord
    unvisited_ids = set(point_coords.keys())
    path = []
    total_distance = 0.0

    while unvisited_ids:
        nearest_id = min(unvisited_ids, key=lambda pid: haversine(current_coord, point_coords[pid]))
        
        distance_to_nearest = haversine(current_coord, point_coords[nearest_id])
        total_distance += distance_to_nearest
        
        current_coord = point_coords[nearest_id]
        path.append(nearest_id)
        unvisited_ids.remove(nearest_id)
    
    # Add distance from the last point back to the depot
    total_distance += haversine(current_coord, depot_coord)
        
    return path, round(total_distance, 2)


@st.cache_data
def run_clustering(df, depot_lat, depot_lon, costs):
    """
    A more robust, rule-based greedy clustering algorithm.
    """
    clusters = []
    cluster_id_counter = 1
    unclustered_societies = []
    
    # Create a copy to avoid modifying the original cached DataFrame
    remaining_df = df.copy()
    depot_coord = (depot_lat, depot_lon)

    # --- MAIN CLUSTERS (>= 180 orders) ---
    # Prioritize single large societies that qualify on their own
    main_candidates = remaining_df[remaining_df['Orders'] >= 180].copy()
    for _, society in main_candidates.iterrows():
        society_dict = [society.to_dict()]
        path, distance = get_delivery_sequence(society_dict, depot_coord)
        clusters.append({
            'Cluster ID': f"Main-{cluster_id_counter}",
            'Type': 'Main', 'Societies': society_dict, 'Orders': society['Orders'], 
            'Distance': distance, 'Path': path, 'Cost': costs['main']
        })
        cluster_id_counter += 1
    remaining_df = remaining_df.drop(main_candidates.index)

    # --- CLUSTER BUILDING LOOP for MINI and MICRO ---
    # Group by Hub ID, as clusters cannot span hubs
    for hub_id, hub_group in remaining_df.groupby('Hub ID'):
        
        # Sort by distance from depot to start with closer societies
        hub_group['dist_from_depot'] = hub_group.apply(
            lambda row: haversine(depot_coord, (row['Latitude'], row['Longitude'])), axis=1)
        
        societies_in_hub = hub_group.sort_values('dist_from_depot').to_dict('records')
        
        while societies_in_hub:
            seed = societies_in_hub.pop(0)
            current_cluster = [seed]
            current_orders = seed['Orders']
            
            # Find nearby societies to potentially add
            potential_additions = []
            for other in societies_in_hub:
                # Proximity check: within 2km of the seed society
                if haversine((seed['Latitude'], seed['Longitude']), (other['Latitude'], other['Longitude'])) <= 2.0:
                    potential_additions.append(other)
            
            # --- Attempt to form a MINI Cluster (121-179 orders) ---
            # Greedily add closest points until order limit is reached
            temp_cluster_mini = list(current_cluster)
            temp_orders_mini = current_orders
            for addition in sorted(potential_additions, key=lambda p: haversine((seed['Latitude'], seed['Longitude']), (p['Latitude'], p['Longitude']))):
                if temp_orders_mini + addition['Orders'] <= 179:
                    temp_cluster_mini.append(addition)
                    temp_orders_mini += addition['Orders']
            
            # Check if a valid Mini cluster was formed
            if 121 <= temp_orders_mini <= 179:
                path, distance = get_delivery_sequence(temp_cluster_mini, depot_coord)
                # Mini cluster distance constraint: Total route <= 15km
                if distance <= 15.0:
                    clusters.append({
                        'Cluster ID': f"Mini-{cluster_id_counter}",
                        'Type': 'Mini', 'Societies': temp_cluster_mini, 'Orders': temp_orders_mini, 
                        'Distance': distance, 'Path': path, 'Cost': costs['mini']
                    })
                    cluster_id_counter += 1
                    # Remove used societies from the pool
                    used_ids = {s['Society ID'] for s in temp_cluster_mini}
                    societies_in_hub = [s for s in societies_in_hub if s['Society ID'] not in used_ids]
                    continue # Move to next seed

            # --- If not a Mini, attempt to form a MICRO Cluster (<= 120 orders) ---
            temp_cluster_micro = list(current_cluster)
            temp_orders_micro = current_orders
            for addition in sorted(potential_additions, key=lambda p: haversine((seed['Latitude'], seed['Longitude']), (p['Latitude'], p['Longitude']))):
                if temp_orders_micro + addition['Orders'] <= 120:
                    temp_cluster_micro.append(addition)
                    temp_orders_micro += addition['Orders']

            path, distance = get_delivery_sequence(temp_cluster_micro, depot_coord)
            # Micro cluster distance constraint: Total route <= 10km
            if distance <= 10.0:
                clusters.append({
                    'Cluster ID': f"Micro-{cluster_id_counter}",
                    'Type': 'Micro', 'Societies': temp_cluster_micro, 'Orders': temp_orders_micro,
                    'Distance': distance, 'Path': path, 'Cost': costs['micro']
                })
                cluster_id_counter += 1
                used_ids = {s['Society ID'] for s in temp_cluster_micro}
                societies_in_hub = [s for s in societies_in_hub if s['Society ID'] not in used_ids]
            else:
                # If seed couldn't form a valid cluster, it becomes unclustered for now
                unclustered_societies.append(seed)

    # Process remaining unclustered societies
    for society in unclustered_societies:
        path, distance = get_delivery_sequence([society], depot_coord)
        clusters.append({
            'Cluster ID': f"Unclustered-{society['Society ID']}",
            'Type': 'Unclustered', 'Societies': [society], 'Orders': society['Orders'], 
            'Distance': distance, 'Path': path, 'Cost': 0
        })

    return clusters

def create_summary_df(clusters):
    summary_rows = []
    for c in clusters:
        total_orders = c['Orders']
        cpo = (c['Cost'] / total_orders) if total_orders > 0 else 0
        summary_rows.append({
            'Cluster ID': c['Cluster ID'],
            'Cluster Type': c['Type'],
            'No. of Societies': len(c['Societies']),
            'Total Orders': total_orders,
            'Total Distance (km)': c['Distance'],
            'CPO (₹)': round(cpo, 2),
            'Delivery Sequence (by ID)': ' -> '.join(map(str, c['Path'])),
        })
    return pd.DataFrame(summary_rows)

def create_unified_map(clusters, depot_coord):
    m = folium.Map(location=depot_coord, zoom_start=12, tiles="CartoDB positron")
    folium.Marker(depot_coord, popup="Depot", icon=folium.Icon(color='black', icon='industry', prefix='fa')).add_to(m)
    
    colors = {'Main': 'blue', 'Mini': 'green', 'Micro': 'purple', 'Unclustered': 'red'}
    
    for c in clusters:
        color = colors.get(c['Type'], 'gray')
        
        # Create a feature group for the cluster
        fg = folium.FeatureGroup(name=f"{c['Cluster ID']} ({c['Type']})")

        # Get coordinates for the delivery path
        path_coords = [depot_coord]
        seq_map = {s['Society ID']: (s['Latitude'], s['Longitude']) for s in c['Societies']}
        for society_id in c['Path']:
            if society_id in seq_map:
                path_coords.append(seq_map[society_id])
        path_coords.append(depot_coord) # Return to depot
        
        # Draw lines and markers
        if len(path_coords) > 2:
            folium.PolyLine(path_coords, color=color, weight=2.5, opacity=0.8).add_to(fg)
        for society in c['Societies']:
            folium.Marker(
                location=[society['Latitude'], society['Longitude']],
                popup=f"<b>{society['Society Name']}</b><br>Orders: {society['Orders']}<br>Cluster: {c['Cluster ID']}",
                icon=folium.Icon(color=color, icon='info-sign')
            ).add_to(fg)
        
        fg.add_to(m)
        
    folium.LayerControl().add_to(m)
    return m


# --- STREAMLIT UI ---
st.title("🚚 Logistics Cluster Optimizer")

# --- SIDEBAR ---
with st.sidebar:
    st.header("1. Depot Settings")
    depot_lat = st.number_input("Depot Latitude", value=12.9716, format="%.6f")
    depot_long = st.number_input("Depot Longitude", value=77.5946, format="%.6f")

    st.header("2. Cluster Costs (Van + CEE)")
    main_cost = st.number_input("Main Cluster Cost (₹)", value=834 + 333)
    mini_cost = st.number_input("Mini Cluster Cost (₹)", value=700 + 200)
    micro_cost = st.number_input("Micro Cluster Cost (₹)", value=500 + 167)
    costs = {'main': main_cost, 'mini': mini_cost, 'micro': micro_cost}

    st.header("3. Upload Data")
    template = pd.DataFrame({'Society ID': [], 'Society Name': [], 'Latitude': [], 'Longitude': [], 'Orders': [], 'Hub ID': []})
    st.download_button("Download Template", template.to_csv(index=False), "input_template.csv")
    file = st.file_uploader("Upload Society Data CSV", type=["csv"])

# --- MAIN PAGE ---
if file is None:
    st.info("Please upload a society data file to begin analysis.")
    st.stop()

try:
    df_raw = pd.read_csv(file)
    validation_error = validate_columns(df_raw)
    if validation_error:
        st.error(validation_error)
        st.stop()
except Exception as e:
    st.error(f"Error reading or parsing file: {e}")
    st.stop()

# Use session state to store results
if 'clusters' not in st.session_state:
    st.session_state.clusters = None

if st.button("🚀 Generate Clusters", type="primary"):
    with st.spinner("Analyzing data and forming clusters..."):
        st.session_state.clusters = run_clustering(df_raw, depot_lat, depot_long, costs)

if st.session_state.clusters is not None:
    clusters = st.session_state.clusters
    summary_df = create_summary_df(clusters)
    
    # --- DISPLAY RESULTS ---
    st.header("📊 Cluster Summary")
    st.dataframe(summary_df)
    
    csv_buffer = BytesIO()
    summary_df.to_csv(csv_buffer, index=False)
    st.download_button(
        "Download Full Summary (CSV)",
        data=csv_buffer.getvalue(),
        file_name="cluster_summary.csv",
        mime="text/csv"
    )

    st.header("🗺️ Unified Map View")
    st.info("You can toggle clusters on/off using the layer control icon in the top-right of the map.")
    unified_map = create_unified_map(clusters, (depot_lat, depot_long))
    st_folium(unified_map, width=1200, height=600, returned_objects=[])

    st.header("🔍 Individual Cluster Details")
    cluster_id_to_show = st.selectbox(
        "Select a Cluster to Inspect",
        options=summary_df['Cluster ID']
    )
    
    selected_cluster = next((c for c in clusters if c['Cluster ID'] == cluster_id_to_show), None)

    if selected_cluster:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"Details for {selected_cluster['Cluster ID']}")
            cluster_details_df = pd.DataFrame(selected_cluster['Societies'])
            st.dataframe(cluster_details_df[['Society ID', 'Society Name', 'Orders']])
            
            detail_csv_buffer = BytesIO()
            cluster_details_df.to_csv(detail_csv_buffer, index=False)
            st.download_button(
                f"Download Details for {selected_cluster['Cluster ID']}",
                data=detail_csv_buffer.getvalue(),
                file_name=f"cluster_{selected_cluster['Cluster ID']}_details.csv",
                mime="text/csv"
            )
        with col2:
            st.subheader("Route Map")
            cluster_map = create_unified_map([selected_cluster], (depot_lat, depot_long))
            st_folium(cluster_map, width=600, height=400, returned_objects=[])
