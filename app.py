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
    """Validate if the dataframe contains all required columns and correct data types."""
    required_cols = {'Society ID', 'Society Name', 'Latitude', 'Longitude', 'Orders', 'Hub ID'}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        return f"File is missing required columns: {', '.join(missing_cols)}"
    
    for col in ['Latitude', 'Longitude', 'Orders', 'Hub ID']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    if df.isnull().values.any():
        return "File contains non-numeric or empty values in required numeric columns (Lat, Lon, Orders, Hub ID). Please check."
    return None

def get_delivery_sequence(points, depot_coord):
    """
    Calculates the delivery sequence (path) and total round-trip distance for a cluster.
    Route: Depot -> Society 1 -> ... -> Society N -> Depot.
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
    
    total_distance += haversine(current_coord, depot_coord) # Return to depot
        
    return path, round(total_distance, 2)

def get_distance_to_last_society(points, depot_coord):
    """
    Calculates the TSP path and then finds the straight-line distance from the depot to the last society in that path.
    """
    if not points:
        return 0.0
    
    # We only need the path, not the full round-trip distance for this specific check
    path, _ = get_delivery_sequence(points, depot_coord)
    if not path:
        return 0.0
        
    last_society_id = path[-1]
    point_coords = {p['Society ID']: (p['Latitude'], p['Longitude']) for p in points}
    last_society_coord = point_coords[last_society_id]
    
    return round(haversine(depot_coord, last_society_coord), 2)


@st.cache_data
def run_clustering(df, depot_lat, depot_lon, costs):
    """
    Corrected clustering algorithm with prioritized rule checking.
    """
    clusters = []
    cluster_id_counter = 1
    
    unprocessed_societies = {s['Society ID']: s for s in df.to_dict('records')}
    depot_coord = (depot_lat, depot_lon)
    
    # Process hub by hub
    for hub_id in df['Hub ID'].unique():
        hub_society_ids = {sid for sid, s in unprocessed_societies.items() if s['Hub ID'] == hub_id}
        
        # Keep finding seeds until all societies in the hub are processed
        while hub_society_ids:
            # Pick a seed and remove it from this hub's pool to avoid re-processing
            seed_id = hub_society_ids.pop()
            seed = unprocessed_societies[seed_id]
            
            cluster_formed = False
            
            # --- ATTEMPT 1: FORM MAIN CLUSTER (180-220 orders, <2km proximity) ---
            if 180 <= seed['Orders'] <= 220: # Optimization: if seed alone fits, form it
                potential_cluster = [seed]
                potential_orders = seed['Orders']
            elif seed['Orders'] < 180: # Try to add more
                potential_cluster = [seed]
                potential_orders = seed['Orders']
                neighbors = sorted(
                    [unprocessed_societies[sid] for sid in hub_society_ids if haversine((seed['Latitude'], seed['Longitude']), (unprocessed_societies[sid]['Latitude'], unprocessed_societies[sid]['Longitude'])) < 2.0],
                    key=lambda s: s['Orders'], reverse=True
                )
                for neighbor in neighbors:
                    if potential_orders + neighbor['Orders'] <= 220:
                        potential_cluster.append(neighbor)
                        potential_orders += neighbor['Orders']
            else: # Seed is > 220, cannot start a Main cluster
                potential_orders = seed['Orders']

            if 180 <= potential_orders <= 220:
                path, distance = get_delivery_sequence(potential_cluster, depot_coord)
                clusters.append({
                    'Cluster ID': f"Main-{cluster_id_counter}", 'Type': 'Main', 'Societies': potential_cluster, 
                    'Orders': potential_orders, 'Distance': distance, 'Path': path, 'Cost': costs['main']
                })
                cluster_id_counter += 1
                cluster_formed = True


            # --- ATTEMPT 2: FORM MINI CLUSTER (121-179 orders, <2km proximity) ---
            if not cluster_formed:
                if 121 <= seed['Orders'] <= 179:
                    potential_cluster = [seed]
                    potential_orders = seed['Orders']
                elif seed['Orders'] < 121:
                    potential_cluster = [seed]
                    potential_orders = seed['Orders']
                    neighbors = sorted(
                        [unprocessed_societies[sid] for sid in hub_society_ids if haversine((seed['Latitude'], seed['Longitude']), (unprocessed_societies[sid]['Latitude'], unprocessed_societies[sid]['Longitude'])) < 2.0],
                        key=lambda s: s['Orders'], reverse=True
                    )
                    for neighbor in neighbors:
                        if potential_orders + neighbor['Orders'] <= 179:
                            potential_cluster.append(neighbor)
                            potential_orders += neighbor['Orders']
                else:
                    potential_orders = seed['Orders']

                if 121 <= potential_orders <= 179:
                    path, distance = get_delivery_sequence(potential_cluster, depot_coord)
                    clusters.append({
                        'Cluster ID': f"Mini-{cluster_id_counter}", 'Type': 'Mini', 'Societies': potential_cluster,
                        'Orders': potential_orders, 'Distance': distance, 'Path': path, 'Cost': costs['mini']
                    })
                    cluster_id_counter += 1
                    cluster_formed = True

            # --- ATTEMPT 3: FORM MICRO CLUSTER (1-120 orders, <15km depot-to-last) ---
            if not cluster_formed:
                if seed['Orders'] <= 120:
                    potential_cluster = [seed]
                    potential_orders = seed['Orders']
                    # Micro cluster has no 2km proximity rule, can take any from hub
                    neighbors = sorted(
                        [unprocessed_societies[sid] for sid in hub_society_ids],
                        key=lambda s: haversine(depot_coord, (s['Latitude'], s['Longitude'])) # Add closest from depot
                    )
                    for neighbor in neighbors:
                        if potential_orders + neighbor['Orders'] <= 120:
                            potential_cluster.append(neighbor)
                            potential_orders += neighbor['Orders']

                    # Check Micro's unique distance constraint
                    dist_to_last = get_distance_to_last_society(potential_cluster, depot_coord)
                    if dist_to_last < 15.0:
                        path, distance = get_delivery_sequence(potential_cluster, depot_coord)
                        clusters.append({
                            'Cluster ID': f"Micro-{cluster_id_counter}", 'Type': 'Micro', 'Societies': potential_cluster,
                            'Orders': potential_orders, 'Distance': distance, 'Path': path, 'Cost': costs['micro']
                        })
                        cluster_id_counter += 1
                        cluster_formed = True

            # --- FINALIZE AND CLEANUP ---
            if cluster_formed:
                # Remove all used societies from the main pool and the hub's pool
                used_ids = {s['Society ID'] for s in potential_cluster}
                for used_id in used_ids:
                    unprocessed_societies.pop(used_id, None)
                    hub_society_ids.discard(used_id)
            else:
                # If seed couldn't form ANY cluster, it becomes Unclustered
                path, distance = get_delivery_sequence([seed], depot_coord)
                clusters.append({
                    'Cluster ID': f"Unclustered-{seed_id}", 'Type': 'Unclustered',
                    'Societies': [seed], 'Orders': seed['Orders'],
                    'Distance': distance, 'Path': path, 'Cost': 0
                })
                unprocessed_societies.pop(seed_id) # Remove from main pool

    return clusters

def create_summary_df(clusters):
    # (This function remains the same)
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
    # (This function remains the same)
    m = folium.Map(location=depot_coord, zoom_start=12, tiles="CartoDB positron")
    folium.Marker(depot_coord, popup="Depot", icon=folium.Icon(color='black', icon='industry', prefix='fa')).add_to(m)
    
    colors = {'Main': 'blue', 'Mini': 'green', 'Micro': 'purple', 'Unclustered': 'red'}
    
    for c in clusters:
        color = colors.get(c['Type'], 'gray')
        fg = folium.FeatureGroup(name=f"{c['Cluster ID']} ({c['Type']})")
        path_coords = [depot_coord]
        seq_map = {s['Society ID']: (s['Latitude'], s['Longitude']) for s in c['Societies']}
        for society_id in c['Path']:
            if society_id in seq_map:
                path_coords.append(seq_map[society_id])
        path_coords.append(depot_coord)
        
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

with st.sidebar:
    st.header("1. Depot Settings")
    depot_lat = st.number_input("Depot Latitude", value=12.9716, format="%.6f")
    depot_long = st.number_input("Depot Longitude", value=77.5946, format="%.6f")

    st.header("2. Cluster Costs")
    main_van_cost = st.number_input("Main Cluster Van Cost (₹)", value=833)
    main_cee_cost = st.number_input("Main Cluster CEE Cost (₹)", value=333)
    micro_van_cost = st.number_input("Micro Cluster Van Cost (₹)", value=500)
    micro_cee_cost = st.number_input("Micro Cluster CEE Cost (₹)", value=200)
    mini_van_cost = st.number_input("Mini Cluster Van Cost (₹)", value=1000)
    mini_cee_cost = st.number_input("Mini Cluster CEE Cost (₹)", value=200)
    
    costs = {
        'main': main_van_cost + main_cee_cost,
        'mini': mini_van_cost + mini_cee_cost,
        'micro': micro_van_cost + micro_cee_cost
    }

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

if 'clusters' not in st.session_state:
    st.session_state.clusters = None

if st.button("🚀 Generate Clusters", type="primary"):
    with st.spinner("Analyzing data and forming clusters..."):
        st.session_state.clusters = run_clustering(df_raw, depot_lat, depot_long, costs)

if st.session_state.clusters is not None:
    # (The rest of the UI display code remains the same)
    clusters = st.session_state.clusters
    summary_df = create_summary_df(clusters)
    
    st.header("📊 Cluster Summary")
    st.dataframe(summary_df)
    
    csv_buffer = BytesIO()
    summary_df.to_csv(csv_buffer, index=False, encoding='utf-8')
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
            cluster_details_df.to_csv(detail_csv_buffer, index=False, encoding='utf-8')
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
