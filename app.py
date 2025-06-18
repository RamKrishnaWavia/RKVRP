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
    A robust, multi-pass, prioritized clustering algorithm.
    """
    all_clusters = []
    cluster_id_counter = 1
    
    societies_map = {s['Society ID']: s for s in df.to_dict('records')}
    unprocessed_ids = set(societies_map.keys())
    depot_coord = (depot_lat, depot_lon)

    for hub_id in df['Hub ID'].unique():
        hub_society_ids = {sid for sid in unprocessed_ids if societies_map[sid]['Hub ID'] == hub_id}
        
        # PASS 1: Form MAIN Clusters
        sorted_seeds = sorted(list(hub_society_ids), key=lambda sid: societies_map[sid]['Orders'], reverse=True)
        for seed_id in sorted_seeds:
            if seed_id not in hub_society_ids: continue
            seed = societies_map[seed_id]
            potential_cluster = [seed]
            potential_orders = seed['Orders']
            neighbors = [societies_map[nid] for nid in hub_society_ids if nid != seed_id and haversine((seed['Latitude'], seed['Longitude']), (societies_map[nid]['Latitude'], societies_map[nid]['Longitude'])) < 2.0]
            for neighbor in sorted(neighbors, key=lambda s: s['Orders'], reverse=True):
                if potential_orders + neighbor['Orders'] <= 220:
                    potential_cluster.append(neighbor)
                    potential_orders += neighbor['Orders']
            if 180 <= potential_orders <= 220:
                path, distance = get_delivery_sequence(potential_cluster, depot_coord)
                all_clusters.append({'Cluster ID': f"Main-{cluster_id_counter}", 'Type': 'Main', 'Societies': potential_cluster, 'Orders': potential_orders, 'Distance': distance, 'Path': path, 'Cost': costs['main']})
                cluster_id_counter += 1
                used_ids = {s['Society ID'] for s in potential_cluster}
                hub_society_ids -= used_ids

        # PASS 2: Form MINI Clusters
        sorted_seeds = sorted(list(hub_society_ids), key=lambda sid: societies_map[sid]['Orders'], reverse=True)
        for seed_id in sorted_seeds:
            if seed_id not in hub_society_ids: continue
            seed = societies_map[seed_id]
            potential_cluster = [seed]
            potential_orders = seed['Orders']
            neighbors = [societies_map[nid] for nid in hub_society_ids if nid != seed_id and haversine((seed['Latitude'], seed['Longitude']), (societies_map[nid]['Latitude'], societies_map[nid]['Longitude'])) < 2.0]
            for neighbor in sorted(neighbors, key=lambda s: s['Orders'], reverse=True):
                if potential_orders + neighbor['Orders'] <= 179:
                    potential_cluster.append(neighbor)
                    potential_orders += neighbor['Orders']
            if 121 <= potential_orders <= 179:
                path, distance = get_delivery_sequence(potential_cluster, depot_coord)
                all_clusters.append({'Cluster ID': f"Mini-{cluster_id_counter}", 'Type': 'Mini', 'Societies': potential_cluster, 'Orders': potential_orders, 'Distance': distance, 'Path': path, 'Cost': costs['mini']})
                cluster_id_counter += 1
                used_ids = {s['Society ID'] for s in potential_cluster}
                hub_society_ids -= used_ids
                
        # PASS 3: Form MICRO Clusters
        sorted_seeds = sorted(list(hub_society_ids), key=lambda sid: societies_map[sid]['Orders'], reverse=True)
        for seed_id in sorted_seeds:
            if seed_id not in hub_society_ids: continue
            seed = societies_map[seed_id]
            potential_cluster = [seed]
            potential_orders = seed['Orders']
            neighbors = [societies_map[nid] for nid in hub_society_ids if nid != seed_id]
            for neighbor in sorted(neighbors, key=lambda s: s['Orders'], reverse=True):
                if potential_orders + neighbor['Orders'] <= 120:
                    potential_cluster.append(neighbor)
                    potential_orders += neighbor['Orders']
            if 1 <= potential_orders <= 120:
                dist_to_last = get_distance_to_last_society(potential_cluster, depot_coord)
                if dist_to_last < 15.0:
                    path, distance = get_delivery_sequence(potential_cluster, depot_coord)
                    all_clusters.append({'Cluster ID': f"Micro-{cluster_id_counter}", 'Type': 'Micro', 'Societies': potential_cluster, 'Orders': potential_orders, 'Distance': distance, 'Path': path, 'Cost': costs['micro']})
                    cluster_id_counter += 1
                    used_ids = {s['Society ID'] for s in potential_cluster}
                    hub_society_ids -= used_ids

        # PASS 4: Handle Unclustered
        for sid in hub_society_ids:
            society = societies_map[sid]
            path, distance = get_delivery_sequence([society], depot_coord)
            all_clusters.append({'Cluster ID': f"Unclustered-{sid}", 'Type': 'Unclustered', 'Societies': [society], 'Orders': society['Orders'], 'Distance': distance, 'Path': path, 'Cost': 0})
        
        unprocessed_ids -= {s['Society ID'] for s in df[df['Hub ID'] == hub_id].to_dict('records')}

    return all_clusters

def create_summary_df(clusters):
    """
    Creates the summary DataFrame with society names in the delivery sequence.
    """
    summary_rows = []
    for c in clusters:
        total_orders = c['Orders']
        cpo = (c['Cost'] / total_orders) if total_orders > 0 else 0
        
        # Create a mapping from Society ID to Society Name for this cluster
        id_to_name = {s['Society ID']: s['Society Name'] for s in c['Societies']}
        
        # Build the delivery sequence string using names
        delivery_sequence_str = ' -> '.join([id_to_name[sid] for sid in c['Path']])
        
        summary_rows.append({
            'Cluster ID': c['Cluster ID'],
            'Cluster Type': c['Type'],
            'No. of Societies': len(c['Societies']),
            'Total Orders': total_orders,
            'Total Distance (km)': c['Distance'],
            'CPO (₹)': round(cpo, 2),
            'Delivery Sequence': delivery_sequence_str, # CHANGED
        })
    return pd.DataFrame(summary_rows)

def create_unified_map(clusters, depot_coord):
    """
    Creates the map with individual line segments showing distance for each leg of the journey.
    """
    m = folium.Map(location=depot_coord, zoom_start=12, tiles="CartoDB positron")
    folium.Marker(depot_coord, popup="Depot", icon=folium.Icon(color='black', icon='industry', prefix='fa')).add_to(m)
    
    colors = {'Main': 'blue', 'Mini': 'green', 'Micro': 'purple', 'Unclustered': 'red'}
    
    for c in clusters:
        color = colors.get(c['Type'], 'gray')
        fg = folium.FeatureGroup(name=f"{c['Cluster ID']} ({c['Type']})")
        
        # Create helper maps for this cluster
        id_to_name = {s['Society ID']: s['Society Name'] for s in c['Societies']}
        id_to_coord = {s['Society ID']: (s['Latitude'], s['Longitude']) for s in c['Societies']}

        # Build a full path list with names and coordinates for descriptive tooltips
        full_path_info = [("Depot", depot_coord)]
        for sid in c['Path']:
            full_path_info.append((id_to_name[sid], id_to_coord[sid]))
        full_path_info.append(("Depot", depot_coord)) # Add return to depot

        # --- NEW: Draw individual line segments with distance tooltips ---
        for i in range(len(full_path_info) - 1):
            start_name, start_coord = full_path_info[i]
            end_name, end_coord = full_path_info[i+1]
            
            dist = haversine(start_coord, end_coord)
            tooltip_text = f"{start_name} to {end_name}: {dist:.2f} km"
            
            folium.PolyLine(
                locations=[start_coord, end_coord],
                color=color,
                weight=2.5,
                opacity=0.8,
                tooltip=tooltip_text
            ).add_to(fg)
        
        # Add markers for each society (unchanged)
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
    clusters = st.session_state.clusters
    summary_df = create_summary_df(clusters)
    
    st.header("📊 Cluster Summary")
    st.dataframe(summary_df.sort_values(by=['Cluster Type', 'Cluster ID']))
    
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
        options=sorted(summary_df['Cluster ID'].tolist())
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
