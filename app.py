import streamlit as st
import pandas as pd
import folium
from folium.plugins import AntPath
from streamlit_folium import st_folium
from haversine import haversine, Unit
from io import BytesIO

st.set_page_config(layout="wide")

# --- DATA & CONSTANTS ---
PREDEFINED_DEPOTS = {
    "Ahmedabad": {"lat": 22.911, "lon": 72.425},
    "Bangalore Nelamanagala": {"lat": 13.06821829, "lon": 77.44607278},
    "Banagalore Soukya": {"lat": 12.98946539, "lon": 77.78629337},
    "Chennai": {"lat": 13.045, "lon": 80.024},
    "Gurgaon Udyog Vihar": {"lat": 28.4813978, "lon": 77.0522889},
    "Hyderabad Balanagar": {"lat": 17.48467009, "lon": 78.44890182},
    "Kolkata": {"lat": 22.494, "lon": 88.594},
    "Mumbai": {"lat": 19.276, "lon": 73.092},
    "Noida": {"lat": 28.53, "lon": 77.412},
    "Pune Hinjewadi": {"lat": 18.528, "lon": 73.863},
}

# --- UTILITY & ALGORITHM FUNCTIONS ---
# (These functions are correct and remain unchanged)

def validate_columns(df):
    """Validate if the dataframe contains all required columns and correct data types."""
    required_cols = {'Society ID', 'Society Name', 'Latitude', 'Longitude', 'Orders', 'Hub ID', 'Hub Name', 'Number of Blocks'} # Added block column
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        return f"File is missing required columns: {', '.join(missing_cols)}"
    for col in ['Latitude', 'Longitude', 'Orders', 'Hub ID', 'Number of Blocks']:  #Also validate the blocks
        df[col] = pd.to_numeric(df[col], errors='coerce')
    if df.isnull().values.any():
        return "File contains non-numeric or empty values in required numeric columns (Lat, Lon, Orders, Hub ID, Number of Blocks). Please check." #Update message
    return None

def calculate_route_distance(coord1, coord2, circuity_factor):
    """Calculates the estimated driving distance using a circuity factor."""
    return haversine(coord1, coord2) * circuity_factor

def get_delivery_sequence(points, depot_coord, circuity_factor):
    """Calculates the delivery sequence and total distance using a more robust algorithm."""
    if not points: return [], 0.0
    point_coords = {p['Society ID']: (p['Latitude'], p['Longitude']) for p in points}
    if len(points) == 1:
        point_id = list(point_coords.keys())[0]
        dist = calculate_route_distance(depot_coord, point_coords[point_id], circuity_factor) * 2
        return [point_id], round(dist, 2)
    best_path, best_distance = [], float('inf')
    for start_node_id in point_coords.keys():
        current_path, unvisited_ids = [start_node_id], set(point_coords.keys()) - {start_node_id}
        current_distance = calculate_route_distance(depot_coord, point_coords[start_node_id], circuity_factor)
        current_point_id = start_node_id
        while unvisited_ids:
            current_coord = point_coords[current_point_id]
            nearest_neighbor_id = min(unvisited_ids, key=lambda pid: calculate_route_distance(current_coord, point_coords[pid], circuity_factor))
            current_distance += calculate_route_distance(current_coord, point_coords[nearest_neighbor_id], circuity_factor)
            current_path.append(nearest_neighbor_id)
            unvisited_ids.remove(nearest_neighbor_id)
            current_point_id = nearest_neighbor_id
        current_distance += calculate_route_distance(point_coords[current_point_id], depot_coord, circuity_factor)
        if current_distance < best_distance:
            best_distance, best_path = current_distance, current_path
    return best_path, round(best_distance, 2)

def get_distance_to_last_society(points, depot_coord, circuity_factor):
    """Calculates depot-to-last-society distance using the circuity factor."""
    if not points: return 0.0
    path, _ = get_delivery_sequence(points, depot_coord, circuity_factor)
    if not path: return 0.0
    last_society_id = path[-1]
    point_coords = {p['Society ID']: (p['Latitude'], p['Longitude']) for p in points}
    return round(calculate_route_distance(depot_coord, point_coords[last_society_id], circuity_factor), 2)

@st.cache_data
def run_clustering(df, depot_lat, depot_lon, costs, circuity_factor):
    """A robust, multi-pass, prioritized clustering algorithm using the hybrid distance approach."""
    all_clusters, cluster_id_counter = [], 1
    societies_map = {s['Society ID']: s for s in df.to_dict('records')}
    unprocessed_ids = set(societies_map.keys())
    depot_coord = (depot_lat, depot_lon)
    for hub_name in df['Hub Name'].unique():
        hub_society_ids = {sid for sid in unprocessed_ids if societies_map[sid]['Hub Name'] == hub_name}
        for cluster_type in ['Main', 'Mini', 'Micro']:
            # **Sort by Orders, then Number of Blocks (ascending)**
            sorted_seeds = sorted(list(hub_society_ids), key=lambda sid: (societies_map[sid]['Orders'], societies_map[sid]['Number of Blocks']), reverse=True) # Less blocks first - also include order for priority
            for seed_id in sorted_seeds:
                if seed_id not in hub_society_ids: continue
                seed = societies_map[seed_id]
                potential_cluster, potential_orders = [seed], seed['Orders']
                if cluster_type in ['Main', 'Mini']:
                    max_orders, proximity = (220, 2.0) if cluster_type == 'Main' else (179, 2.0)
                    neighbors = [societies_map[nid] for nid in hub_society_ids if nid != seed_id and haversine((seed['Latitude'], seed['Longitude']), (societies_map[nid]['Latitude'], societies_map[nid]['Longitude'])) < proximity]
                else: # Micro
                    max_orders = 120
                    neighbors = [societies_map[nid] for nid in hub_society_ids if nid != seed_id]
                # **Sort by Orders, then Number of Blocks (ascending)**
                for neighbor in sorted(neighbors, key=lambda s: (s['Orders'], s['Number of Blocks']), reverse=True):
                    if potential_orders + neighbor['Orders'] <= max_orders:
                        potential_cluster.append(neighbor); potential_orders += neighbor['Orders']
                valid = False
                if cluster_type == 'Main' and 180 <= potential_orders <= 220: valid = True
                elif cluster_type == 'Mini' and 121 <= potential_orders <= 179: valid = True
                elif cluster_type == 'Micro' and 1 <= potential_orders <= 120 and get_distance_to_last_society(potential_cluster, depot_coord, circuity_factor) < 15.0: valid = True
                if valid:
                    path, distance = get_delivery_sequence(potential_cluster, depot_coord, circuity_factor)
                    all_clusters.append({'Cluster ID': f"{cluster_type}-{cluster_id_counter}", 'Type': cluster_type, 'Societies': potential_cluster, 'Orders': potential_orders, 'Distance': distance, 'Path': path, 'Cost': costs[cluster_type.lower()], 'Hub Name': hub_name})
                    cluster_id_counter += 1; hub_society_ids -= {s['Society ID'] for s in potential_cluster}
        for sid in hub_society_ids:
            society = societies_map[sid]
            path, distance = get_delivery_sequence([society], depot_coord, circuity_factor)
            all_clusters.append({'Cluster ID': f"Unclustered-{sid}", 'Type': 'Unclustered', 'Societies': [society], 'Orders': society['Orders'], 'Distance': distance, 'Path': path, 'Cost': 0, 'Hub Name': hub_name})
        unprocessed_ids -= {s['Society ID'] for s in df[df['Hub Name'] == hub_name].to_dict('records')}
    return all_clusters

def create_summary_df(clusters, depot_coord, circuity_factor, df_raw):
    """Creates summary DataFrame, with the delivery sequence showing only societies."""
    summary_rows = []
    society_id_to_blocks = df_raw.set_index('Society ID')['Number of Blocks'].to_dict() # Get blocks to add to the table.
    for c in clusters:
        total_orders, cpo = c['Orders'], (c['Cost'] / c['Orders']) if c['Orders'] > 0 else 0
        id_to_name, id_to_coord = {s['Society ID']: s['Society Name'] for s in c['Societies']}, {s['Society ID']: (s['Latitude'], s['Longitude']) for s in c['Societies']}
        internal_distance = 0.0
        if c['Path'] and len(c['Path']) > 0:
            total_distance = c['Distance']
            first_stop_coord = id_to_coord[c['Path'][0]]
            last_stop_coord = id_to_coord[c['Path'][-1]]
            dist_depot_to_first = calculate_route_distance(depot_coord, first_stop_coord, circuity_factor)
            dist_last_to_depot = calculate_route_distance(last_stop_coord, depot_coord, circuity_factor)
            internal_distance = total_distance - dist_depot_to_first - dist_last_to_depot
            internal_distance = max(0, internal_distance)
        delivery_sequence_str = ""
        if c['Path']:
            if len(c['Path']) == 1:
                delivery_sequence_str = id_to_name.get(c['Path'][0], "N/A")
            else:
                first_society_name = id_to_name.get(c['Path'][0], "N/A")
                sequence_parts = [first_society_name]
                for i in range(1, len(c['Path'])):
                    start_coord, end_coord = id_to_coord[c['Path'][i-1]], id_to_coord[c['Path'][i]]
                    end_name = id_to_name.get(c['Path'][i])
                    dist = calculate_route_distance(start_coord, end_coord, circuity_factor)
                    sequence_parts.append(f" -> {end_name} ({dist:.2f} km)")
                delivery_sequence_str = "".join(sequence_parts)
        # Include blocks in the summary
        society_ids_in_cluster = [s['Society ID'] for s in c['Societies']]
        blocks_str = ", ".join([str(society_id_to_blocks.get(sid, 'N/A')) for sid in society_ids_in_cluster])
        summary_rows.append({'Cluster ID': c['Cluster ID'], 'Cluster Type': c['Type'], 'No. of Societies': len(c['Societies']), 'Total Orders': total_orders, 'Total Distance Fwd + Rev Leg (km)': c['Distance'], 'Distance Between the Societies (km)': round(internal_distance, 2), 'CPO (in Rs.)': round(cpo, 2), 'Delivery Sequence': delivery_sequence_str, 'Number of Blocks': blocks_str})
    return pd.DataFrame(summary_rows)

def create_unified_map(clusters, depot_coord, circuity_factor, use_ant_path=False):
    """Creates the map. If use_ant_path is True, it draws animated directional lines with distance tooltips."""
    m = folium.Map(location=depot_coord, zoom_start=12, tiles="CartoDB positron")
    folium.Marker(depot_coord, popup="Depot", icon=folium.Icon(color='black', icon='industry', prefix='fa')).add_to(m)
    colors = {'Main': 'blue', 'Mini': 'green', 'Micro': 'purple', 'Unclustered': 'red'}
    for c in clusters:
        color, fg = colors.get(c['Type'], 'gray'), folium.FeatureGroup(name=f"{c['Cluster ID']} ({c['Type']})")
        id_to_name, id_to_coord = {s['Society ID']: s['Society Name'] for s in c['Societies']}, {s['Society ID']: (s['Latitude'], s['Longitude']) for s in c['Societies']}
        full_path_info = [("Depot", depot_coord)] + ([(id_to_name.get(sid, str(sid)), id_to_coord.get(sid)) for sid in c['Path']] if c['Path'] else []) + [("Depot", depot_coord)]
        path_locations = [coord for name, coord in full_path_info if coord is not None]
        if len(path_locations) > 1:
            if use_ant_path:
                AntPath(locations=path_locations, delay=800, dash_array=[20, 30], color=color, weight=5, pulse_color="#DDDDDD").add_to(fg)
                for i in range(len(full_path_info) - 1):
                    start_name, start_coord, end_name, end_coord = *full_path_info[i], *full_path_info[i+1]
                    if start_coord and end_coord:
                        dist = calculate_route_distance(start_coord, end_coord, circuity_factor)
                        folium.PolyLine(locations=[start_coord, end_coord], tooltip=f"{start_name} to {end_name}: {dist:.2f} km", color=color, weight=2.5, opacity=0.8).add_to(fg)
            else:
                folium.PolyLine(locations=path_locations, color=color, weight=2.5, opacity=0.8).add_to(fg)
        for society in c['Societies']:
            popup_text = f"<b>{society['Society Name']}</b><br>Orders: {society['Orders']}<br>Cluster: {c['Cluster ID']}<br>Hub: {society['Hub Name']}"
            folium.Marker(location=[society['Latitude'], society['Longitude']], popup=popup_text, icon=folium.Icon(color=color, icon='info-sign')).add_to(fg)
        fg.add_to(m)
    folium.LayerControl().add_to(m)
    return m

# --- STREAMLIT UI ---
st.markdown("<div style='text-align: center;'><h1> 🚚 RK - Delivery Cluster Optimizer and Sequencing</h1></div>", unsafe_allow_html=True)

with st.sidebar:
    st.header("1. Depot Settings")
    def update_depot_from_selection():
        city = st.session_state.city_selector
        if city != "Custom":
            st.session_state.depot_lat, st.session_state.depot_lon = PREDEFINED_DEPOTS[city]["lat"], PREDEFINED_DEPOTS[city]["lon"]
    if "depot_lat" not in st.session_state:
        first_city = list(PREDEFINED_DEPOTS.keys())[0]
        st.session_state.depot_lat, st.session_state.depot_lon = PREDEFINED_DEPOTS[first_city]["lat"], PREDEFINED_DEPOTS[first_city]["lon"]
        if "city_selector" not in st.session_state:
            st.session_state.city_selector = first_city
    st.selectbox("Select a Predefined Depot", options=list(PREDEFINED_DEPOTS.keys()) + ["Custom"], key="city_selector", on_change=update_depot_from_selection)
    depot_lat = st.number_input("Depot Latitude", key="depot_lat", format="%.6f")
    depot_long = st.number_input("Depot Longitude", key="depot_lon", format="%.6f")
    st.caption("Select a depot to pre-fill coordinates, or choose 'Custom' and edit them manually.")
    depot_coord = (depot_lat, depot_long)
    st.header("2. Routing & Cost Settings")
    circuity_factor = st.slider("Circuity Factor (for driving distance estimation)", 1.0, 2.0, 1.4, 0.1, help="Adjust to estimate driving distance from straight-line distance. 1.4 = 40% longer.")
    st.subheader("Main (180 to 220) Costs")
    main_van_cost, main_cee_cost = st.number_input("Main Van Cost (₹)", value=833, min_value=0, key="main_van"), st.number_input("Main CEE Cost (₹)", value=333, min_value=0, key="main_cee")
    st.subheader("Mini (121 to 179) Costs")
    mini_van_cost, mini_cee_cost = st.number_input("Mini Van Cost (₹)", value=1000, min_value=0, key="mini_van"), st.number_input("Mini CEE Cost (₹)", value=200, min_value=0, key="mini_cee")
    st.subheader("Micro (1 to 120) Costs")
    micro_van_cost, micro_cee_cost = st.number_input("Micro Van Cost (₹)", value=500, min_value=0, key="micro_van"), st.number_input("Micro CEE Cost (₹)", value=200, min_value=0, key="micro_cee")
    costs = {'main': main_van_cost + main_cee_cost, 'mini': mini_van_cost + mini_cee_cost, 'micro': micro_van_cost + micro_cee_cost}
    st.header("3. Upload Data")
    template_df = pd.DataFrame(columns=['Society ID', 'Society Name', 'Latitude', 'Longitude', 'Orders', 'Hub ID', 'Hub Name', 'Number of Blocks']) #Added block column
    st.download_button("Download Template CSV", template_df.to_csv(index=False).encode('utf-8'), 'input_template.csv', 'text/csv')
    file = st.file_uploader("Upload Society Data CSV", type=["csv"])

if file is None:
    st.info("Please upload a society data file to begin analysis."); st.stop()
try:
    with st.spinner("Reading and validating data..."):
        df_raw = pd.read_csv(file, encoding='utf-8') # First attempt
        validation_error = validate_columns(df_raw)
        if validation_error: st.error(validation_error); st.stop()
except UnicodeDecodeError:
    st.warning("UTF-8 decoding failed. Trying a different encoding (latin-1).")
    file.seek(0)  # Reset file pointer to the beginning
    try:
        with st.spinner("Reading and validating data... (using latin-1 encoding)"):
            df_raw = pd.read_csv(file, encoding='latin-1') # Fallback encoding
            validation_error = validate_columns(df_raw)
            if validation_error: st.error(validation_error); st.stop()
    except Exception as e:
        st.error(f"Error reading or parsing file with latin-1 encoding: {e}"); st.stop()
except Exception as e:
    st.error(f"Error reading or parsing file: {e}"); st.stop()

if 'clusters' not in st.session_state:
    st.session_state.clusters, st.session_state.last_run_depot_coord = None, None

if st.button("🚀 Generate Clusters", type="primary"):
    with st.spinner("Analyzing data and forming clusters..."):
        st.session_state.clusters = run_clustering(df_raw, depot_lat, depot_long, costs, circuity_factor)
        st.session_state.last_run_depot_coord = depot_coord

if st.session_state.get('clusters') is not None:
    if st.session_state.last_run_depot_coord != depot_coord:
        st.warning("Depot settings have changed. The displayed results are for the previous location. Please click 'Generate Clusters' to update.")
    with st.container():
        all_clusters = st.session_state.clusters
        full_summary_df = create_summary_df(all_clusters, depot_coord, circuity_factor, df_raw) #Pass df_raw to add block details
        st.header("📊 Overall Cluster Summary")
        column_order = ['Cluster ID', 'Cluster Type', 'No. of Societies', 'Total Orders', 'Total Distance Fwd + Rev Leg (km)', 'Distance Between the Societies (km)', 'CPO (in Rs.)', 'Delivery Sequence', 'Number of Blocks'] #Include 'Number of Blocks'
        st.dataframe(full_summary_df.sort_values(by=['Cluster Type', 'Cluster ID'])[column_order])
        st.download_button("Download Full Summary (CSV)", full_summary_df[column_order].to_csv(index=False).encode('utf-8'), "cluster_summary.csv", "text/csv")
        st.header("📈 Overall Cumulative Summary")
        temp_df_all = full_summary_df.copy()
        temp_df_all['Total Cost'] = temp_df_all['CPO (in Rs.)'] * temp_df_all['Total Orders']
        cumulative_summary_all = temp_df_all.groupby('Cluster Type').agg(Total_Routes=('Cluster ID', 'count'), Total_Societies=('No. of Societies', 'sum'), Total_Orders=('Total Orders', 'sum'), Total_Cost=('Total Cost', 'sum')).reset_index()
        cumulative_summary_all['Overall CPO (in Rs.)'] = (cumulative_summary_all['Total_Cost'] / cumulative_summary_all['Total_Orders']).round(2)
        st.dataframe(cumulative_summary_all[['Cluster Type', 'Total_Routes', 'Total_Societies', 'Total_Orders', 'Overall CPO (in Rs.)']])
        st.header("🗺️ Unified Map View")
        st.info("You can toggle clusters on/off using the layer control icon in the top-right of the map.")
        st_folium(create_unified_map(all_clusters, depot_coord, circuity_factor), width=1200, height=600, returned_objects=[])
        st.divider()
        st.header("🔍 Drill-Down & Detailed View")
        hub_names = ['All Hubs'] + sorted(df_raw['Hub Name'].unique().tolist())
        selected_hub = st.selectbox("Filter by Hub Name to Drill Down", options=hub_names)
        clusters_for_drilldown = [c for c in all_clusters if c.get('Hub Name') == selected_hub] if selected_hub != "All Hubs" else all_clusters
        cluster_id_options = sorted([c['Cluster ID'] for c in clusters_for_drilldown])
        if not cluster_id_options:
            st.warning("No individual clusters to select for this hub.")
        else:
            cluster_id_to_show = st.selectbox("Select a Cluster to Inspect", cluster_id_options)
            selected_cluster = next((c for c in clusters_for_drilldown if c['Cluster ID'] == cluster_id_to_show), None)
            if selected_cluster:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader(f"Details for {selected_cluster['Cluster ID']}")
                    cluster_details_df_unsorted = pd.DataFrame(selected_cluster['Societies'])
                    delivery_path = selected_cluster['Path']
                    cluster_details_df = cluster_details_df_unsorted.loc[cluster_details_df_unsorted['Society ID'].isin(delivery_path)].set_index('Society ID').loc[delivery_path].reset_index() if delivery_path else cluster_details_df_unsorted
                    cluster_orders, cluster_cost = selected_cluster['Orders'], selected_cluster['Cost']
                    cluster_cpo = (cluster_cost / cluster_orders) if cluster_orders > 0 else 0
                    cluster_details_df['CPO (in Rs.)'] = round(cluster_cpo, 2)
                    cluster_details_df.rename(columns={'Orders': 'Total Orders'}, inplace=True)
                    detail_cols = ['Society ID', 'Society Name', 'Hub ID', 'Hub Name', 'Total Orders', 'CPO (in Rs.)', 'Number of Blocks']  #Include blocks
                    st.dataframe(cluster_details_df[detail_cols])
                    st.download_button(f"Download Details for {selected_cluster['Cluster ID']}", cluster_details_df[detail_cols].to_csv(index=False).encode('utf-8'), f"cluster_{selected_cluster['Cluster ID']}_details.csv", "text/csv")
                with col2:
                    st.subheader("Route Map")
                    st_folium(create_unified_map([selected_cluster], depot_coord, circuity_factor, use_ant_path=True), width=600, height=400, returned_objects=[])
