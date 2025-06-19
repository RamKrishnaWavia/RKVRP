import streamlit as st
import pandas as pd
import folium
from folium.plugins import PolyLineTextPath # Import the new plugin
from streamlit_folium import st_folium
from haversine import haversine, Unit
from io import BytesIO

st.set_page_config(layout="wide")

# --- UTILITY & ALGORITHM FUNCTIONS ---
# (These functions are correct and remain unchanged)

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

def calculate_route_distance(coord1, coord2, circuity_factor):
    """Calculates the estimated driving distance using a circuity factor."""
    return haversine(coord1, coord2) * circuity_factor

def get_delivery_sequence(points, depot_coord, circuity_factor):
    """Calculates the delivery sequence and total distance, using the circuity factor."""
    if not points: return [], 0.0
    point_coords = {p['Society ID']: (p['Latitude'], p['Longitude']) for p in points}
    current_coord, unvisited_ids, path, total_distance = depot_coord, set(point_coords.keys()), [], 0.0
    while unvisited_ids:
        nearest_id = min(unvisited_ids, key=lambda pid: haversine(current_coord, point_coords[pid]))
        total_distance += calculate_route_distance(current_coord, point_coords[nearest_id], circuity_factor)
        current_coord = point_coords[nearest_id]
        path.append(nearest_id)
        unvisited_ids.remove(nearest_id)
    total_distance += calculate_route_distance(current_coord, depot_coord, circuity_factor)
    return path, round(total_distance, 2)

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
    for hub_id in df['Hub ID'].unique():
        hub_society_ids = {sid for sid in unprocessed_ids if societies_map[sid]['Hub ID'] == hub_id}
        for cluster_type in ['Main', 'Mini', 'Micro']:
            sorted_seeds = sorted(list(hub_society_ids), key=lambda sid: societies_map[sid]['Orders'], reverse=True)
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
                for neighbor in sorted(neighbors, key=lambda s: s['Orders'], reverse=True):
                    if potential_orders + neighbor['Orders'] <= max_orders:
                        potential_cluster.append(neighbor); potential_orders += neighbor['Orders']
                
                valid = False
                if cluster_type == 'Main' and 180 <= potential_orders <= 220: valid = True
                elif cluster_type == 'Mini' and 121 <= potential_orders <= 179: valid = True
                elif cluster_type == 'Micro' and 1 <= potential_orders <= 120 and get_distance_to_last_society(potential_cluster, depot_coord, circuity_factor) < 15.0: valid = True
                
                if valid:
                    path, distance = get_delivery_sequence(potential_cluster, depot_coord, circuity_factor)
                    all_clusters.append({'Cluster ID': f"{cluster_type}-{cluster_id_counter}", 'Type': cluster_type, 'Societies': potential_cluster, 'Orders': potential_orders, 'Distance': distance, 'Path': path, 'Cost': costs[cluster_type.lower()]})
                    cluster_id_counter += 1; hub_society_ids -= {s['Society ID'] for s in potential_cluster}
        for sid in hub_society_ids:
            society = societies_map[sid]
            path, distance = get_delivery_sequence([society], depot_coord, circuity_factor)
            all_clusters.append({'Cluster ID': f"Unclustered-{sid}", 'Type': 'Unclustered', 'Societies': [society], 'Orders': society['Orders'], 'Distance': distance, 'Path': path, 'Cost': 0})
        unprocessed_ids -= {s['Society ID'] for s in df[df['Hub ID'] == hub_id].to_dict('records')}
    return all_clusters

def create_summary_df(clusters, depot_coord, circuity_factor):
    """Creates summary DataFrame, with the delivery sequence showing only societies."""
    summary_rows = []
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
                    start_coord = id_to_coord[c['Path'][i-1]]
                    end_name = id_to_name.get(c['Path'][i])
                    end_coord = id_to_coord[c['Path'][i]]
                    dist = calculate_route_distance(start_coord, end_coord, circuity_factor)
                    sequence_parts.append(f" -> {end_name} ({dist:.2f} km)")
                delivery_sequence_str = "".join(sequence_parts)
        summary_rows.append({'Cluster ID': c['Cluster ID'], 'Cluster Type': c['Type'], 'No. of Societies': len(c['Societies']), 'Total Orders': total_orders, 'Total Distance Fwd + Rev Leg (km)': c['Distance'], 'Distance Between the Societies (km)': round(internal_distance, 2), 'CPO (in Rs.)': round(cpo, 2), 'Delivery Sequence': delivery_sequence_str})
    return pd.DataFrame(summary_rows)

def create_unified_map(clusters, depot_coord, circuity_factor, use_arrows=False):
    """
    Creates the map. If use_arrows is True, it draws directional arrows on the lines.
    """
    m = folium.Map(location=depot_coord, zoom_start=12, tiles="CartoDB positron")
    folium.Marker(depot_coord, popup="Depot", icon=folium.Icon(color='black', icon='industry', prefix='fa')).add_to(m)
    colors = {'Main': 'blue', 'Mini': 'green', 'Micro': 'purple', 'Unclustered': 'red'}
    for c in clusters:
        color, fg = colors.get(c['Type'], 'gray'), folium.FeatureGroup(name=f"{c['Cluster ID']} ({c['Type']})")
        id_to_name, id_to_coord = {s['Society ID']: s['Society Name'] for s in c['Societies']}, {s['Society ID']: (s['Latitude'], s['Longitude']) for s in c['Societies']}
        full_path_info = [("Depot", depot_coord)] + ([(id_to_name.get(sid, str(sid)), id_to_coord.get(sid)) for sid in c['Path']] if c['Path'] else []) + [("Depot", depot_coord)]
        
        path_locations = [coord for name, coord in full_path_info if coord is not None]

        if len(path_locations) > 1:
            # Draw the base line for the route
            folium.PolyLine(
                locations=path_locations,
                color=color,
                weight=2.5,
                opacity=0.8
            ).add_to(fg)
            
            # If requested, add the directional arrows on top of the line
            if use_arrows:
                PolyLineTextPath(
                    path_locations,
                    '  ▶  ',  # Using a unicode arrow character
                    repeat=True,
                    offset=0,
                    attributes={'font-size': '16', 'fill': color, 'font-weight': 'bold'}
                ).add_to(fg)

        for society in c['Societies']:
            folium.Marker(location=[society['Latitude'], society['Longitude']], popup=f"<b>{society['Society Name']}</b><br>Orders: {society['Orders']}<br>Cluster: {c['Cluster ID']}", icon=folium.Icon(color=color, icon='info-sign')).add_to(fg)
        fg.add_to(m)
    folium.LayerControl().add_to(m)
    return m

# --- STREAMLIT UI ---
st.markdown("<div style='text-align: center;'><h1> 🚚 RK - Delivery Cluster Optimizer and Sequencing</h1></div>", unsafe_allow_html=True)

with st.sidebar:
    st.header("1. Depot Settings")
    depot_lat = st.number_input("Depot Latitude", value=12.9716, format="%.6f")
    depot_long = st.number_input("Depot Longitude", value=77.5946, format="%.6f")
    depot_coord = (depot_lat, depot_long)

    st.header("2. Routing & Cost Settings")
    circuity_factor = st.slider("Circuity Factor (for driving distance estimation)", 1.0, 2.0, 1.4, 0.1, help="Adjust to estimate driving distance from straight-line distance. 1.4 = 40% longer.")
    
    costs = {'main': st.number_input("Main Cluster Van Cost (₹)", 833) + st.number_input("Main Cluster CEE Cost (₹)", 333),
             'mini': st.number_input("Mini Cluster Van Cost (₹)", 1000) + st.number_input("Mini Cluster CEE Cost (₹)", 200),
             'micro': st.number_input("Micro Cluster Van Cost (₹)", 500) + st.number_input("Micro Cluster CEE Cost (₹)", 200)}

    st.header("3. Upload Data")
    file = st.file_uploader("Upload Society Data CSV", type=["csv"])

if file is None:
    st.info("Please upload a society data file to begin analysis."); st.stop()
try:
    df_raw = pd.read_csv(file)
    validation_error = validate_columns(df_raw)
    if validation_error: st.error(validation_error); st.stop()
except Exception as e:
    st.error(f"Error reading or parsing file: {e}"); st.stop()

if 'clusters' not in st.session_state:
    st.session_state.clusters = None

if st.button("🚀 Generate Clusters", type="primary"):
    with st.spinner("Analyzing data and forming clusters..."):
        st.session_state.clusters = run_clustering(df_raw, depot_lat, depot_long, costs, circuity_factor)

if st.session_state.get('clusters') is not None:
    with st.container():
        clusters = st.session_state.clusters
        
        summary_df = create_summary_df(clusters, depot_coord, circuity_factor) 
        st.header("📊 Cluster Summary")
        column_order = ['Cluster ID', 'Cluster Type', 'No. of Societies', 'Total Orders', 'Total Distance Fwd + Rev Leg (km)', 'Distance Between the Societies (km)', 'CPO (in Rs.)', 'Delivery Sequence']
        st.dataframe(summary_df.sort_values(by=['Cluster Type', 'Cluster ID'])[column_order])
        csv_buffer = BytesIO(); summary_df[column_order].to_csv(csv_buffer, index=False, encoding='utf-8')
        st.download_button("Download Full Summary (CSV)", csv_buffer.getvalue(), "cluster_summary.csv", "text/csv")

        st.header("📈 Cumulative Summary by Cluster Type")
        temp_df = summary_df.copy()
        temp_df['Total Cost'] = temp_df['CPO (in Rs.)'] * temp_df['Total Orders']
        cumulative_summary = temp_df.groupby('Cluster Type').agg(
            Total_Routes=('Cluster ID', 'count'), Total_Societies=('No. of Societies', 'sum'),
            Total_Orders=('Total Orders', 'sum'), Total_Cost=('Total Cost', 'sum')).reset_index()
        cumulative_summary['Overall CPO (in Rs.)'] = (cumulative_summary['Total_Cost'] / cumulative_summary['Total_Orders']).round(2)
        st.dataframe(cumulative_summary[['Cluster Type', 'Total_Routes', 'Total_Societies', 'Total_Orders', 'Overall CPO (in Rs.)']])

        st.header("🗺️ Unified Map View")
        st.info("You can toggle clusters on/off using the layer control icon in the top-right of the map.")
        map_data = st_folium(create_unified_map(clusters, depot_coord, circuity_factor), width=1200, height=600, returned_objects=[])

        st.header("🔍 Individual Cluster Details")
        cluster_id_to_show = st.selectbox("Select a Cluster to Inspect", sorted(summary_df['Cluster ID'].tolist()))
        selected_cluster = next((c for c in clusters if c['Cluster ID'] == cluster_id_to_show), None)

        if selected_cluster:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader(f"Details for {selected_cluster['Cluster ID']}")
                cluster_details_df = pd.DataFrame(selected_cluster['Societies'])
                cluster_orders = selected_cluster['Orders']
                cluster_cost = selected_cluster['Cost']
                cluster_cpo = (cluster_cost / cluster_orders) if cluster_orders > 0 else 0
                cluster_details_df['CPO (in Rs.)'] = round(cluster_cpo, 2)
                cluster_details_df.rename(columns={'Orders': 'Total Orders'}, inplace=True)
                st.dataframe(cluster_details_df[['Society ID', 'Society Name', 'Total Orders', 'CPO (in Rs.)']])
                detail_csv_buffer = BytesIO(); cluster_details_df.to_csv(detail_csv_buffer, index=False, encoding='utf-8')
                st.download_button(f"Download Details for {selected_cluster['Cluster ID']}", detail_csv_buffer.getvalue(), f"cluster_{selected_cluster['Cluster ID']}_details.csv", "text/csv")
            with col2:
                st.subheader("Route Map")
                # --- THE FIX IS HERE ---
                # Call the map function with use_arrows=True for the individual map
                _ = st_folium(create_unified_map([selected_cluster], depot_coord, circuity_factor, use_arrows=True), width=600, height=400, returned_objects=[])
