import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
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

def get_delivery_sequence(points, depot_coord):
    """Calculates the delivery sequence (path) and total round-trip distance for a cluster."""
    if not points: return [], 0.0
    point_coords = {p['Society ID']: (p['Latitude'], p['Longitude']) for p in points}
    current_coord, unvisited_ids, path, total_distance = depot_coord, set(point_coords.keys()), [], 0.0
    while unvisited_ids:
        nearest_id = min(unvisited_ids, key=lambda pid: haversine(current_coord, point_coords[pid]))
        total_distance += haversine(current_coord, point_coords[nearest_id])
        current_coord = point_coords[nearest_id]
        path.append(nearest_id)
        unvisited_ids.remove(nearest_id)
    total_distance += haversine(current_coord, depot_coord)
    return path, round(total_distance, 2)

def get_distance_to_last_society(points, depot_coord):
    """Calculates the TSP path and then finds the straight-line distance from the depot to the last society in that path."""
    if not points: return 0.0
    path, _ = get_delivery_sequence(points, depot_coord)
    if not path: return 0.0
    last_society_id = path[-1]
    point_coords = {p['Society ID']: (p['Latitude'], p['Longitude']) for p in points}
    return round(haversine(depot_coord, point_coords[last_society_id]), 2)

@st.cache_data
def run_clustering(df, depot_lat, depot_lon, costs):
    """A robust, multi-pass, prioritized clustering algorithm."""
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
                elif cluster_type == 'Micro' and 1 <= potential_orders <= 120 and get_distance_to_last_society(potential_cluster, depot_coord) < 15.0: valid = True
                
                if valid:
                    path, distance = get_delivery_sequence(potential_cluster, depot_coord)
                    all_clusters.append({'Cluster ID': f"{cluster_type}-{cluster_id_counter}", 'Type': cluster_type, 'Societies': potential_cluster, 'Orders': potential_orders, 'Distance': distance, 'Path': path, 'Cost': costs[cluster_type.lower()]})
                    cluster_id_counter += 1; hub_society_ids -= {s['Society ID'] for s in potential_cluster}
        for sid in hub_society_ids:
            society = societies_map[sid]
            path, distance = get_delivery_sequence([society], depot_coord)
            all_clusters.append({'Cluster ID': f"Unclustered-{sid}", 'Type': 'Unclustered', 'Societies': [society], 'Orders': society['Orders'], 'Distance': distance, 'Path': path, 'Cost': 0})
        unprocessed_ids -= {s['Society ID'] for s in df[df['Hub ID'] == hub_id].to_dict('records')}
    return all_clusters

def create_summary_df(clusters, depot_coord):
    """
    Creates the summary DataFrame with the new requested column headers.
    """
    summary_rows = []
    for c in clusters:
        total_orders = c['Orders']
        cpo = (c['Cost'] / total_orders) if total_orders > 0 else 0
        
        id_to_name = {s['Society ID']: s['Society Name'] for s in c['Societies']}
        id_to_coord = {s['Society ID']: (s['Latitude'], s['Longitude']) for s in c['Societies']}
        
        internal_distance = 0.0
        if len(c['Path']) > 1:
            for i in range(len(c['Path']) - 1):
                internal_distance += haversine(id_to_coord[c['Path'][i]], id_to_coord[c['Path'][i+1]])

        delivery_sequence_str = "Depot"
        if c['Path']:
            nodes = [("Depot", depot_coord)] + [(id_to_name.get(sid), id_to_coord.get(sid)) for sid in c['Path']]
            for i in range(1, len(nodes)):
                start_coord = nodes[i-1][1]
                end_name = nodes[i][0]
                end_coord = nodes[i][1]
                dist = haversine(start_coord, end_coord)
                delivery_sequence_str += f" -> {end_name} ({dist:.2f} km)"
            last_node_coord = nodes[-1][1]
            dist_to_depot = haversine(last_node_coord, depot_coord)
            delivery_sequence_str += f" -> Depot ({dist_to_depot:.2f} km)"
        else:
             society_name = c['Societies'][0]['Society Name']
             society_coord = id_to_coord[c['Societies'][0]['Society ID']]
             dist = haversine(depot_coord, society_coord)
             delivery_sequence_str = f"Depot -> {society_name} ({dist:.2f} km) -> Depot ({dist:.2f} km)"

        summary_rows.append({
            'Cluster ID': c['Cluster ID'],
            'Cluster Type': c['Type'],
            'No. of Societies': len(c['Societies']),
            'Total Orders': total_orders,
            'Total Distance Fwd + Rev Leg (km)': c['Distance'],             # RENAMED
            'Distance Between the Societies (km)': round(internal_distance, 2), # RENAMED
            'CPO (in Rs.)': round(cpo, 2),                                  # RENAMED
            'Delivery Sequence': delivery_sequence_str,
        })
    return pd.DataFrame(summary_rows)

def create_unified_map(clusters, depot_coord):
    """Creates the map with individual line segments showing distance for each leg of the journey."""
    m = folium.Map(location=depot_coord, zoom_start=12, tiles="CartoDB positron")
    folium.Marker(depot_coord, popup="Depot", icon=folium.Icon(color='black', icon='industry', prefix='fa')).add_to(m)
    colors = {'Main': 'blue', 'Mini': 'green', 'Micro': 'purple', 'Unclustered': 'red'}
    for c in clusters:
        color, fg = colors.get(c['Type'], 'gray'), folium.FeatureGroup(name=f"{c['Cluster ID']} ({c['Type']})")
        id_to_name, id_to_coord = {s['Society ID']: s['Society Name'] for s in c['Societies']}, {s['Society ID']: (s['Latitude'], s['Longitude']) for s in c['Societies']}
        full_path_info = [("Depot", depot_coord)] + ([(id_to_name.get(sid, str(sid)), id_to_coord.get(sid)) for sid in c['Path']] if c['Path'] else []) + [("Depot", depot_coord)]
        for i in range(len(full_path_info) - 1):
            start_name, start_coord = full_path_info[i]; end_name, end_coord = full_path_info[i+1]
            if start_coord and end_coord:
                folium.PolyLine(locations=[start_coord, end_coord], color=color, weight=2.5, opacity=0.8, tooltip=f"{start_name} to {end_name}: {haversine(start_coord, end_coord):.2f} km").add_to(fg)
        for society in c['Societies']: folium.Marker(location=[society['Latitude'], society['Longitude']], popup=f"<b>{society['Society Name']}</b><br>Orders: {society['Orders']}<br>Cluster: {c['Cluster ID']}", icon=folium.Icon(color=color, icon='info-sign')).add_to(fg)
        fg.add_to(m)
    folium.LayerControl().add_to(m)
    return m


# --- STREAMLIT UI ---
st.title("🚚 Logistics Cluster Optimizer")

with st.sidebar:
    st.header("1. Depot Settings")
    depot_lat = st.number_input("Depot Latitude", value=12.9716, format="%.6f")
    depot_long = st.number_input("Depot Longitude", value=77.5946, format="%.6f")
    depot_coord = (depot_lat, depot_long)

    st.header("2. Cluster Costs")
    costs = {'main': st.number_input("Main Cluster Van Cost (₹)", 833) + st.number_input("Main Cluster CEE Cost (₹)", 333),
             'mini': st.number_input("Mini Cluster Van Cost (₹)", 1000) + st.number_input("Mini Cluster CEE Cost (₹)", 200),
             'micro': st.number_input("Micro Cluster Van Cost (₹)", 500) + st.number_input("Micro Cluster CEE Cost (₹)", 200)}

    st.header("3. Upload Data")
    template = pd.DataFrame({'Society ID':[], 'Society Name':[], 'Latitude':[], 'Longitude':[], 'Orders':[], 'Hub ID':[]})
    st.download_button("Download Template", template.to_csv(index=False), "input_template.csv")
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
        st.session_state.clusters = run_clustering(df_raw, depot_lat, depot_long, costs)

if st.session_state.get('clusters') is not None:
    clusters = st.session_state.clusters
    
    summary_df = create_summary_df(clusters, depot_coord) 
    st.header("📊 Cluster Summary")
    
    # Define the new column order for better readability
    column_order = [
        'Cluster ID', 
        'Cluster Type', 
        'No. of Societies', 
        'Total Orders', 
        'Total Distance Fwd + Rev Leg (km)',      # RENAMED
        'Distance Between the Societies (km)',    # RENAMED
        'CPO (in Rs.)',                           # RENAMED
        'Delivery Sequence'
    ]
    st.dataframe(summary_df.sort_values(by=['Cluster Type', 'Cluster ID'])[column_order])
    
    # Ensure the downloaded CSV also uses the new column order
    csv_buffer = BytesIO(); summary_df[column_order].to_csv(csv_buffer, index=False, encoding='utf-8')
    st.download_button("Download Full Summary (CSV)", csv_buffer.getvalue(), "cluster_summary.csv", "text/csv")

    st.header("🗺️ Unified Map View")
    st.info("You can toggle clusters on/off using the layer control icon in the top-right of the map.")
    st_folium(create_unified_map(clusters, depot_coord), width=1200, height=600, returned_objects=[])

    st.header("🔍 Individual Cluster Details")
    cluster_id_to_show = st.selectbox("Select a Cluster to Inspect", sorted(summary_df['Cluster ID'].tolist()))
    selected_cluster = next((c for c in clusters if c['Cluster ID'] == cluster_id_to_show), None)

    if selected_cluster:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"Details for {selected_cluster['Cluster ID']}")
            cluster_details_df = pd.DataFrame(selected_cluster['Societies'])
            st.dataframe(cluster_details_df[['Society ID', 'Society Name', 'Orders']])
            detail_csv_buffer = BytesIO(); cluster_details_df.to_csv(detail_csv_buffer, index=False, encoding='utf-8')
            st.download_button(f"Download Details for {selected_cluster['Cluster ID']}", detail_csv_buffer.getvalue(), f"cluster_{selected_cluster['Cluster ID']}_details.csv", "text/csv")
        with col2:
            st.subheader("Route Map"); st_folium(create_unified_map([selected_cluster], depot_coord), width=600, height=400, returned_objects=[])
