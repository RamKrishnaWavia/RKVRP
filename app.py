import os
import pandas as pd
import numpy as np
import folium
import io

from flask import (Flask, render_template, request, redirect,
                   url_for, flash, send_file, session)
from werkzeug.utils import secure_filename
from haversine import haversine, Unit

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}
REQUIRED_COLUMNS = ['Society ID', 'Society Name', 'Latitude', 'Longitude', 'Orders', 'Hub ID']

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'a_very_secret_key_for_sessions'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- Helper Functions ---

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_columns(df):
    """Validate if the dataframe contains all required columns."""
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        return f"File is missing required columns: {', '.join(missing_cols)}"
    return None

def solve_tsp_nearest_neighbor(points, depot_coord):
    """
    Solves the TSP for a set of points using the Nearest Neighbor heuristic.
    Returns the ordered list of points (IDs) and total distance.
    """
    if not points:
        return [], 0.0

    point_coords = {p['id']: (p['lat'], p['lon']) for p in points}
    
    # Start at the depot
    current_point_id = 'Depot'
    current_coord = depot_coord
    
    unvisited_ids = set(point_coords.keys())
    path = ['Depot']
    total_distance = 0.0

    while unvisited_ids:
        nearest_neighbor_id = min(unvisited_ids, key=lambda pid: haversine(current_coord, point_coords[pid]))
        
        distance_to_nearest = haversine(current_coord, point_coords[nearest_neighbor_id], unit=Unit.KILOMETERS)
        total_distance += distance_to_nearest
        
        current_point_id = nearest_neighbor_id
        current_coord = point_coords[current_point_id]
        
        path.append(current_point_id)
        unvisited_ids.remove(current_point_id)
        
    return path, total_distance

# --- Main Clustering Logic ---

def perform_clustering(df, depot_coord, costs):
    """
    The core logic to create clusters based on predefined rules.
    """
    clusters = []
    unclustered_societies = []
    cluster_id_counter = 1

    # Group data by Hub ID, as clusters cannot span across hubs
    for hub_id, hub_group in df.groupby('Hub ID'):
        # Sort societies by order count to greedily form larger clusters first
        remaining_societies = hub_group.sort_values('Orders', ascending=False).to_dict('records')
        
        # --- Attempt to form MAIN clusters (>= 180 orders) ---
        while True:
            current_cluster_societies = []
            current_order_total = 0
            
            # Greedily add societies until orders >= 180
            temp_remaining = list(remaining_societies)
            for society in temp_remaining:
                if current_order_total < 180:
                    current_cluster_societies.append(society)
                    current_order_total += society['Orders']
                    remaining_societies.remove(society)
            
            if current_order_total >= 180:
                cluster_points = [{'id': s['Society ID'], 'lat': s['Latitude'], 'lon': s['Longitude']} for s in current_cluster_societies]
                path, distance = solve_tsp_nearest_neighbor(cluster_points, depot_coord)
                
                clusters.append({
                    'Cluster ID': f"C{cluster_id_counter}",
                    'Cluster Type': 'Main',
                    'Societies': current_cluster_societies,
                    'Total Orders': current_order_total,
                    'Total Distance (km)': distance,
                    'Delivery Sequence': path,
                    'Hub ID': hub_id,
                    'Cost': costs['main']
                })
                cluster_id_counter += 1
            else:
                # Put societies back if a valid Main cluster wasn't formed
                remaining_societies.extend(current_cluster_societies)
                break # Move to next cluster type

        # --- Attempt to form MINI clusters (121-179 orders) ---
        # This is a complex combinatorial problem. We'll use a simplified greedy approach.
        # Note: A more advanced solution would use bin packing or other optimization algorithms.
        while True:
            # Try to find a combination that fits the mini cluster criteria
            # For simplicity, we form one cluster at a time.
            best_combo = []
            best_combo_orders = 0
            
            # Let's take the largest remaining and see what fits
            if not remaining_societies: break
            
            temp_remaining = list(remaining_societies)
            base_society = temp_remaining.pop(0)
            
            current_combo = [base_society]
            current_orders = base_society['Orders']
            
            for society in temp_remaining:
                if 121 <= current_orders + society['Orders'] <= 179:
                    current_combo.append(society)
                    current_orders += society['Orders']
            
            if 121 <= current_orders <= 179:
                cluster_points = [{'id': s['Society ID'], 'lat': s['Latitude'], 'lon': s['Longitude']} for s in current_combo]
                path, distance = solve_tsp_nearest_neighbor(cluster_points, depot_coord)

                if distance <= 15: # Mini cluster distance constraint
                    clusters.append({
                        'Cluster ID': f"C{cluster_id_counter}",
                        'Cluster Type': 'Mini',
                        'Societies': current_combo,
                        'Total Orders': current_orders,
                        'Total Distance (km)': distance,
                        'Delivery Sequence': path,
                        'Hub ID': hub_id,
                        'Cost': costs['mini']
                    })
                    cluster_id_counter += 1
                    # Remove used societies from the main pool
                    for society in current_combo:
                        if society in remaining_societies:
                            remaining_societies.remove(society)
                else:
                    # If it fails distance check, try again without this combo
                    # For this simple model, we'll just move on.
                    break
            else:
                break # No suitable mini cluster found in this iteration

        # --- Attempt to form MICRO clusters (<= 120 orders) ---
        while remaining_societies:
            base_society = remaining_societies.pop(0)
            current_cluster_societies = [base_society]
            current_order_total = base_society['Orders']

            # Find nearby societies to add to the micro cluster
            temp_remaining = list(remaining_societies)
            for other_society in temp_remaining:
                # Proximity check: every society must be close to the base society
                dist_to_base = haversine((base_society['Latitude'], base_society['Longitude']),
                                         (other_society['Latitude'], other_society['Longitude']), unit=Unit.KILOMETERS)
                
                if (current_order_total + other_society['Orders'] <= 120 and dist_to_base <= 2.0):
                    current_cluster_societies.append(other_society)
                    current_order_total += other_society['Orders']
                    remaining_societies.remove(other_society)

            cluster_points = [{'id': s['Society ID'], 'lat': s['Latitude'], 'lon': s['Longitude']} for s in current_cluster_societies]
            path, distance = solve_tsp_nearest_neighbor(cluster_points, depot_coord)

            if distance <= 10.0: # Micro cluster distance constraint
                clusters.append({
                    'Cluster ID': f"C{cluster_id_counter}",
                    'Cluster Type': 'Micro',
                    'Societies': current_cluster_societies,
                    'Total Orders': current_order_total,
                    'Total Distance (km)': distance,
                    'Delivery Sequence': path,
                    'Hub ID': hub_id,
                    'Cost': costs['micro']
                })
                cluster_id_counter += 1
            else:
                # If it fails distance, societies become unclustered
                unclustered_societies.extend(current_cluster_societies)
    
    # Process any societies that were left over
    for society in unclustered_societies:
        clusters.append({
            'Cluster ID': f"U-{society['Society ID']}",
            'Cluster Type': 'Unclustered',
            'Societies': [society],
            'Total Orders': society['Orders'],
            'Total Distance (km)': 0,
            'Delivery Sequence': ['Depot', society['Society ID']],
            'Hub ID': society['Hub ID'],
            'Cost': 0 # Or some penalty cost
        })

    return clusters

def create_summary_and_map(clusters, depot_coord):
    """
    Generates a summary DataFrame and an interactive Folium map from cluster data.
    """
    summary_data = []
    if not clusters:
        # Create an empty map centered on the depot if no clusters
        m = folium.Map(location=depot_coord, zoom_start=12)
        folium.Marker(depot_coord, popup="Depot", icon=folium.Icon(color='black', icon='industry', prefix='fa')).add_to(m)
        return [], m._repr_html_()

    for cluster in clusters:
        total_orders = cluster['Total Orders']
        cpo = (cluster['Cost'] / total_orders) if total_orders > 0 else 0
        summary_data.append({
            'Cluster ID': cluster['Cluster ID'],
            'Cluster Type': cluster['Cluster Type'],
            'Total Orders': total_orders,
            'Societies': [s['Society Name'] for s in cluster['Societies']],
            'Society IDs': [s['Society ID'] for s in cluster['Societies']],
            'Society Count': len(cluster['Societies']),
            'Total Distance (km)': cluster['Total Distance (km)'],
            'CPO': cpo,
            'Delivery Sequence': [str(p) for p in cluster['Delivery Sequence']],
            'Hub ID': cluster['Hub ID']
        })

    # Create Map
    map_center = depot_coord
    m = folium.Map(location=map_center, zoom_start=12)
    
    # Add Depot Marker
    folium.Marker(depot_coord, popup="Depot", icon=folium.Icon(color='black', icon='industry', prefix='fa')).add_to(m)

    colors = {'Main': 'blue', 'Mini': 'green', 'Micro': 'orange', 'Unclustered': 'red'}

    for cluster in clusters:
        color = colors.get(cluster['Cluster Type'], 'gray')
        
        # Draw lines for the delivery sequence (TSP path)
        path_coords = [depot_coord]
        seq_map = {s['Society ID']: (s['Latitude'], s['Longitude']) for s in cluster['Societies']}
        
        # Start from depot, follow the sequence
        for society_id in cluster['Delivery Sequence'][1:]: # Skip 'Depot' at start
            if society_id in seq_map:
                path_coords.append(seq_map[society_id])
        
        if len(path_coords) > 1:
            folium.PolyLine(
                path_coords,
                color=color,
                weight=2.5,
                opacity=0.8,
                tooltip=f"Cluster {cluster['Cluster ID']} ({cluster['Cluster Type']})"
            ).add_to(m)
        
        # Add markers for each society
        for society in cluster['Societies']:
            folium.Marker(
                location=[society['Latitude'], society['Longitude']],
                popup=f"<b>{society['Society Name']}</b><br>ID: {society['Society ID']}<br>Orders: {society['Orders']}<br>Cluster: {cluster['Cluster ID']}",
                tooltip=f"{society['Society Name']} ({society['Orders']} Orders)",
                icon=folium.Icon(color=color, icon='info-sign')
            ).add_to(m)

    return summary_data, m._repr_html_()


# --- Flask Routes ---

@app.route('/')
def index():
    """Renders the main upload page."""
    return render_template('index.html')

@app.route('/download_template')
def download_template():
    """Provides a downloadable CSV template."""
    template_df = pd.DataFrame(columns=REQUIRED_COLUMNS)
    buffer = io.StringIO()
    template_df.to_csv(buffer, index=False)
    buffer.seek(0)
    return send_file(
        io.BytesIO(buffer.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='society_template.csv'
    )

@app.route('/process', methods=['POST'])
def process_data():
    """Handles file upload, processing, and renders results."""
    if 'file' not in request.files:
        flash('No file part in the request.', 'danger')
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        flash('No file selected.', 'danger')
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        try:
            # Get form data
            depot_lat = float(request.form.get('depot_lat'))
            depot_lon = float(request.form.get('depot_lon'))
            costs = {
                'main': float(request.form.get('main_cost')),
                'mini': float(request.form.get('mini_cost')),
                'micro': float(request.form.get('micro_cost'))
            }
            depot_coord = (depot_lat, depot_lon)

            # Read and validate data
            df = pd.read_csv(file) if file.filename.endswith('.csv') else pd.read_excel(file)
            validation_error = validate_columns(df)
            if validation_error:
                flash(validation_error, 'danger')
                return redirect(url_for('index'))

            # Perform clustering
            clusters = perform_clustering(df, depot_coord, costs)
            
            # Generate summary and map
            summary_data, map_html = create_summary_and_map(clusters, depot_coord)

            # Store results in session for downloading
            session['summary_data'] = summary_data
            session['full_cluster_data'] = clusters
            
            flash('Processing complete!', 'success')
            return render_template('results.html', summary_data=summary_data, map_html=map_html)

        except Exception as e:
            flash(f'An error occurred: {str(e)}', 'danger')
            return redirect(url_for('index'))
    else:
        flash('File type not allowed. Please use CSV or XLSX.', 'warning')
        return redirect(url_for('index'))

@app.route('/download_summary')
def download_summary():
    """Downloads the full summary table as a CSV."""
    summary_data = session.get('summary_data')
    if not summary_data:
        flash('No summary data available to download.', 'warning')
        return redirect(url_for('index'))

    df = pd.DataFrame(summary_data)
    # Convert list columns to strings for CSV
    df['Societies'] = df['Societies'].apply(lambda x: ', '.join(x))
    df['Delivery Sequence'] = df['Delivery Sequence'].apply(lambda x: ' -> '.join(x))
    
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    return send_file(
        io.BytesIO(buffer.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='cluster_summary.csv'
    )

@app.route('/download_cluster_details/<cluster_id>')
def download_cluster_details(cluster_id):
    """Downloads the detailed society data for a single cluster."""
    full_cluster_data = session.get('full_cluster_data')
    if not full_cluster_data:
        return "No cluster data found in session.", 404

    target_cluster = next((c for c in full_cluster_data if c['Cluster ID'] == cluster_id), None)
    
    if not target_cluster:
        return f"Cluster with ID {cluster_id} not found.", 404
        
    df = pd.DataFrame(target_cluster['Societies'])
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    return send_file(
        io.BytesIO(buffer.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'cluster_{cluster_id}_details.csv'
    )

if __name__ == '__main__':
    app.run(debug=True)
