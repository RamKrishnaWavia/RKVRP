import streamlit as st
import pandas as pd
from sklearn.cluster import DBSCAN
from geopy.distance import geodesic
import folium
from streamlit_folium import st_folium
from shapely.geometry import MultiPoint
import io

# Set page config
st.set_page_config(page_title="Delivery Cluster Optimizer", layout="wide")

st.title("📦 Delivery Cluster Optimizer with CPO Calculation")

# Upload Excel file
uploaded_file = st.file_uploader("Upload Excel File with Society Data", type=["xlsx"])

# Vehicle Cost input
vehicle_monthly_cost = st.number_input("Enter Monthly Vehicle Cost (₹)", value=35000)
working_days = st.number_input("Enter Working Days in a Month", value=30)
orders_per_cluster_min = st.number_input("Min Orders per Cluster", value=200)
orders_per_cluster_max = st.number_input("Max Orders per Cluster", value=450)
max_distance_m = st.number_input("Max Distance Between Societies in a Cluster (meters)", value=300)

def haversine_distance(latlon1, latlon2):
    return geodesic(latlon1, latlon2).meters

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # Ensure correct columns exist
    required_cols = {'Society ID', 'Society Name', 'Latitude', 'Longitude', 'Orders'}
    if not required_cols.issubset(set(df.columns)):
        st.error(f"Missing columns in Excel. Required: {required_cols}")
        st.stop()

    coords = df[['Latitude', 'Longitude']].values
    db = DBSCAN(eps=max_distance_m / 1000, min_samples=1, metric=lambda x, y: haversine_distance(x, y)/1000)
    df['Cluster'] = db.fit_predict(coords)

    # Cluster summary
    cluster_summary = df.groupby('Cluster').agg(
        Total_Orders=('Orders', 'sum'),
        Num_Societies=('Society ID', 'count'),
        Latitude=('Latitude', 'mean'),
        Longitude=('Longitude', 'mean')
    ).reset_index()

    # Filter valid clusters based on order volume
    valid_clusters = cluster_summary[
        (cluster_summary['Total_Orders'] >= orders_per_cluster_min) &
        (cluster_summary['Total_Orders'] <= orders_per_cluster_max)
    ].copy()

    valid_clusters['CPO (₹)'] = round(vehicle_monthly_cost / (working_days * valid_clusters['Total_Orders']), 2)

    st.success(f"✔ Found {len(valid_clusters)} valid clusters.")

    # Map visualization
    m = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=13)
    colors = ['green', 'blue', 'red', 'purple', 'orange', 'darkred', 'cadetblue']

    for _, row in valid_clusters.iterrows():
        cluster_df = df[df['Cluster'] == row['Cluster']]
        color = colors[row['Cluster'] % len(colors)]
        for _, point in cluster_df.iterrows():
            folium.CircleMarker(
                location=[point['Latitude'], point['Longitude']],
                radius=5,
                popup=f"{point['Society Name']} ({point['Orders']} orders)",
                color=color,
                fill=True
            ).add_to(m)

    st_folium(m, width=900, height=600)

    # Show summary table
    st.subheader("📊 Cluster Summary")
    st.dataframe(valid_clusters)

    # Download as CSV
    csv = valid_clusters.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Cluster Summary CSV", csv, file_name="cluster_summary.csv", mime='text/csv')
