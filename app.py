import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle

def cluster_societies(df, vehicle_cost=35000, eps_km=1.0, min_orders=150, max_orders=200):
    coords = df[['Latitude', 'Longitude']].to_numpy()
    coords_rad = np.radians(coords)

    kms_per_radian = 6371.0088
    db = DBSCAN(eps=eps_km / kms_per_radian, min_samples=1, metric='haversine').fit(coords_rad)
    df['Cluster'] = db.labels_

    clusters = []
    for cluster_id in np.unique(db.labels_):
        cluster_df = df[df['Cluster'] == cluster_id].copy()
        total_orders = cluster_df['Orders'].sum()

        if min_orders <= total_orders <= max_orders:
            centroid_lat = cluster_df['Latitude'].mean()
            centroid_lon = cluster_df['Longitude'].mean()
            centroid = (centroid_lat, centroid_lon)

            distances = cluster_df.apply(
                lambda row: great_circle((row['Latitude'], row['Longitude']), centroid).km,
                axis=1
            )
            max_distance_km = distances.max()

            cpo = vehicle_cost / total_orders

            clusters.append({
                'Cluster ID': cluster_id,
                'Total Orders': total_orders,
                'Centroid Latitude': centroid_lat,
                'Centroid Longitude': centroid_lon,
                'Max Distance (km)': max_distance_km,
                'Cost per Order (₹)': round(cpo, 2),
                'Societies': cluster_df[['Society ID', 'Society Name', 'Orders']].to_dict(orient='records')
            })

    return clusters

st.title("Society Clustering for Milk Delivery")

uploaded_file = st.file_uploader("Upload societies CSV file", type=['csv'])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    vehicle_cost = st.number_input("Vehicle Monthly Cost (₹)", value=35000, step=1000)
    eps_km = st.slider("Max distance between societies (km)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    min_orders = st.number_input("Min orders per cluster (Green Cluster)", value=150, step=1)
    max_orders = st.number_input("Max orders per cluster (Green Cluster)", value=200, step=1)

    if st.button("Generate Clusters"):
        with st.spinner("Clustering societies..."):
            clusters = cluster_societies(df, vehicle_cost, eps_km, min_orders, max_orders)

        if not clusters:
            st.warning("No clusters found with given parameters.")
        else:
            st.success(f"Found {len(clusters)} clusters.")
            for c in clusters:
                st.markdown(f"### Cluster {c['Cluster ID']}")
                st.write(f"**Total Orders:** {c['Total Orders']}")
                st.write(f"**Cost per Order (₹):** ₹{c['Cost per Order (₹)']}")
                st.write(f"**Centroid Latitude, Longitude:** ({c['Centroid Latitude']:.5f}, {c['Centroid Longitude']:.5f})")
                st.write(f"**Max Distance from Centroid (km):** {c['Max Distance (km)']:.2f}")
                st.write("**Societies in Cluster:**")
                st.table(pd.DataFrame(c['Societies']))
