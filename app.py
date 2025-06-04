import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle


def cluster_societies(df, vehicle_cost=35000, eps_km=1.0, min_orders=200, max_orders=450):
    coords = df[['Latitude', 'Longitude']].to_numpy()
    coords_rad = np.radians(coords)

    kms_per_radian = 6371.0088
    db = DBSCAN(eps=eps_km / kms_per_radian, min_samples=1, metric='haversine').fit(coords_rad)
    df['Cluster'] = db.labels_

    clusters = []
    for cluster_id in np.unique(df['Cluster']):
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


# --- Streamlit UI ---
st.set_page_config(page_title="Society Clustering", layout="wide")
st.title("\U0001F69A Milk Delivery Cluster Optimizer")

uploaded_file = st.file_uploader("Upload Society CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    vehicle_cost = st.number_input("Vehicle Monthly Cost (₹)", value=35000, step=500)
    eps_km = st.slider("Max Distance between Societies (km)", 0.1, 5.0, value=1.0, step=0.1)
    min_orders = 200
    max_orders = 450

    if st.button("Generate Green Clusters"):
        with st.spinner("Clustering based on orders & distance..."):
            clusters = cluster_societies(df, vehicle_cost, eps_km, min_orders, max_orders)

        if not clusters:
            st.warning("No clusters found within the 200–450 order range.")
        else:
            st.success(f"✅ {len(clusters)} clusters formed!")
            for c in clusters:
                st.markdown(f"### 🟢 Green Cluster {c['Cluster ID']}")
                st.write(f"**Total Orders:** {c['Total Orders']}")
                st.write(f"**Cost per Order (₹):** ₹{c['Cost per Order (₹)']}")
                st.write(f"**Centroid:** ({c['Centroid Latitude']:.5f}, {c['Centroid Longitude']:.5f})")
                st.write(f"**Max Distance from Centroid:** {c['Max Distance (km)']:.2f} km")
                st.write("**Societies:**")
                st.dataframe(pd.DataFrame(c['Societies']))

            export_df = pd.DataFrame([
                {
                    'Cluster ID': c['Cluster ID'],
                    'Total Orders': c['Total Orders'],
                    'Cost per Order (₹)': c['Cost per Order (₹)'],
                    'Centroid Lat': c['Centroid Latitude'],
                    'Centroid Lon': c['Centroid Longitude'],
                    'Max Distance (km)': c['Max Distance (km)'],
                    'Societies': "; ".join(s['Society Name'] for s in c['Societies'])
                }
                for c in clusters
            ])
            csv = export_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Cluster Summary", csv, "green_clusters.csv", "text/csv")
