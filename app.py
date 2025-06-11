import streamlit as st
import pandas as pd
import numpy as np
from geopy.distance import great_circle
import folium
from streamlit_folium import st_folium
from io import StringIO

# Helper to calculate route distance
def calculate_route_distance(route):
    distance = 0.0
    for i in range(len(route) - 1):
        distance += great_circle(route[i], route[i+1]).km
    return distance

# Helper to compute distance in meters
def is_within_distance(coord1, coord2, max_dist=2.0):
    return great_circle(coord1, coord2).km <= max_dist

st.title("Milk & Grocery Delivery Clustering Tool")

# Template file download
st.subheader("Download Template")
template = pd.DataFrame({"Society ID": [], "Society": [], "Latitude": [], "Longitude": [], "Orders": []})
st.download_button("Download Template CSV", data=template.to_csv(index=False), file_name="society_template.csv")

# File upload
st.subheader("Upload Society Data")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data", df)

    df = df.sort_values(by='Orders', ascending=False).reset_index(drop=True)
    df['Cluster'] = -1
    cluster_id = 0

    unassigned = df.copy()

    while not unassigned.empty:
        seed = unassigned.iloc[0]
        seed_coord = (seed['Latitude'], seed['Longitude'])
        seed_orders = seed['Orders']

        cluster_members = [seed.name]
        cluster_orders = seed_orders

        for idx, row in unassigned.iloc[1:].iterrows():
            if cluster_orders >= 230:
                break
            coord = (row['Latitude'], row['Longitude'])
            if all(is_within_distance(coord, (df.loc[i]['Latitude'], df.loc[i]['Longitude'])) for i in cluster_members):
                if cluster_orders + row['Orders'] <= 230:
                    cluster_members.append(row.name)
                    cluster_orders += row['Orders']

        df.loc[cluster_members, 'Cluster'] = cluster_id
        cluster_id += 1
        unassigned = df[df['Cluster'] == -1]

    cluster_summary = []
    cluster_map = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=12)

    for label in sorted(df['Cluster'].unique()):
        cluster_df = df[df['Cluster'] == label]
        total_orders = cluster_df['Orders'].sum()
        valid_cluster = 190 <= total_orders <= 230

        coords = list(zip(cluster_df['Latitude'], cluster_df['Longitude']))
        for _, row in cluster_df.iterrows():
            folium.Marker(
                location=[row['Latitude'], row['Longitude']],
                popup=f"{row['Society']} ({row['Orders']} orders)",
                icon=folium.Icon(color='blue', icon='info-sign')
            ).add_to(cluster_map)

        route = list(zip(cluster_df['Latitude'], cluster_df['Longitude']))
        distance_km = calculate_route_distance(route)

        cluster_summary.append({
            "Cluster": label,
            "Society IDs": ", ".join(cluster_df['Society ID'].astype(str).tolist()),
            "Societies": ", ".join(cluster_df['Society'].tolist()),
            "No. of Societies": len(cluster_df),
            "Total Orders": total_orders,
            "Total Distance (km)": round(distance_km, 2),
            "Valid Cluster (190–230 Orders)": "Yes" if valid_cluster else "No"
        })

    # Show map
    st.subheader("Cluster Map")
    st_data = st_folium(cluster_map, width=725)

    # Show cluster summary
    summary_df = pd.DataFrame(cluster_summary)
    st.subheader("Cluster Summary")
    st.dataframe(summary_df)

    # Download summary
    csv = summary_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Cluster Summary CSV", data=csv, file_name="cluster_summary.csv")
