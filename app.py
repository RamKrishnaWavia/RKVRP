import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
from geopy.distance import geodesic
import numpy as np
import io

# Title and Page Config
st.set_page_config(page_title="RK - Delivery Route Optimizer", layout="wide")
st.title("RK - Delivery Route Optimizer")
st.markdown("Optimize early morning deliveries (4 AM to 7 AM) from Soukya Road to societies, ensuring each route is under ₹4 cost per order.")

# Input Vehicle Monthly Cost
vehicle_cost = st.number_input("Enter Monthly Cost per Vehicle (₹):", min_value=1000, max_value=100000, value=35000, step=500)

# CSV Template Download
if st.button("📅 Download CSV Template"):
    template = pd.DataFrame({
        "Society ID": [101, 102],
        "Apartment": ["ABC Residency", "Green Heights"],
        "Latitude": [12.935, 12.938],
        "Longitude": [77.614, 77.610],
        "Orders": [120, 90]
    })
    csv = template.to_csv(index=False).encode('utf-8')
    st.download_button("Download Template", data=csv, file_name="milk_delivery_template.csv", mime='text/csv')

# Upload CSV
uploaded_file = st.file_uploader("Upload Delivery Data CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if not {"Latitude", "Longitude", "Orders", "Society ID", "Apartment"}.issubset(df.columns):
        st.error("CSV must contain 'Society ID', 'Apartment', 'Latitude', 'Longitude', and 'Orders' columns.")
        st.stop()

    # Add source (Soukya Road)
    source_latlon = (12.8910, 77.7764)  # Soukya Road
    source = pd.DataFrame({
        "Society ID": [0],
        "Apartment": ["Soukya Road"],
        "Latitude": [source_latlon[0]],
        "Longitude": [source_latlon[1]],
        "Orders": [0]
    })
    df = pd.concat([source, df], ignore_index=True)

    # KMeans clustering to divide into routes (200 orders per route)
    total_orders = df["Orders"].sum()
    num_routes = int(np.ceil(total_orders / 200))
    kmeans = KMeans(n_clusters=num_routes, random_state=0)
    df = df.copy()
    df.loc[df["Society ID"] != 0, "Cluster"] = kmeans.fit_predict(df.loc[df["Society ID"] != 0, ["Latitude", "Longitude"]])
    df["Cluster"] = df["Cluster"].fillna(-1).astype(int)

    st.success(f"🔁 Divided into {num_routes} optimized routes")

    # Process each route
    route_summary = []
    for route_id in sorted(df["Cluster"].unique()):
        cluster_df = df[(df["Cluster"] == route_id) | (df["Society ID"] == 0)].reset_index(drop=True)
        coords = list(zip(cluster_df.Latitude, cluster_df.Longitude))

        # Simple nearest neighbor sequence
        visited = [0]  # start from Soukya Road
        while len(visited) < len(coords):
            last = visited[-1]
            remaining = list(set(range(len(coords))) - set(visited))
            next_stop = min(remaining, key=lambda x: geodesic(coords[last], coords[x]).km)
            visited.append(next_stop)

        ordered_cluster = cluster_df.iloc[visited].reset_index(drop=True)
        ordered_cluster["Stop"] = range(1, len(ordered_cluster)+1)

        # Calculate distance
        total_km = 0
        for i in range(len(ordered_cluster)-1):
            loc1 = (ordered_cluster.loc[i, "Latitude"], ordered_cluster.loc[i, "Longitude"])
            loc2 = (ordered_cluster.loc[i+1, "Latitude"], ordered_cluster.loc[i+1, "Longitude"])
            total_km += geodesic(loc1, loc2).km

        total_orders_cluster = ordered_cluster["Orders"].sum()
        cost_per_order = (vehicle_cost / 30) / total_orders_cluster  # per day cost

        # Map
        m = folium.Map(location=coords[0], zoom_start=13)
        for i, row in ordered_cluster.iterrows():
            folium.Marker(
                location=[row["Latitude"], row["Longitude"]],
                popup=f"Stop {row['Stop']}: {row['Apartment']} (ID: {row['Society ID']}, {row['Orders']} orders)",
                icon=folium.DivIcon(html=f"<div style='font-size: 12pt'>{row['Stop']}</div>")
            ).add_to(m)
            if i > 0:
                folium.PolyLine(
                    locations=[
                        (ordered_cluster.loc[i-1, "Latitude"], ordered_cluster.loc[i-1, "Longitude"]),
                        (row["Latitude"], row["Longitude"])
                    ],
                    color="blue",
                    tooltip=f"{geodesic((ordered_cluster.loc[i-1, 'Latitude'], ordered_cluster.loc[i-1, 'Longitude']), (row['Latitude'], row['Longitude'])).km:.2f} km"
                ).add_to(m)
        st.subheader(f"🗺 Route #{route_id+1}")
        st_folium(m, height=450, width=800)

        # Route Summary
        st.markdown(f"**Total Distance:** {total_km:.2f} km")
        st.markdown(f"**Total Orders:** {int(total_orders_cluster)}")
        st.markdown(f"**Cost per Order:** ₹{cost_per_order:.2f} (Target < ₹4)")
        route_summary.append({
            "Route ID": route_id + 1,
            "Total Orders": int(total_orders_cluster),
            "Total Distance (km)": round(total_km, 2),
            "Cost per Order (₹)": round(cost_per_order, 2)
        })

        # CSV download for this route
        csv_data = ordered_cluster[["Stop", "Society ID", "Apartment", "Latitude", "Longitude", "Orders"]]
        csv_file = csv_data.to_csv(index=False).encode('utf-8')
        st.download_button(f"📄 Download Route #{route_id+1} CSV", data=csv_file, file_name=f"route_{route_id+1}.csv", mime='text/csv')

    # Overall Summary
    if route_summary:
        st.subheader("📊 Route Summary")
        st.dataframe(pd.DataFrame(route_summary))
