import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import folium
from streamlit_folium import folium_static
from math import radians, cos, sin, asin, sqrt

st.set_page_config(layout="wide")
st.title("Milk Delivery Route Optimizer")

st.sidebar.header("Vehicle and Cost Settings")
vehicle_capacity = st.sidebar.number_input("Max Orders per Vehicle", min_value=1, value=200)
vehicle_cost = st.sidebar.number_input("Cost per Vehicle per Month (₹)", min_value=1000, value=35000)
target_cpo = 4.0

def haversine(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371
    return c * r

uploaded_file = st.file_uploader("Upload Society Data CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    required_columns = {"Society ID", "Apartment", "Latitude", "Longitude", "Orders"}
    if not required_columns.issubset(df.columns):
        st.error(f"Uploaded CSV must contain columns: {', '.join(required_columns)}")
    else:
        source = df[df['Society ID'].str.lower() == 'source']
        df = df[df['Society ID'].str.lower() != 'source'].reset_index(drop=True)

        coords = df[["Latitude", "Longitude"]].values
        total_orders = df["Orders"].sum()
        num_clusters = max(1, int(total_orders // vehicle_capacity) + (1 if total_orders % vehicle_capacity > 0 else 0))
        num_clusters = min(num_clusters, len(df))

        kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(coords)
        df['Cluster'] = kmeans.labels_

        m = folium.Map(location=[df["Latitude"].mean(), df["Longitude"].mean()], zoom_start=12)

        route_summaries = []

        for cluster_id in range(num_clusters):
            cluster_df = df[df['Cluster'] == cluster_id].copy()
            cluster_df = cluster_df.sort_values(by=["Latitude", "Longitude"]).reset_index(drop=True)
            route_orders = cluster_df["Orders"].sum()

            route_coords = [(source.iloc[0]["Latitude"], source.iloc[0]["Longitude"])]
            route_coords += list(zip(cluster_df["Latitude"], cluster_df["Longitude"]))
            route_coords += [(source.iloc[0]["Latitude"], source.iloc[0]["Longitude"])]

            total_distance = 0.0
            for i in range(len(route_coords) - 1):
                total_distance += haversine(*route_coords[i], *route_coords[i+1])

            cost_per_order = (vehicle_cost / route_orders) if route_orders > 0 else float('inf')
            route_summaries.append({
                "Route": cluster_id + 1,
                "Total Orders": route_orders,
                "Distance (km)": round(total_distance, 2),
                "Cost/Order (₹)": round(cost_per_order, 2)
            })

            for i, row in cluster_df.iterrows():
                folium.Marker(
                    location=[row["Latitude"], row["Longitude"]],
                    popup=f"{i+1}. {row['Apartment']} (ID: {row['Society ID']}, {row['Orders']} orders)",
                    icon=folium.DivIcon(html=f"<div style='font-size: 12pt'>{i+1}</div>")
                ).add_to(m)

            folium.PolyLine(route_coords, color='blue', weight=3, opacity=0.7).add_to(m)

        st.subheader("Optimized Routes Map")
        folium_static(m, width=1200, height=700)

        st.subheader("Route Summary")
        summary_df = pd.DataFrame(route_summaries)
        st.dataframe(summary_df)

        st.download_button("Download Route Summary CSV", summary_df.to_csv(index=False), file_name="route_summary.csv")
