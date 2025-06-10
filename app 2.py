import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from math import radians, cos, sin, sqrt, atan2

st.set_page_config(layout="wide")
st.title("Milk Delivery Route Optimizer")

# Sidebar options
min_orders = st.sidebar.number_input("Min Orders per CEE", min_value=100, max_value=300, value=150)
max_orders = st.sidebar.number_input("Max Orders per CEE", min_value=150, max_value=300, value=200)
avg_speed = st.sidebar.number_input("Vehicle Speed (km/h)", min_value=10, max_value=40, value=20)

st.sidebar.markdown("---")
lat_edit = st.sidebar.checkbox("Enable Lat/Long Manual Edit")

# Input File Upload
uploaded_file = st.file_uploader("Upload input Excel file", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)

    required_cols = {"Society_ID", "Society_Name", "Latitude", "Longitude", "Orders", "Current_CEE_Allocated"}
    if not required_cols.issubset(df.columns):
        st.error(f"Input file missing required columns: {required_cols - set(df.columns)}")
        st.stop()

    if lat_edit:
        st.subheader("Edit Latitude & Longitude (Optional)")
        df = st.data_editor(df, num_rows="dynamic")

    df['Latitude'] = df['Latitude'].astype(float)
    df['Longitude'] = df['Longitude'].astype(float)

    # Distance Calculation Function
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371.0088
        phi1, phi2 = radians(lat1), radians(lat2)
        dphi = radians(lat2 - lat1)
        dlambda = radians(lon2 - lon1)
        a = sin(dphi / 2)**2 + cos(phi1) * cos(phi2) * sin(dlambda / 2)**2
        return 2 * R * atan2(sqrt(a), sqrt(1 - a))

    # Auto Clustering Based on Orders
    def auto_group_minimize_cees(df):
        clusters = []
        leftovers = []
        used = set()

        sorted_df = df.sort_values(by="Orders", ascending=False).reset_index(drop=True)
        for idx, row in sorted_df.iterrows():
            if row["Society_ID"] in used:
                continue
            current_orders = row["Orders"]
            cluster = [row]
            used.add(row["Society_ID"])
            for j in range(idx + 1, len(sorted_df)):
                next_row = sorted_df.loc[j]
                if next_row["Society_ID"] in used:
                    continue
                if current_orders + next_row["Orders"] <= max_orders:
                    cluster.append(next_row)
                    current_orders += next_row["Orders"]
                    used.add(next_row["Society_ID"])
                if current_orders >= min_orders:
                    break
            if current_orders >= min_orders:
                clusters.append(pd.DataFrame(cluster))
            else:
                leftovers.append(pd.DataFrame(cluster))

        all_clusters = clusters + leftovers
        cee_routes = []
        cee_id = 1

        for cluster_id, cluster_df in enumerate(all_clusters, start=1):
            G = nx.Graph()
            for i, r1 in cluster_df.iterrows():
                for j, r2 in cluster_df.iterrows():
                    if i != j:
                        dist = haversine(r1["Latitude"], r1["Longitude"], r2["Latitude"], r2["Longitude"])
                        G.add_edge(r1["Society_ID"], r2["Society_ID"], weight=dist)

            tsp_path = nx.approximation.traveling_salesman_problem(G, cycle=False, method='greedy')

            cluster_df["CEE_ID"] = cee_id
            cluster_df["Cluster_ID"] = cluster_id
            cluster_df["TSP_Sequence"] = tsp_path
            cee_routes.append(cluster_df)
            cee_id += 1

        result_df = pd.concat(cee_routes, ignore_index=True)
        return result_df

    st.success("File loaded and processed. Running routing logic...")
    routed_df = auto_group_minimize_cees(df)

    # Group Summary
    grouped = routed_df.groupby("CEE_ID").agg({
        "Society_ID": "count",
        "Orders": "sum"
    }).rename(columns={"Society_ID": "Total_Societies", "Orders": "Total_Orders"})

    grouped["Estimated_Time_Min"] = (grouped["Total_Orders"] / 10) * 5  # estimate 5 mins per 10 orders
    grouped["Estimated_Distance_km"] = grouped["Estimated_Time_Min"] / 60 * avg_speed

    final_output = routed_df.merge(df[["Society_ID", "Current_CEE_Allocated"]], on="Society_ID", how="left")

    # Display Outputs
    st.subheader("Routed Data")
    st.dataframe(final_output[[
        "Society_ID", "Society_Name", "Orders", "Latitude", "Longitude",
        "Cluster_ID", "CEE_ID", "Current_CEE_Allocated"
    ]])

    st.subheader("CEE Summary")
    st.dataframe(grouped)

    st.download_button("Download Full Output", final_output.to_csv(index=False), "final_output.csv", "text/csv")
    st.download_button("Download CEE Summary", grouped.reset_index().to_csv(index=False), "cee_summary.csv", "text/csv")

    st.map(final_output[["Latitude", "Longitude"]])
