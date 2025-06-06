import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from io import BytesIO
import base64

st.set_page_config(layout="wide")
st.title("Milk Delivery Route Optimizer with CEE Clustering")

# Sidebar controls
st.sidebar.header("Settings")
min_orders = st.sidebar.number_input("Minimum orders per vehicle (CEE)", value=150, step=10)
max_orders = st.sidebar.number_input("Maximum orders per vehicle (CEE)", value=200, step=10)
avg_speed_kmph = st.sidebar.number_input("Average speed (km/h)", value=20)
monthly_cost_per_vehicle = st.sidebar.number_input("Vehicle monthly cost ₹", value=35000)
editable_lat_long = st.sidebar.checkbox("Make Latitude & Longitude editable", value=True)

# File upload
uploaded_file = st.file_uploader("Upload input file (.csv)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Show raw data
    st.subheader("Raw Input Data")
    if editable_lat_long:
        df_edit = st.data_editor(df, num_rows="dynamic")
    else:
        df_edit = df.copy()

    df = df_edit.copy()

    # Drop duplicates and clean
    df.dropna(subset=["Society_ID", "Latitude", "Longitude", "Orders"], inplace=True)
    df["Latitude"] = df["Latitude"].astype(float)
    df["Longitude"] = df["Longitude"].astype(float)
    df["Orders"] = df["Orders"].astype(int)

    # Clustering
    st.subheader("Cluster Formation")

    coords = df[["Latitude", "Longitude"]]
    k = max(1, int(df["Orders"].sum() // max_orders))  # Initial cluster estimate

    kmeans = KMeans(n_clusters=k, random_state=0)
    df["Cluster"] = kmeans.fit_predict(coords)

    st.map(df[["Latitude", "Longitude"]].assign(Cluster=df["Cluster"]))

    # CEE assignment logic
    def assign_cees_to_clusters(df):
        summary = []
        cee_routes = []

        for cluster_id, group in df.groupby("Cluster"):
            group = group.sort_values("Society_ID").reset_index(drop=True)
            total_orders = group["Orders"].sum()
            cees_needed = max(1, int(np.ceil(total_orders / max_orders)))

            route_split = np.array_split(group, cees_needed)
            for i, subroute in enumerate(route_split):
                subroute["CEE_ID"] = f"C{cluster_id}_V{i+1}"
                subroute["Cluster_ID"] = cluster_id
                cee_routes.append(subroute)

            summary.append({
                "Cluster_ID": cluster_id,
                "Total_Orders": total_orders,
                "Allocated_CEEs": group["Allocated_CEEs"].sum() if "Allocated_CEEs" in group else "N/A",
                "Required_CEEs": cees_needed
            })

        final_df = pd.concat(cee_routes, ignore_index=True)
        return final_df, pd.DataFrame(summary)

    routed_df, cluster_summary = assign_cees_to_clusters(df)

    st.subheader("CEE Allocation Summary")
    st.dataframe(cluster_summary)

    st.subheader("CEE Route Details")
    st.dataframe(routed_df[["Society_ID", "CEE_ID", "Cluster_ID", "Orders", "Latitude", "Longitude"]])

    # Download output as CSV
    def get_csv_download_link(df, filename):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        return f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Download {filename}</a>'

    st.markdown(get_csv_download_link(routed_df, "cee_routes_output.csv"), unsafe_allow_html=True)
    st.markdown(get_csv_download_link(cluster_summary, "cee_cluster_summary.csv"), unsafe_allow_html=True)

else:
    st.info("📄 Please upload a CSV file with society delivery data.")
