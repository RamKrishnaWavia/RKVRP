# Milk Delivery Route Optimizer

This Streamlit app helps optimize last-mile milk delivery routes using KMeans clustering and Google OR-Tools.

## Features

- Accepts CSV file with delivery orders and coordinates
- Automatically clusters orders based on vehicle capacity
- Uses OR-Tools for optimal routing within each cluster
- Calculates cost per order
- Visualizes routes on a map
- Allows downloading the optimized summary

## Deployment

Deploy on [Streamlit Cloud](https://streamlit.io/cloud) with:

- `app.py` as main script
- `requirements.txt` for dependencies

## Input CSV Format

| Society ID | Society Name | City     | Drop Point   | Latitude | Longitude | Orders |
|------------|--------------|----------|--------------|----------|-----------|--------|
| S1         | Heaven Apt   | Bangalore| Soukya Road  | 12.9456  | 77.7501   | 120    |

