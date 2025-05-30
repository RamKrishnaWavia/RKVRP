
# 🥛 Milk Delivery Route Optimizer (Streamlit App)

This Streamlit app helps optimize last-mile milk delivery routes to apartments/societies between 4:00 AM to 6:30 AM, minimizing the delivery cost per order. It uses Google OR-Tools for route optimization and Folium for route visualization on a map.

---

## 🚀 Features
- Upload CSV file with delivery data
- Specify number of vehicles and their capacity
- Optimize routes based on time windows and demand
- Visualize delivery routes on an interactive map
- Download optimized route plan as CSV

---

## 📂 Input CSV Format

Make sure your file has the following columns:

```csv
apartment_name,latitude,longitude,order_qty,start_time,end_time
Depot,12.935,77.614,0,4:00,6:30
Apt1,12.937,77.620,20,4:00,6:00
Apt2,12.940,77.622,15,4:30,6:30
Apt3,12.945,77.618,10,4:00,5:30
Apt4,12.943,77.613,5,4:20,6:10
```

- **Depot** should be the first row and will be treated as the starting point.
- Time format should be `HH:MM` in 24-hour format.

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the App

```bash
streamlit run app.py
```

---

## ☁️ Deploy on Streamlit Cloud

1. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
2. Sign in and click “New app”
3. Upload these files and deploy.

---

## 🧾 License
MIT License
