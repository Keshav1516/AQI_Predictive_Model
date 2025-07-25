# 🌫️ AQI Predictive Model – Real-Time Air Quality Forecasting

> Predict Air Quality Index (AQI) in real-time using machine learning models trained on live and historical pollutant data.

---

## 📌 Project Overview

This repository showcases an AQI forecasting pipeline that ingests **real-time data** from air-monitoring stations, preprocesses pollutants metrics 
(e.g. PM₂.₅, PM₁₀, NO₂, CO, SO₂), and implements multiple regression models—such as Random Forest, K‑Neighbors Regression, Gradient Boosting, and XGBoost—to deliver robust 
AQI predictions using Python and scikit-learn.

---

## 🧩 Features & Highlights

- 🚀 **Real-time AQI prediction** from live station data (JSON).
- 📊 **Multiple regression models** benchmarked for accuracy:
  - Random Forest Regressor
  - K‑Neighbors Regressor
  - Gradient Boosting Regressor
  - Extreme Gradient Boosting (XGBoost)
- 🔄 **Data preprocessing pipeline** with cleaning, normalization, and outlier handling.
- 📈 **Performance comparison** across model types for optimal selection.
- 🔄 **Extensible design** ready for Streamlit or Flask-based deployment.

---

## 📂 Repository Structure

├── AllStationRealtimeData.json # Sample real-time monitoring data
├── All_Station_Reatime_Data_PJ.py # Script to ingest and preprocess data
├── .gitignore # Git ignore configuration
├── output/ # (Optional) Predicted AQI outputs or logs
├── requirements.txt # Python dependencies list
└── README.md # Project overview and instructions
