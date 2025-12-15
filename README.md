Retail Sales Forecasting (End-to-End ML Project)

Project Overview
This project focuses on predicting weekly sales for a retail giant (Walmart) using historical time-series data. The goal is to assist store managers in inventory planning by accurately forecasting future demand.

The solution demonstrates a complete Machine Learning pipeline: Data Engineering -> Model Training (XGBoost) -> Deployment (Flask API) -> User Dashboard (Streamlit).

Tech Stack
- Language: Python 3.x
- Data Analysis: Pandas, NumPy, Matplotlib, Seaborn
- Machine Learning: XGBoost, Scikit-Learn
- Time Series: Lag Features, Rolling Windows
- Deployment: Flask (REST API)
- Dashboard: Streamlit, Plotly

Project Structure
Week-1-Retail-Forecasting
   app.py                   (Flask API for serving predictions)
   dashboard.py             (Streamlit User Interface)
   sales_forecast_model.pkl (Trained XGBoost Model)
   Walmart.csv              (Dataset - Historical Sales)
   notebook.ipynb           (Jupyter Notebook - Training & Analysis)
   README.md                (Project Documentation)

Key Features
1. Multi-Store Forecasting: The model handles 45 different stores using a single Global XGBoost model.
2. Feature Engineering: Includes Lag features (Previous Week Sales), Rolling Means (4-week trends), and Seasonality extraction (Week, Month).
3. Interactive Dashboard: A Streamlit app allowing users to select a store and view live forecasts.
4. API Integration: A Flask endpoint that accepts JSON input and returns sales predictions.

Model Performance
- Model Used: XGBoost Regressor
- Evaluation Metric: MAPE (Mean Absolute Percentage Error)
- Performance: Achieved ~6% MAPE on the test set, indicating high accuracy for retail planning.

Future Improvements
- Incorporate external factors like Weather and CPI into the model.
- Test Deep Learning approaches (LSTM) for potentially better long-term trends.
- Dockerize the application for easier cloud deployment.
