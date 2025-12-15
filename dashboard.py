import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(page_title="Walmart Sales Forecast", layout="wide")
st.title("📊 Walmart Sales Forecasting Dashboard")
st.markdown("Select a store below to view historical sales and future predictions.")

# ==========================================
# 2. LOAD DATA & MODEL
# ==========================================
@st.cache_data
def load_data():
    # Load your CSV file
    df = pd.read_csv("Walmart.csv") 
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.sort_values(by=['Store', 'Date'])
    return df

@st.cache_resource
def load_model():
    with open('sales_forecast_model.pkl', 'rb') as f:
        package = pickle.load(f)
    return package['model'], package['features']

try:
    df = load_data()
    model, feature_names = load_model()
    st.success("Data and Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# ==========================================
# 3. SIDEBAR - USER INPUTS
# ==========================================
st.sidebar.header("Configuration")

# Store Selection
store_ids = df['Store'].unique()
selected_store = st.sidebar.selectbox("Select Store ID", store_ids)

# Filter data for selected store
store_data = df[df['Store'] == selected_store].copy()

# ==========================================
# 4. GENERATE FORECAST
# ==========================================
# Create inputs for prediction (using the last available data point as a base)
last_row = store_data.iloc[-1]

# Let's forecast for the next 4 weeks
future_dates = pd.date_range(start=last_row['Date'], periods=5, freq='W')[1:]
forecast_predictions = []

for date in future_dates:
    # Construct input for the model
    # Note: In a real scenario, you'd recalculate rolling means properly.
    # Here we use the static last known values for simplicity to demo the dashboard.
    input_row = pd.DataFrame([last_row])
    
    # Update date-based features
    input_row['month'] = date.month
    input_row['week'] = date.week
    
    # Ensure all model features exist
    for col in feature_names:
        if col not in input_row.columns:
            input_row[col] = 0
            
    # Reorder columns
    input_row = input_row[feature_names]
    
    # Predict
    pred = model.predict(input_row)[0]
    forecast_predictions.append(pred)

# Create Forecast DataFrame
forecast_df = pd.DataFrame({
    'Date': future_dates,
    'Weekly_Sales': forecast_predictions,
    'Type': 'Forecast'
})

# ==========================================
# 5. VISUALIZATION (Main Panel)
# ==========================================

# A. Key Metrics
current_sales = store_data.iloc[-1]['Weekly_Sales']
predicted_next_week = forecast_predictions[0]
delta = ((predicted_next_week - current_sales) / current_sales) * 100

col1, col2, col3 = st.columns(3)
col1.metric("Last Week Sales", f"${current_sales:,.2f}")
col2.metric("Next Week Forecast", f"${predicted_next_week:,.2f}", f"{delta:.2f}%")
col3.metric("Store ID", selected_store)

# B. Interactive Plot
fig = go.Figure()

# Plot History (Last 50 weeks for clarity)
history_plot = store_data.tail(50)
fig.add_trace(go.Scatter(
    x=history_plot['Date'], 
    y=history_plot['Weekly_Sales'], 
    mode='lines', 
    name='Historical Sales',
    line=dict(color='blue')
))

# Plot Forecast
fig.add_trace(go.Scatter(
    x=forecast_df['Date'], 
    y=forecast_df['Weekly_Sales'], 
    mode='lines+markers', 
    name='Forecast (Next 4 Weeks)',
    line=dict(color='red', dash='dash')
))

fig.update_layout(title=f"Sales Trend: Store {selected_store}", xaxis_title="Date", yaxis_title="Sales")
st.plotly_chart(fig, use_container_width=True)

# C. Data Table
st.subheader("Forecast Data")
st.dataframe(forecast_df)
# ... existing code ...
col1, col2, col3 = st.columns(3)
col1.metric("Last Week Sales", f"${current_sales:,.2f}")
# ... existing code ...

# ==========================================
# NEW: MODEL ACCURACY CHECK (VALIDATION)
# ==========================================
st.divider()
st.subheader("🔍 Model Accuracy Check")

if st.checkbox("Show Model Performance on Past Data"):
    # 1. Create a validation set (Last 20% of this store's history)
    split_idx = int(len(store_data) * 0.8)
    validation_data = store_data.iloc[split_idx:].copy()
    
    # 2. Make predictions on this past data
    val_preds = []
    # Note: We need to reconstruct features for these past dates
    # For a quick check, we assume features exist or we create simple ones
    for index, row in validation_data.iterrows():
        input_row = pd.DataFrame([row])
        
        # Ensure features match model expectation
        for col in feature_names:
            if col not in input_row.columns:
                input_row[col] = 0 # Fill missing engineered features with 0 for this quick test
        
        # Select correct columns
        input_row = input_row[feature_names]
        pred = model.predict(input_row)[0]
        val_preds.append(pred)

    # 3. Calculate Error
    from sklearn.metrics import mean_absolute_percentage_error
    store_mape = mean_absolute_percentage_error(validation_data['Weekly_Sales'], val_preds)

    # 4. Display
    st.write(f"**Model Accuracy for Store {selected_store}:**")
    st.info(f"MAPE: {store_mape:.2%}")
    
    # Plot Actual vs Predicted for this validation period
    val_fig = go.Figure()
    val_fig.add_trace(go.Scatter(x=validation_data['Date'], y=validation_data['Weekly_Sales'], name='Actual', line=dict(color='blue')))
    val_fig.add_trace(go.Scatter(x=validation_data['Date'], y=val_preds, name='Predicted', line=dict(color='red', dash='dot')))
    val_fig.update_layout(title="Actual vs Predicted (Validation Test)", xaxis_title="Date", yaxis_title="Sales")
    st.plotly_chart(val_fig, use_container_width=True)