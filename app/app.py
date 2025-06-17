import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Page title
st.title("📈 AAPL Stock Forecasting: ARIMA vs Prophet")

# Load preprocessed data
df = pd.read_csv('data/processed/AAPL_stock_data_cleaned.csv', parse_dates=['Date'])
df.set_index('Date', inplace=True)
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df.dropna(subset=['Close'], inplace=True)

# ARIMA Model
model_arima = ARIMA(df['Close'], order=(1, 1, 1))
arima_result = model_arima.fit()
df['ARIMA_Pred'] = arima_result.predict(start=1, end=len(df)-1, typ='levels')

# Prophet Model
prophet_df = df[['Close']].reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
prophet_model = Prophet(daily_seasonality=True)
prophet_model.fit(prophet_df)
future = prophet_model.make_future_dataframe(periods=0)
forecast = prophet_model.predict(future)
df['Prophet_Pred'] = forecast.loc[forecast['ds'] <= df.index[-1], 'yhat'].values

# Evaluation Function
def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return round(rmse, 2), round(mae, 2), round(mape, 2)

# Evaluate both models
df_eval = df.dropna(subset=['ARIMA_Pred', 'Prophet_Pred'])
arima_rmse, arima_mae, arima_mape = evaluate(df_eval['Close'], df_eval['ARIMA_Pred'])
prophet_rmse, prophet_mae, prophet_mape = evaluate(df_eval['Close'], df_eval['Prophet_Pred'])

# Plot Forecasts
st.subheader("📊 Forecast vs Actual")
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df_eval['Close'], label='Actual', color='black')
ax.plot(df_eval['ARIMA_Pred'], label='ARIMA Forecast', linestyle='--')
ax.plot(df_eval['Prophet_Pred'], label='Prophet Forecast', linestyle=':')
ax.set_title('Forecast Comparison')
ax.legend()
st.pyplot(fig)

# Show Metrics
st.subheader("📋 Evaluation Metrics")
metrics_df = pd.DataFrame({
    'Model': ['ARIMA', 'Prophet'],
    'RMSE': [arima_rmse, prophet_rmse],
    'MAE': [arima_mae, prophet_mae],
    'MAPE (%)': [arima_mape, prophet_mape]
})
st.dataframe(metrics_df)
