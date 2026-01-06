import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import timedelta
import warnings

# Suppress version and projection warnings
warnings.filterwarnings("ignore")

# 1. Data Loading and Preparation
df = pd.read_csv('resultados_temporais.csv')
df['data'] = pd.to_datetime(df['data'])
df = df.sort_values('data')

def prepare_features(data):
    """Creates time-based features for the ML model."""
    df_feat = data.copy()
    df_feat['mes'] = df_feat['data'].dt.month
    df_feat['dia_ano'] = df_feat['data'].dt.dayofyear
    df_feat['ano'] = df_feat['data'].dt.year
    # Lag 1: Previous observation value (system memory)
    df_feat['nbr_lag1'] = df_feat['media_NBR'].shift(1)
    df_feat['ndvi_lag1'] = df_feat['media_NDVI'].shift(1)
    return df_feat.dropna()

df_model = prepare_features(df)

# 2. Variable Definition (Features and Target)
features = ['dias_desde_inicio', 'mes', 'dia_ano', 'nbr_lag1', 'ndvi_lag1']
X = df_model[features]
y_nbr = df_model['media_NBR']

# Temporal split (80% training, 20% testing)
split = int(len(df_model) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train_nbr, y_test_nbr = y_nbr.iloc[:split], y_nbr.iloc[split:]

# 3. Model Training (Random Forest)
model_nbr = RandomForestRegressor(n_estimators=200, random_state=42)
model_nbr.fit(X_train, y_train_nbr)

# 4. Evaluation
pred_nbr = model_nbr.predict(X_test)
print(f"--- NBR Model Performance ---")
print(f"R² Score: {r2_score(y_test_nbr, pred_nbr):.4f}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test_nbr, pred_nbr):.4f}")

# 5. Future Forecast (Next 12 months)
def forecast_future(model, last_row, days_to_forecast=365):
    future_dates = []
    future_preds = []
    
    current_nbr = last_row['media_NBR']
    current_ndvi = last_row['media_NDVI']
    current_date = last_row['data']
    start_days = last_row['dias_desde_inicio']

    for i in range(1, 13):
        next_date = current_date + timedelta(days=30 * i)
        next_days = start_days + (30 * i)
        
        X_fut = pd.DataFrame([[
            next_days,
            next_date.month,
            next_date.timetuple().tm_yday,
            current_nbr,
            current_ndvi
        ]], columns=features)
        
        pred = model.predict(X_fut)[0]
        future_dates.append(next_date)
        future_preds.append(pred)
        current_nbr = pred  # autoregressive update

    return future_dates, future_preds

f_dates, f_nbr = forecast_future(model_nbr, df.iloc[-1])

# 6. Results Visualization (WITH FIX FOR MATPLOTLIB/PANDAS)
plt.figure(figsize=(15, 7))

# Convert to .values to avoid ValueError in some environments
plt.plot(
    df['data'].values,
    df['media_NBR'].values,
    label='Historical Observations',
    color='black',
    linewidth=1.5
)

# Plot validation data
test_dates = df_model.iloc[split:]['data'].values
plt.plot(
    test_dates,
    pred_nbr,
    'o',
    label='Validation (Test)',
    color='blue',
    alpha=0.6
)

# Plot future forecast
plt.plot(
    f_dates,
    f_nbr,
    '--',
    label='Future Forecast (2026)',
    color='red',
    linewidth=2
)

# Highlight prediction zone
plt.axvspan(
    df['data'].values[-1],
    f_dates[-1],
    color='red',
    alpha=0.05,
    label='Prediction Zone'
)

plt.title('Machine Learning Model: NBR Forecast – Pantanal', fontsize=15)
plt.xlabel('Date')
plt.ylabel('Mean NBR')
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig('ml_nbr_forecast_pantanal.png')
print("\nPlot 'ml_nbr_forecast_pantanal.png' successfully generated.")
