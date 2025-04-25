import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv("veriler.csv")

# Prophet için veri formatını düzenle
prophet_df = pd.DataFrame()
prophet_df['ds'] = pd.to_datetime(df['Fecha'] + ' ' + df['Hora'])
prophet_df['y'] = df['Demanda']

total_size = len(prophet_df)
train_size = int(total_size * 0.8)
val_size = int(total_size * 0.1)

train_df = prophet_df[:train_size]
val_df = prophet_df[train_size:train_size+val_size]
test_df = prophet_df[train_size+val_size:]

print("\nVeri Seti Bölümleme:")
print("-" * 50)
print(f"Toplam veri: {total_size}")
print(f"Eğitim seti: {len(train_df)} (%80)")
print(f"Doğrulama seti: {len(val_df)} (%10)")
print(f"Test seti: {len(test_df)} (%10)")

def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    smape = np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100
    return rmse, mae, r2, mape, smape

print("\nProphet modeli eğitiliyor...")
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=True,
    changepoint_prior_scale=0.05
)
model.fit(train_df)

val_dates = pd.DataFrame({'ds': val_df['ds']})
val_predictions = model.predict(val_dates)
val_metrics = calculate_metrics(val_df['y'].values, val_predictions['yhat'].values)

test_dates = pd.DataFrame({'ds': test_df['ds']})
test_predictions = model.predict(test_dates)
test_metrics = calculate_metrics(test_df['y'].values, test_predictions['yhat'].values)

print("\nDoğrulama Seti Metrikleri:")
print(f"RMSE: {val_metrics[0]:.2f}")
print(f"MAE: {val_metrics[1]:.2f}")
print(f"R²: {val_metrics[2]:.4f}")
print(f"MAPE: {val_metrics[3]:.2f}%")
print(f"sMAPE: {val_metrics[4]:.2f}%")

print("\nTest Seti Metrikleri:")
print(f"RMSE: {test_metrics[0]:.2f}")
print(f"MAE: {test_metrics[1]:.2f}")
print(f"R²: {test_metrics[2]:.4f}")
print(f"MAPE: {test_metrics[3]:.2f}%")
print(f"sMAPE: {test_metrics[4]:.2f}%")

plt.figure(figsize=(15,7))
plt.plot(test_df['ds'], test_df['y'], label='Gerçek Değerler', color='blue')
plt.plot(test_predictions['ds'], test_predictions['yhat'], label='Tahmin (yhat)', color='red')
plt.fill_between(test_predictions['ds'], 
                 test_predictions['yhat_lower'], 
                 test_predictions['yhat_upper'], 
                 color='green', alpha=0.3, label='Tahmin Güven Aralığı')
plt.title('Test Seti: Gerçek Değerler ve Prophet Tahminleri')
plt.xlabel('Zaman (ds)')
plt.ylabel('Demanda')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


