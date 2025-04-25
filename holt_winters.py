import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing


df = pd.read_csv('veriler.csv', parse_dates=['Fecha'], index_col='Fecha')
demand_hourly = df['Demanda'].values

total_len = len(demand_hourly)
train_size = int(total_len * 0.8)
val_size = int(total_len * 0.1)
test_size = total_len - train_size - val_size 

train_data = demand_hourly[:train_size]
val_data = demand_hourly[train_size:train_size + val_size]
test_data = demand_hourly[train_size + val_size:]

print(f"Eğitim seti uzunluğu: {len(train_data)}")
print(f"Doğrulama seti uzunluğu: {len(val_data)}")
print(f"Test seti uzunluğu: {len(test_data)}")

model = ExponentialSmoothing(
    train_data,
    trend= None,
    seasonal="add",
    seasonal_periods=168,  # Haftalık periyot (7 gün * 24 saat)
    initialization_method="estimated",
    
)
model_fit = model.fit(smoothing_level=0.2,       # alpha (seviye düzeltme katsayısı)
     smoothing_seasonal=0.39,  # gamma (mevsimsel düzeltme katsayısı)
      optimized = False
    )


val_predictions = model_fit.forecast(len(val_data))


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    epsilon = 1e-10
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) + 1e-10
    diff = np.abs(y_true - y_pred)
    smape = np.mean(2 * diff / denominator) * 100
    return smape

val_mae = mean_absolute_error(val_data, val_predictions)
val_rmse = np.sqrt(mean_squared_error(val_data, val_predictions))
val_r2 = r2_score(val_data, val_predictions)
val_mape = mean_absolute_percentage_error(val_data, val_predictions)
val_smape = symmetric_mean_absolute_percentage_error(val_data, val_predictions)

print("\nDoğrulama Seti Performansı:")
print(f"MAE: {val_mae:.2f}")
print(f"RMSE: {val_rmse:.2f}")
print(f"R^2: {val_r2:.2f}")
print(f"MAPE: {val_mape:.2f}%")
print(f"SMAPE: {val_smape:.2f}%")

# Test seti tahmini ve değerlendirme
test_predictions = model_fit.forecast(len(test_data))

test_mae = mean_absolute_error(test_data, test_predictions)
test_rmse = np.sqrt(mean_squared_error(test_data, test_predictions))
test_r2 = r2_score(test_data, test_predictions)
test_mape = mean_absolute_percentage_error(test_data, test_predictions)
test_smape = symmetric_mean_absolute_percentage_error(test_data, test_predictions)

print("\nTest Seti Performansı:")
print(f"MAE: {test_mae:.2f}")
print(f"RMSE: {test_rmse:.2f}")
print(f"R^2: {test_r2:.2f}")
print(f"MAPE: {test_mape:.2f}%")
print(f"SMAPE: {test_smape:.2f}%")


plt.figure(figsize=(14, 7))
plt.plot(test_data, label='Gerçek Test Verisi', color='blue')
plt.plot(test_predictions, label='Test Tahminleri', color='red', alpha=0.7)
plt.title('Test Seti: Gerçek Veri vs Tahmin')
plt.xlabel('Zaman (Saat)')
plt.ylabel('Talep')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



