import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import STL
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv('veriler.csv')
date_range = pd.date_range(start='2022-01-01 00:00', end='2024-03-31 23:00', freq='H')
df.index = date_range

series = df['Demanda']

stl = STL(series, period=168)
result = stl.fit()

trend = result.trend
seasonal = result.seasonal
residual = result.resid

result.plot()
plt.suptitle('STL Decomposition')
plt.show()

# mevsimsellik etkisini çıkar
series_deseasonalized = trend + residual

def adf_test(series):
    print("ADF Testi Sonuçları:")
    result = adfuller(series.dropna())  # NaN değerleri çıkar
    print(f"ADF Statistic: {result[0]:.6f}")
    print(f"p-value: {result[1]:.6f}")
    for key, value in result[4].items():
        print(f"Critical Value ({key}): {value:.6f}")
    if result[1] <= 0.05:
        print("=> Veri durağan.")
        return True
    else:
        print("=> Veri durağan değil.")
        return False

is_stationary = adf_test(series_deseasonalized)

#ACF/PACF grafikleri
fig, axes = plt.subplots(1, 2, figsize=(15,5))
plot_acf(series_deseasonalized.dropna(), ax=axes[0], lags=40)
axes[0].set_title('ACF Grafiği (Deseasonalized)')
plot_pacf(series_deseasonalized.dropna(), ax=axes[1], lags=40, method='ywm')
axes[1].set_title('PACF Grafiği (Deseasonalized)')
plt.show()

# veriyi eğitim, doğrulama ve test setlerine ayır
series_clean = series_deseasonalized.dropna()
n = len(series_clean)
train_end = int(n * 0.8)
val_end = int(n * 0.9)

train = series_clean.iloc[:train_end]
val = series_clean.iloc[train_end:val_end]
test = series_clean.iloc[val_end:]

print(f"Eğitim seti uzunluğu: {len(train)}")
print(f"Doğrulama seti uzunluğu: {len(val)}")
print(f"Test seti uzunluğu: {len(test)}")

model = ARIMA(train, order=(4, 1, 4))
model_fit = model.fit()

print(model_fit.summary())

# forecast ile tahmin
pred = model_fit.forecast(steps=len(test))
pred.index = test.index

# tahminlere mevsimsellik bileşeni ekle (orijinal ölçeğe döndürmek için)
pred_final = pred + seasonal.loc[pred.index]

y_true = series.loc[pred_final.index]  # Orijinal serinin gerçek değerleri
y_pred = pred_final

def smape(y_true, y_pred):
    return 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

r2 = r2_score(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
smape_val = smape(y_true.values, y_pred.values)

print("\nModel Performans Metrikleri (STL decomposition sonrası):")
print(f"R^2: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MAPE: {mape:.4f}%")
print(f"RMSE: {rmse:.4f}")
print(f"SMAPE: {smape_val:.4f}%")

plt.figure(figsize=(15,6))
plt.plot(y_true.index, y_true.values, label='Gerçek Değerler', color='blue')
plt.plot(y_pred.index, y_pred.values, label='Tahmin Değerleri', color='red', alpha=0.7)
plt.title('Test Seti: Gerçek ve Tahmin Edilen Değerler')
plt.xlabel('Tarih')
plt.ylabel('Demanda')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
