import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
import math
import matplotlib.pyplot as plt

df = pd.read_csv('veriler.csv')

df['Fecha'] = pd.to_datetime(df['Fecha'])

data = df['Demanda'].values.reshape(-1, 1)

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

total_len = len(data_scaled)
train_size = int(total_len * 0.8)
val_size = int(total_len * 0.1)
test_size = total_len - train_size - val_size

train_data = data_scaled[:train_size]
val_data = data_scaled[train_size:train_size + val_size]
test_data = data_scaled[train_size + val_size:]

def create_dataset(dataset, n_input=24, n_output=24):
    X, y = [], []
    for i in range(len(dataset) - n_input - n_output + 1):
        X.append(dataset[i:(i + n_input), 0])
        y.append(dataset[(i + n_input):(i + n_input + n_output), 0])
    return np.array(X), np.array(y)

n_input = 24
n_output = 24

X_train, y_train = create_dataset(train_data, n_input, n_output)
X_val, y_val = create_dataset(val_data, n_input, n_output)
X_test, y_test = create_dataset(test_data, n_input, n_output)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))


model = Sequential()
model.add(LSTM(128, activation='tanh', input_shape=(n_input, 1)))
model.add(Dense(n_output))  # Çok adımlı çıktı (10 adım)

optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
model.compile(optimizer=optimizer, loss='mse')

history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=256,
    validation_data=(X_val, y_val),
    verbose=2,
    shuffle=False
)
y_pred_scaled = model.predict(X_test)

y_test_inv = scaler.inverse_transform(y_test)
y_pred_inv = scaler.inverse_transform(y_pred_scaled)


def smape(y_true, y_pred):
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    epsilon = 1e-10  # sıfıra bölünmeyi önlemek için
    return np.mean(numerator / (denominator + epsilon)) * 100

# Adım bazında performans metrikleri
print("Adım bazında performans metrikleri:\n")
for i in range(n_output):
    y_true_step = y_test_inv[:, i]
    y_pred_step = y_pred_inv[:, i]

    mae = mean_absolute_error(y_true_step, y_pred_step)
    mape = np.mean(np.abs((y_true_step - y_pred_step) / (y_true_step + 1e-10))) * 100
    rmse = math.sqrt(np.mean((y_true_step - y_pred_step) ** 2))
    smape_val = smape(y_true_step, y_pred_step)
    r2 = r2_score(y_true_step, y_pred_step)

    print(f"Step {i+1}: MAE={mae:.4f}, MAPE={mape:.2f}%, RMSE={rmse:.4f}, SMAPE={smape_val:.2f}%, R2={r2:.4f}")


mae_avg = mean_absolute_error(y_test_inv.flatten(), y_pred_inv.flatten())
mape_avg = np.mean(np.abs((y_test_inv.flatten() - y_pred_inv.flatten()) / (y_test_inv.flatten() + 1e-10))) * 100
rmse_avg = math.sqrt(np.mean((y_test_inv.flatten() - y_pred_inv.flatten()) ** 2))
smape_avg = smape(y_test_inv.flatten(), y_pred_inv.flatten())
r2_avg = r2_score(y_test_inv.flatten(), y_pred_inv.flatten())

print("\nOrtalama performans metrikleri:")
print(f"MAE={mae_avg:.4f}, MAPE={mape_avg:.2f}%, RMSE={rmse_avg:.4f}, SMAPE={smape_avg:.2f}%, R2={r2_avg:.4f}")

plt.figure(figsize=(15, 6))

# Tüm test setindeki tahmin ve gerçek değerleri tek boyuta indiriyoruz
y_test_all = y_test_inv.flatten()
y_pred_all = y_pred_inv.flatten()

plt.plot(y_test_all, label='Gerçek Değerler')
plt.plot(y_pred_all, label='Tahmin Edilen Değerler')

plt.title('Test Seti Gerçek ve Tahmin Edilen Değerler')
plt.xlabel('Zaman Adımı (Tüm Test Seti)')
plt.ylabel('Demanda')
plt.legend()
plt.grid(True)
plt.show()



