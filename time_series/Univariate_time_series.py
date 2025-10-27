import pandas as pd
import tensorflow as tf
from keras.layers import LSTM
import matplotlib.pyplot as plt

# 1. Memuat dan Mempersiapkan Dataset
df_uni = pd.read_csv('DailyDelhiClimateTrain.csv')
df_uni['date'] = pd.to_datetime(df_uni['date'])

dates_raw = df_uni['date'].values
temp_raw = df_uni['meantemp'].values

# 2. Membagi Data Menjadi Training (80%) dan Validation (20%)
split_percentage = 0.8
split_time = int(len(temp_raw) * split_percentage)

x_train = temp_raw[:split_time]
time_train = dates_raw[:split_time]

x_valid = temp_raw[split_time:]
time_valid = dates_raw[split_time:]

# 3. Fungsi untuk Membuat Windowed Dataset
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[-1:]))
    return ds.batch(batch_size).prefetch(1)

# 4. Mempersiapkan Windowed Dataset
window_size = 60
batch_size = 100
shuffle_buffer = 1000

train_set = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer)
validation_set = windowed_dataset(x_valid, window_size, batch_size, shuffle_buffer)

# 5. Membangun Model LSTM
model_uni = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(60, return_sequences=True, input_shape=[None, 1]),
    tf.keras.layers.LSTM(60),
    tf.keras.layers.Dense(30, activation="relu"),
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(1),
])

# 6. Mengompilasi Model
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-5, momentum=0.9)
model_uni.compile(
    loss=tf.keras.losses.Huber(),
    optimizer=optimizer,
    metrics=["mae"]
)

# 7. Melatih Model
print("\n--- Memulai Pelatihan Model Univariate ---")
history_uni = model_uni.fit(
    train_set,
    epochs=100,
    validation_data=validation_set
)
print("--- Pelatihan Model Univariate Selesai ---\n")

# 8. Visualisasi Hasil Training dan Validasi
mae = history_uni.history['mae']
val_mae = history_uni.history['val_mae']
loss = history_uni.history['loss']
val_loss = history_uni.history['val_loss']

epochs_range = range(len(mae))

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, mae, label='Training MAE')
plt.plot(epochs_range, val_mae, label='Validation MAE')
plt.legend(loc='upper right')
plt.title('Training and Validation MAE')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()