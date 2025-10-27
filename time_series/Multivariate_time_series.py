import pandas as pd
import tensorflow as tf

# 1. Memuat Dataset
df_multi = pd.read_csv(
    'household_power.csv',
    sep=',',
    infer_datetime_format=True,
    index_col='datetime',
    header=0
)

# 2. Fungsi Normalisasi dan Penerapannya
def normalize_series(data, min_val, max_val):
    data = data - min_val
    data = data / max_val
    return data

data_values = df_multi.values
data_normalized = normalize_series(data_values, data_values.min(axis=0), data_values.max(axis=0))

# 3. Membagi Data Latih (50%) dan Validasi (50%)
N_FEATURES = len(df_multi.columns)
SPLIT_TIME = int(len(data_normalized) * 0.5)

x_train = data_normalized[:SPLIT_TIME]
x_valid = data_normalized[SPLIT_TIME:]

# 4. Fungsi Windowed Dataset untuk Multivariate
def windowed_dataset_multi(series, batch_size, n_past=24, n_future=24, shift=1):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(size=n_past + n_future, shift=shift, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(n_past + n_future))
    ds = ds.map(lambda w: (w[:n_past], w[n_past:]))
    return ds.batch(batch_size).prefetch(1)

# 5. Mempersiapkan Data Windowed
BATCH_SIZE = 32
N_PAST = 24
N_FUTURE = 24
SHIFT = 1

train_set_multi = windowed_dataset_multi(
    series=x_train, batch_size=BATCH_SIZE,
    n_past=N_PAST, n_future=N_FUTURE,
    shift=SHIFT
)
valid_set_multi = windowed_dataset_multi(
    series=x_valid, batch_size=BATCH_SIZE,
    n_past=N_PAST, n_future=N_FUTURE,
    shift=SHIFT
)

# 6. Membangun Model
model_multi = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, input_shape=(N_PAST, N_FEATURES)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(N_FEATURES)
])

# 7. Callback untuk Menghentikan Pelatihan
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('mae') < 0.055 and logs.get('val_mae') < 0.055:
            print("\nMAE pada training dan validasi sudah di bawah 0.055, menghentikan pelatihan!")
            self.model.stop_training = True

callbacks = myCallback()

# 8. Mengompilasi Model
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model_multi.compile(
    loss='mae',
    optimizer=optimizer,
    metrics=["mae"]
)

# 9. Melatih Model
print("\n--- Memulai Pelatihan Model Multivariate ---")
history_multi = model_multi.fit(
    train_set_multi,
    validation_data=valid_set_multi,
    epochs=100,
    callbacks=[callbacks],
    verbose=1
)
print("--- Pelatihan Model Multivariate Selesai ---\n")

# 10. Contoh Prediksi
print("Contoh hasil prediksi untuk satu batch data latih:")
train_pred = model_multi.predict(train_set_multi.take(1))
print(train_pred[0][0])