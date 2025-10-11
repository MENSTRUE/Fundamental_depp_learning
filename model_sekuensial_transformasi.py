import tensorflow as tf
import numpy as np
from tensorflow import keras

# LANGKAH 1: SIAPKAN DATA (BAHAN-BAHANNYA)
X = np.array([-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
Y = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0], dtype=float)

# LANGKAH 2: BUAT MODEL DENGAN BEBERAPA LAYER (CETAK BIRU ANDA)
# Inilah kode yang Anda berikan tadi
model = tf.keras.Sequential([
    # Tahap Transformasi 1
    tf.keras.layers.Dense(units=20, input_shape=[1], activation='relu'), # Tambahkan aktivasi relu

    # Tahap Transformasi 2
    tf.keras.layers.Dense(units=15, activation='relu'),

    # Tahap Transformasi 3
    tf.keras.layers.Dense(units=10, activation='relu'),

    # Tahap Transformasi Terakhir (Output)
    tf.keras.layers.Dense(units=1)
])

# UNTUK MELIHAT STRUKTUR MODEL ANDA (OPSIONAL TAPI SANGAT BERGUNA)
print("Struktur Model:")
model.summary()

# LANGKAH 3: KOMPILASI MODEL (CARA MELATIHNYA)
model.compile(optimizer='sgd', loss='mean_squared_error')

# LANGKAH 4: LATIH MODEL (MULAI PROSESNYA)
print("\nMemulai pelatihan model dengan beberapa layer...")
model.fit(X, Y, epochs=500, verbose=0) # verbose=0 agar tidak terlalu ramai outputnya
print("Pelatihan selesai.")

# LANGKAH 5: PREDIKSI
print("\nHasil prediksi untuk input [4, 5]:")
prediction = model.predict(np.array([4, 5])).round()
print(prediction)