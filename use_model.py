# Langkah 1: Impor library yang dibutuhkan
import tensorflow as tf
import numpy as np
from tensorflow import keras

# Langkah 2: Siapkan data dalam bentuk NumPy array
# Atribut (input)
X = np.array([-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
# Label (output yang diharapkan)
Y = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0], dtype=float)

# Langkah 3: Buat model Jaringan Saraf Tiruan (JST) yang paling sederhana
# Model Sequential dengan satu layer Dense, yang memiliki satu neuron.
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# Langkah 4: Tentukan optimizer dan loss function untuk model
# 'sgd' = Stochastic Gradient Descent
# 'mean_squared_error' = cocok untuk masalah regresi (memprediksi angka)
model.compile(optimizer='sgd', loss='mean_squared_error')

# Langkah 5: Latih model
# Model akan "belajar" dari data X dan Y sebanyak 500 kali (epochs)
print("Memulai pelatihan model...")
model.fit(X, Y, epochs=500)
print("Pelatihan selesai.")

# Langkah 6: Gunakan model untuk memprediksi data baru
# Kita ingin model menebak hasil untuk input 4 dan 5
print("\nMelakukan prediksi untuk input [4, 5]:")
prediction = model.predict(np.array([4, 5]))
print("Hasil prediksi mentah:")
print(prediction)

# Langkah 7 (Opsional): Bulatkan hasil prediksi agar lebih presisi
print("\nMelakukan prediksi dengan pembulatan:")
rounded_prediction = model.predict(np.array([4, 5])).round()
print("Hasil prediksi yang dibulatkan:")
print(rounded_prediction)