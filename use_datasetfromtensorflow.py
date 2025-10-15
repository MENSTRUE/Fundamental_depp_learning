# ====================================================================================
# LATIHAN MENGGUNAKAN DATASET DARI TENSORFLOW
# File ini menggabungkan dua metode:
# 1. Menggunakan tf.keras.datasets (untuk model sederhana)
# 2. Menggunakan tensorflow_datasets (untuk model CNN)
# ====================================================================================

# ------------------------------------------------------------------------------------
# Bagian Impor Library
# Semua library yang dibutuhkan diimpor di sini.
# ------------------------------------------------------------------------------------
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

# ====================================================================================
# BAGIAN 1: MEMBUAT MODEL MENGGUNAKAN tf.keras.datasets
# ====================================================================================

print("=====================================================")
print("=== Memulai Bagian 1: tf.keras.datasets           ===")
print("=====================================================\n")

# 1. Memuat dataset MNIST
# Fungsi load_data() secara otomatis membagi data menjadi set latih dan uji
print("Langkah 1.1: Memuat dataset MNIST dari tf.keras.datasets...")
mnist = tf.keras.datasets.mnist
(gambar_latih_keras, label_latih_keras), (gambar_testing_keras, label_testing_keras) = mnist.load_data()
print(f"Data latih berhasil dimuat: {gambar_latih_keras.shape[0]} gambar")
print(f"Data testing berhasil dimuat: {gambar_testing_keras.shape[0]} gambar\n")

# 2. Normalisasi Data Gambar
# Nilai piksel diubah dari rentang 0-255 menjadi 0-1 agar model belajar lebih efisien.
print("Langkah 1.2: Melakukan normalisasi data gambar...")
gambar_latih_keras  = gambar_latih_keras / 255.0
gambar_testing_keras = gambar_testing_keras / 255.0
print("Normalisasi selesai.\n")

# 3. Membangun Arsitektur Model Sequential Sederhana (Dense Layers)
print("Langkah 1.3: Membangun arsitektur model Sequential...")
model_keras = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax') # 10 output untuk 10 kelas angka (0-9)
])
model_keras.summary()
print("\n")

# 4. Meng-compile Model
# Menentukan optimizer, fungsi loss, dan metrik yang akan digunakan.
print("Langkah 1.4: Meng-compile model...")
model_keras.compile(optimizer = tf.optimizers.Adam(),
                    loss = 'sparse_categorical_crossentropy',
                    metrics=['accuracy'])
print("Compile selesai.\n")

# 5. Melatih Model
print("Langkah 1.5: Memulai pelatihan model...")
model_keras.fit(gambar_latih_keras, label_latih_keras, epochs=5)
print("Pelatihan model selesai.\n")

# 6. Mengevaluasi Model
# Menguji performa model terhadap data yang belum pernah dilihat sebelumnya.
print("Langkah 1.6: Mengevaluasi model dengan data testing...")
loss, accuracy = model_keras.evaluate(gambar_testing_keras, label_testing_keras, verbose=2)
print(f"\nHasil Evaluasi Model Keras -> Akurasi: {accuracy*100:.2f}%\n")


# ====================================================================================
# BAGIAN 2: MEMBUAT MODEL MENGGUNAKAN tensorflow_datasets (TFDS)
# ====================================================================================

print("\n\n=====================================================")
print("=== Memulai Bagian 2: tensorflow_datasets (TFDS)  ===")
print("=====================================================\n")

# 1. Memuat dataset MNIST menggunakan TFDS
# tfds.as_numpy mengubah output dari tf.data.Dataset menjadi array NumPy.
# batch_size=-1 memuat seluruh dataset ke dalam memori.
print("Langkah 2.1: Memuat dataset MNIST dari TFDS...")
(train_images_tfds, train_labels_tfds), (test_images_tfds, test_labels_tfds) = tfds.as_numpy(tfds.load(
    'mnist',
    split=['train', 'test'],
    batch_size=-1,
    as_supervised=True))
print(f"Data latih berhasil dimuat: {train_images_tfds.shape[0]} gambar")
print(f"Data testing berhasil dimuat: {test_images_tfds.shape[0]} gambar\n")

# 2. Normalisasi dan Reshape Data
# Normalisasi (0-1) dan reshape untuk menambahkan dimensi channel (1 untuk grayscale).
# Ini penting karena layer Conv2D mengharapkan input 4D: (batch, height, width, channels).
print("Langkah 2.2: Melakukan normalisasi dan reshape data untuk CNN...")
train_images_tfds = train_images_tfds.astype('float32') / 255.0
test_images_tfds = test_images_tfds.astype('float32') / 255.0
# Menambahkan dimensi channel
train_images_tfds = np.expand_dims(train_images_tfds, axis=-1)
test_images_tfds = np.expand_dims(test_images_tfds, axis=-1)
print(f"Bentuk data latih baru: {train_images_tfds.shape}\n")

# 3. Membangun Arsitektur Model CNN (Convolutional Neural Network)
print("Langkah 2.3: Membangun arsitektur model CNN...")
model_tfds = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])
model_tfds.summary()
print("\n")

# 4. Meng-compile Model
print("Langkah 2.4: Meng-compile model CNN...")
model_tfds.compile(
    optimizer=tf.keras.optimizers.Adam(), # Menggunakan Adam untuk perbandingan
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy'],
)
print("Compile selesai.\n")

# 5. Melatih Model CNN
print("Langkah 2.5: Memulai pelatihan model CNN...")
model_tfds.fit(train_images_tfds, train_labels_tfds, batch_size=64, epochs=5)
print("Pelatihan model CNN selesai.\n")

# 6. Mengevaluasi Model CNN
print("Langkah 2.6: Mengevaluasi model CNN dengan data testing...")
loss_cnn, accuracy_cnn = model_tfds.evaluate(test_images_tfds, test_labels_tfds, verbose=2)
print(f"\nHasil Evaluasi Model TFDS (CNN) -> Akurasi: {accuracy_cnn*100:.2f}%")

print("\n\n=====================================================")
print("=== SEMUA PROSES SELESAI                       ===")
print("=====================================================")

