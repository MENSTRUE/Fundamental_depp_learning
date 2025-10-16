# 1. Import library yang dibutuhkan
import tensorflow as tf

# 2. Load dan siapkan dataset MNIST
# Dataset ini berisi gambar tulisan tangan angka 0-9
mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# Normalisasi data gambar agar nilainya antara 0 dan 1
training_images  = training_images / 255.0
test_images = test_images / 255.0

# 3. Bangun arsitektur model Sequential
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# 4. Compile model, menentukan optimizer, loss function, dan metrik
model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. Latih model dengan batch_size = 32 (nilai default)
print("--- Memulai Pelatihan dengan batch_size = 32 ---")
model.fit(training_images, training_labels, batch_size=32, epochs=5)

print("\n" + "="*50 + "\n")

# 6. Latih model yang sama dengan batch_size = 128
print("--- Memulai Pelatihan dengan batch_size = 128 ---")
model.fit(training_images, training_labels, batch_size=128, epochs=5)