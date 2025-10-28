import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ==============================================================================
# 1. MEMUAT DAN MEMPERSIAPKAN DATASET (CIFAR-10)
# ==============================================================================
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

img_height, img_width, img_channels = x_train.shape[1:]
num_classes = len(tf.unique(y_train[:, 0]).y)


# ==============================================================================
# 2. MEMBANGUN ARSITEKTUR MODEL CNN
# ==============================================================================
model = Sequential([
    Input(shape=(img_height, img_width, img_channels)),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])


# ==============================================================================
# 3. COMPILE MODEL
# ==============================================================================
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# ==============================================================================
# 4. MELATIH (FIT) MODEL
# ==============================================================================
# (Kita akan menggunakan epoch yang lebih sedikit untuk contoh ini agar cepat)
epochs = 15

print("\n--- Memulai Pelatihan Model ---")
history = model.fit(
    x_train,
    y_train,
    epochs=epochs,
    batch_size=64,
    validation_data=(x_test, y_test) # Data uji digunakan sebagai data validasi
)
print("--- Pelatihan Model Selesai ---\n")


# ==============================================================================
# 5. EVALUASI MODEL (BAGIAN YANG ANDA TANYAKAN)
# ==============================================================================
print("--- Mengevaluasi Model pada Data Uji ---")

# 'x_test' dan 'y_test' berfungsi sama seperti 'test_generator'
# yaitu menyediakan data uji yang belum pernah dilihat model.
evaluation = model.evaluate(x_test, y_test, verbose=2)

# Menampilkan hasil evaluasi dengan format yang lebih mudah dibaca
print("\n--- Hasil Evaluasi Akhir ---")
print(f"Loss (Kesalahan)   : {evaluation[0]:.4f}")
print(f"Accuracy (Akurasi) : {evaluation[1] * 100:.2f}%")