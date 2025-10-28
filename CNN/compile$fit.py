import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ==============================================================================
# 1. MEMUAT DAN MEMPERSIAPKAN DATASET (CIFAR-10)
# ==============================================================================
print("Memuat dataset CIFAR-10...")
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalisasi nilai piksel dari rentang 0-255 menjadi rentang 0-1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Mendapatkan dimensi gambar dan jumlah kelas
img_height, img_width, img_channels = x_train.shape[1:]
num_classes = len(tf.unique(y_train[:, 0]).y)

print(f"Ukuran gambar: {img_height}x{img_width}")
print(f"Jumlah kelas: {num_classes}\n")


# ==============================================================================
# 2. MEMBANGUN ARSITEKTUR MODEL CNN DENGAN DROPOUT
# ==============================================================================
print("Membangun arsitektur model...")
model = Sequential()
model.add(Input(shape=(img_height, img_width, img_channels)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Flatten())
model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))


# ==============================================================================
# 3. COMPILE MODEL
# ==============================================================================
print("Mengompilasi model...")
# Loss function 'sparse_categorical_crossentropy' digunakan karena label y_train
# adalah integer (0, 1, 2, ...), bukan one-hot encoded.
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


# ==============================================================================
# 4. MENDEFINISIKAN CALLBACKS
# ==============================================================================
# Menyimpan model terbaik selama pelatihan
checkpoint = ModelCheckpoint(
    filepath='best_model.keras',      # Nama file untuk menyimpan model
    monitor='val_accuracy',           # Memonitor akurasi pada data validasi
    save_best_only=True,              # Hanya simpan yang terbaik
    mode='max',                       # Mode 'max' karena kita ingin akurasi tertinggi
    verbose=1
)

# Menghentikan pelatihan jika tidak ada peningkatan
early_stopping = EarlyStopping(
    monitor='val_accuracy',           # Memonitor akurasi pada data validasi
    patience=10,                      # Tunggu 10 epoch sebelum berhenti
    restore_best_weights=True,        # Kembalikan bobot ke epoch terbaik
    verbose=1
)


# ==============================================================================
# 5. MELATIH (FIT) MODEL
# ==============================================================================
epochs = 50  # Jumlah maksimum epoch untuk dijalankan

print("\n--- Memulai Pelatihan Model ---")
history = model.fit(
    x_train,
    y_train,
    epochs=epochs,
    batch_size=64,
    validation_data=(x_test, y_test),
    callbacks=[checkpoint, early_stopping]
)
print("--- Pelatihan Model Selesai ---\n")


# ==============================================================================
# 6. EVALUASI MODEL TERBAIK
# ==============================================================================
# Model yang ada di memori sekarang sudah memiliki bobot terbaik karena `restore_best_weights=True`
print("--- Mengevaluasi Model dengan Bobot Terbaik ---")
loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"\nAkurasi pada data uji: {accuracy * 100:.2f}%")

# Jika ingin, Anda juga bisa memuat model dari file yang disimpan
# from tensorflow.keras.models import load_model
# best_model = load_model('best_model.keras')
# loss, accuracy = best_model.evaluate(x_test, y_test, verbose=2)
# print(f"Akurasi dari model yang dimuat: {accuracy * 100:.2f}%")