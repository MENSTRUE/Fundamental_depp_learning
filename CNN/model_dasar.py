import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# 1. MEMUAT DAN MEMPERSIAPKAN DATA (CIFAR-10)
print("Memuat dataset CIFAR-10...")
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalisasi nilai piksel dari 0-255 menjadi 0-1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Mendapatkan dimensi dan jumlah kelas dari data
img_height, img_width, img_channels = x_train.shape[1:]
num_classes = len(tf.unique(y_train[:, 0]).y)

print(f"Ukuran gambar: {img_height}x{img_width}")
print(f"Jumlah kelas: {num_classes}")


# 2. MEMBANGUN ARSITEKTUR MODEL (Kode dari materi Anda)
print("\nMembangun arsitektur model...")
model = Sequential()

# Layer Konvolusi 1 & Pooling 1
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, img_channels)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

# Layer Konvolusi 2 & Pooling 2
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

# Layer Konvolusi 3
model.add(Conv2D(64, (3, 3), activation='relu'))

# Meratakan output untuk dihubungkan ke layer Dense
model.add(Flatten())
model.add(Dropout(0.5))

# Layer Dense (Fully Connected)
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

# Layer Output
model.add(Dense(num_classes, activation='softmax'))

model.summary()


# 3. MENGOMPILASI MODEL
print("\nMengompilasi model...")
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy', # Cocok untuk label integer
    metrics=['accuracy']
)


# 4. MELATIH MODEL
print("\n--- Memulai Pelatihan Model ---")
history = model.fit(
    x_train,
    y_train,
    epochs=15, # Jumlah epoch bisa ditambah untuk akurasi lebih baik
    batch_size=64,
    validation_data=(x_test, y_test) # Menggunakan test set sebagai data validasi
)
print("--- Pelatihan Model Selesai ---\n")


# 5. MENGEVALUASI MODEL
print("\n--- Mengevaluasi Model pada Data Uji ---")
loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"\nAkurasi pada data uji: {accuracy * 100:.2f}%")