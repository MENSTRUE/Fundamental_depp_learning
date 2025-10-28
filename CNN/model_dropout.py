import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Ganti nilai-nilai ini sesuai dengan dataset Anda
img_height = 150
img_width = 150
img_channels = 3
num_classes = 10

# Inisialisasi model Sequential
model_with_dropout = Sequential()

# --- PERBAIKAN UTAMA: Mendefinisikan bentuk input secara eksplisit ---
# Ini adalah cara modern yang direkomendasikan Keras dan akan menghilangkan UserWarning.
model_with_dropout.add(Input(shape=(img_height, img_width, img_channels)))

# Layer Konvolusi 1 & Pooling 1
# 'input_shape' tidak lagi diperlukan di sini karena sudah didefinisikan di atas.
model_with_dropout.add(Conv2D(32, (3, 3), activation='relu'))
model_with_dropout.add(MaxPooling2D((2, 2)))
model_with_dropout.add(Dropout(0.25))

# Layer Konvolusi 2 & Pooling 2
model_with_dropout.add(Conv2D(64, (3, 3), activation='relu'))
model_with_dropout.add(MaxPooling2D((2, 2)))
model_with_dropout.add(Dropout(0.25))

# Layer Konvolusi 3
model_with_dropout.add(Conv2D(64, (3, 3), activation='relu'))

# Meratakan (flatten) output untuk dihubungkan ke layer Dense
model_with_dropout.add(Flatten())
model_with_dropout.add(Dropout(0.5))

# Layer Dense (Fully Connected)
model_with_dropout.add(Dense(64, activation='relu'))
model_with_dropout.add(Dropout(0.5))

# Layer Output untuk klasifikasi
model_with_dropout.add(Dense(num_classes, activation='softmax'))

# Menampilkan ringkasan arsitektur model
print("\n--- Arsitektur Model CNN (Sudah Diperbaiki) ---")
model_with_dropout.summary()