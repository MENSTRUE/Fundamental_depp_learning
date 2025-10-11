# ===================================================================
# LANGKAH 1: IMPORT SEMUA LIBRARY YANG DIBUTUHKAN
# ===================================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ===================================================================
# LANGKAH 2: MEMUAT DAN MEMERIKSA DATA
# ===================================================================

# Anda bisa pilih salah satu dari dua opsi di bawah ini untuk memuat data
# Opsi 1: Dari file lokal 'citrus.csv'
# df = pd.read_csv('citrus.csv')

# Opsi 2: Langsung dari Google Drive (lebih mudah, tidak perlu download)
file_url = 'https://drive.google.com/uc?id=1XZbZ5Z7MBXYaR3UO1HbQ4ncPi4R3HsyF'
df = pd.read_csv(file_url)

print("--- 5 Baris Data Pertama ---")
print(df.head())
print("\n--- Informasi Dataset ---")
df.info()

# ===================================================================
# LANGKAH 3: PRA-PEMROSESAN DATA
# ===================================================================

# Mengubah label kategori ('orange', 'grapefruit') menjadi angka (1, 0)
# Menggunakan LabelEncoder adalah cara yang lebih baik dan umum digunakan
label_encoder = LabelEncoder()
df['name'] = label_encoder.fit_transform(df['name'])
print("\n--- Data Setelah Label Diubah Menjadi Angka ---")
print(df.head())

# Mengubah DataFrame menjadi NumPy array agar bisa diproses model
dataset = df.values

# Memisahkan antara fitur (X) dan label (y)
# X -> semua baris, kolom ke-1 sampai ke-5 (diameter, weight, red, green, blue)
X = dataset[:, 1:6].astype(float) # Pastikan tipe datanya float
# y -> semua baris, hanya kolom ke-0 (name)
y = dataset[:, 0]

# Normalisasi fitur (X) agar nilainya berada dalam rentang 0-1
# Ini penting agar model belajar lebih cepat dan stabil
scaler = MinMaxScaler()
X_scale = scaler.fit_transform(X)

# Membagi data menjadi data latih (70%) dan data uji (30%)
X_train, X_test, Y_train, Y_test = train_test_split(X_scale, y, test_size=0.3, random_state=42) # random_state agar hasilnya konsisten

# ===================================================================
# LANGKAH 4: MEMBANGUN MODEL NEURAL NETWORK
# ===================================================================

# Membuat model Sequential
model = Sequential([
    # Input layer & Hidden layer pertama: 32 neuron, fungsi aktivasi 'relu'
    # input_shape=(5,) karena kita punya 5 fitur (diameter, weight, r, g, b)
    Dense(32, activation='relu', input_shape=(5,)),

    # Hidden layer kedua: 32 neuron, fungsi aktivasi 'relu'
    Dense(32, activation='relu'),

    # Output layer: 1 neuron, fungsi aktivasi 'sigmoid'
    # Sigmoid wajib untuk klasifikasi biner karena menghasilkan probabilitas 0-1
    Dense(1, activation='sigmoid')
])

print("\n--- Arsitektur Model ---")
model.summary()

# ===================================================================
# LANGKAH 5: MENGOMPILASI DAN MELATIH MODEL
# ===================================================================

# Mengompilasi model dengan optimizer, loss function, dan metrik
model.compile(optimizer='sgd',
              loss='binary_crossentropy', # Wajib untuk klasifikasi biner
              metrics=['accuracy'])

# Melatih model dengan data training
print("\n--- Memulai Pelatihan Model ---")
model.fit(X_train, Y_train, epochs=100, verbose=1) # verbose=1 untuk menampilkan progress

# ===================================================================
# LANGKAH 6: MENGEVALUASI MODEL
# ===================================================================

# Mengevaluasi performa model pada data uji yang belum pernah dilihat sebelumnya
print("\n--- Mengevaluasi Model pada Data Uji ---")
loss, accuracy = model.evaluate(X_test, Y_test)
print(f"Loss pada data uji: {loss:.4f}")
print(f"Akurasi pada data uji: {accuracy*100:.2f}%")