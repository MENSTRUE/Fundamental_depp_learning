# ==============================================================================
# 1. IMPOR SEMUA LIBRARY YANG DIBUTUHKAN
# ==============================================================================
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

# ==============================================================================
# 2. PERSIAPAN DATA (LOADING & PREPROCESSING)
# ==============================================================================
# Muat dataset dari URL
df = pd.read_csv('https://drive.google.com/uc?id=1roJ83AbgzDcvRr0Gwud0BmdUQx-oSG-w')

# Buang kolom 'Id' yang tidak relevan
df = df.drop(columns='Id')

# Lakukan One-Hot Encoding pada kolom 'Species' untuk mengubah label menjadi numerik
category = pd.get_dummies(df.Species, dtype=int)
new_df = pd.concat([df, category], axis=1)
new_df = new_df.drop(columns='Species')

# Konversi DataFrame menjadi numpy array
dataset = new_df.values

# Pisahkan atribut (fitur) dan label
# X adalah 4 kolom pertama (fitur bunga)
X = dataset[:,0:4]
# y adalah 3 kolom terakhir (label one-hot encoded)
y = dataset[:,4:7]

# Lakukan normalisasi pada fitur agar nilainya berada di rentang 0-1
scaler = MinMaxScaler()
X_scale = scaler.fit_transform(X)

# Bagi dataset menjadi data latih (70%) dan data uji (30%)
X_train, X_test, Y_train, Y_test = train_test_split(X_scale, y, test_size=0.3, random_state=42)

# ==============================================================================
# 3. MEMBANGUN MODEL DENGAN DROPOUT DAN BATCH NORMALIZATION
# ==============================================================================
print("Membangun model neural network...")
model = Sequential([
    # Input layer dengan 4 neuron (sesuai jumlah fitur)
    Dense(64, activation='relu', input_shape=(4,)),
    # Hidden layer
    Dense(64, activation='relu'),
    # Layer Dropout untuk mencegah overfitting dengan menonaktifkan 50% neuron secara acak
    Dropout(0.5),
    # Layer Batch Normalization untuk menstabilkan dan mempercepat proses belajar
    BatchNormalization(),
    # Output layer dengan 3 neuron (sesuai jumlah kelas) dan aktivasi softmax
    Dense(3, activation='softmax')
])

# Menampilkan ringkasan arsitektur model
model.summary()

# ==============================================================================
# 4. MENGOMPILASI DAN MELATIH MODEL
# ==============================================================================
# Konfigurasi proses belajar model (optimizer, loss function, dan metrik)
model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Mulai proses pelatihan model
# validation_data digunakan agar kita bisa membandingkan performa model pada data latih dan data uji di setiap epoch
print("\nMemulai proses training...")
history = model.fit(X_train, Y_train,
                    epochs=100,
                    validation_data=(X_test, Y_test),
                    verbose=0) # verbose=0 agar tidak menampilkan log per epoch

print("Training selesai.")

# ==============================================================================
# 5. EVALUASI MODEL
# ==============================================================================
# Evaluasi performa akhir model pada data uji
print("\nMengevaluasi model...")
loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
print(f"Akurasi pada data test: {accuracy * 100:.2f}%")
print(f"Loss pada data test: {loss:.4f}")

# ==============================================================================
# 6. VISUALISASI HASIL PELATIHAN
# ==============================================================================
print("\nMenampilkan grafik performa model...")

# Ambil data dari history pelatihan
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(100)

# Buat subplot untuk menampilkan grafik Akurasi dan Loss
plt.figure(figsize=(14, 6))

# Grafik Akurasi
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

# Grafik Loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Tampilkan kedua grafik
plt.show()