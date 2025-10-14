# ==============================================================================
# CONTOH LENGKAP OPTIMASI PELATIHAN DENGAN CALLBACKS DALAM SATU FILE
# ==============================================================================

# -----------------------------------------------------
# 1. IMPORT LIBRARY YANG DIBUTUHKAN
# -----------------------------------------------------
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

print(f"TensorFlow Version: {tf.__version__}")

# -----------------------------------------------------
# 2. PERSIAPAN DATA
# -----------------------------------------------------
print("\nLangkah 2: Memuat dan Mempersiapkan Data...")
# Coba muat data dari URL, jika gagal, muat dari file lokal 'Iris.csv'
try:
    df = pd.read_csv('https://drive.google.com/uc?id=1roJ83AbgzDcvRr0Gwud0BmdUQx-oSG-w')
    print("Data berhasil dimuat dari Google Drive.")
except Exception as e:
    print(f"Gagal memuat data dari Google Drive: {e}")
    print("Mencoba memuat dari file lokal 'Iris.csv'...")
    try:
        df = pd.read_csv('Iris.csv')
        print("Data berhasil dimuat dari file lokal.")
    except FileNotFoundError:
        print("Error: File 'Iris.csv' tidak ditemukan. Silakan unduh dan letakkan di folder yang sama.")
        exit()

# Menghapus kolom 'Id' yang tidak diperlukan
df = df.drop(columns='Id')

# One-Hot Encoding pada kolom target 'Species'
category = pd.get_dummies(df.Species, dtype=int)
new_df = pd.concat([df, category], axis=1)
new_df = new_df.drop(columns='Species')

# Memisahkan atribut (fitur) dan label
dataset = new_df.values
X = dataset[:, 0:4]  # Fitur: 4 kolom pertama
y = dataset[:, 4:7]  # Label: 3 kolom terakhir (hasil one-hot)

# Normalisasi fitur menggunakan MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Membagi data menjadi data latih (training) dan data uji (testing)
# random_state=42 agar hasil pembagian data selalu sama setiap kali dijalankan
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
print("Persiapan data selesai.")

# -----------------------------------------------------
# 3. MEMBANGUN MODEL JARINGAN SARAF TIRUAN
# -----------------------------------------------------
print("\nLangkah 3: Membangun Model...")
model = Sequential([
    Dense(64, activation='relu', input_shape=(4,)),
    Dense(64, activation='relu'),
    # Catatan: Dropout(0,5) pada teks asli sepertinya typo, seharusnya Dropout(0.5)
    Dropout(0.5),
    BatchNormalization(),
    Dense(3, activation='softmax')
])

# Meng-compile model
model.compile(
    optimizer='Adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Menampilkan ringkasan arsitektur model
model.summary()

# -----------------------------------------------------
# 4. KONFIGURASI CALLBACKS
# -----------------------------------------------------
print("\nLangkah 4: Mengonfigurasi Callbacks...")

# --- Opsi A: Custom Callback (seperti pada contoh Anda) ---
# Callback ini akan menghentikan pelatihan jika akurasi training > 95%
class MyCustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.95:
            print("\nAkurasi training > 95%, pelatihan dihentikan oleh Custom Callback!")
            self.model.stop_training = True

# --- Opsi B: Callbacks Bawaan (Praktik Terbaik) ---
# 1. EarlyStopping: Menghentikan pelatihan jika performa pada data validasi tidak membaik.
#    Ini sangat berguna untuk mencegah overfitting.
early_stopping_callback = EarlyStopping(
    monitor='val_loss',         # Pantau loss pada data validasi
    patience=10,                # Tunggu 10 epoch sebelum berhenti jika tidak ada perbaikan
    verbose=1,                  # Tampilkan pesan saat berhenti
    restore_best_weights=True   # Kembalikan bobot model ke epoch terbaik
)

# 2. ModelCheckpoint: Menyimpan model dengan performa terbaik selama pelatihan.
model_checkpoint_callback = ModelCheckpoint(
    filepath='model_iris_terbaik.keras', # Nama file untuk menyimpan model
    monitor='val_accuracy',           # Pantau akurasi pada data validasi
    save_best_only=True,              # Hanya simpan yang terbaik
    mode='max',                       # Mode 'max' karena kita ingin akurasi maksimal
    verbose=1                         # Tampilkan pesan saat model disimpan
)

print("Callbacks siap digunakan.")

# -----------------------------------------------------
# 5. MELATIH MODEL DENGAN CALLBACKS
# -----------------------------------------------------
print("\nLangkah 5: Memulai Pelatihan Model...")

# Kita akan menggunakan callbacks bawaan (Opsi B) karena lebih powerful
callbacks_to_use = [
    early_stopping_callback,
    model_checkpoint_callback
]

# Jika ingin menggunakan custom callback, ganti list di atas menjadi:
# callbacks_to_use = [MyCustomCallback()]

history = model.fit(
    X_train,
    y_train,
    epochs=200,  # Set epoch tinggi, biarkan callback yang menghentikannya
    validation_data=(X_test, y_test),
    callbacks=callbacks_to_use,
    verbose=1 # Tampilkan progress bar untuk setiap epoch
)

print("\nPelatihan Selesai.")

# -----------------------------------------------------
# 6. EVALUASI MODEL
# -----------------------------------------------------
print("\nLangkah 6: Mengevaluasi Performa Model pada Data Uji...")

# Evaluasi akan menggunakan bobot terbaik yang sudah di-restore oleh EarlyStopping
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print(f"\nHasil Evaluasi pada Data Uji:")
print(f"  - Loss    : {loss:.4f}")
print(f"  - Accuracy: {accuracy:.4f} ({accuracy:.2%})")
print("\nModel dengan performa terbaik juga telah disimpan dalam file 'model_iris_terbaik.keras'")