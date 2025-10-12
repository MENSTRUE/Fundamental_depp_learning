# 1. Impor library yang dibutuhkan
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

# 2. Muat dataset dari URL
# Dataset akan diubah menjadi sebuah DataFrame menggunakan Pandas
df = pd.read_csv('https://drive.google.com/uc?id=1roJ83AbgzDcvRr0Gwud0BmdUQx-oSG-w')

# 3. Pra-pemrosesan Data
# Membuang kolom 'Id' yang tidak diperlukan
df = df.drop(columns='Id')

# Melakukan One-Hot Encoding pada kolom 'Species'
# Ini mengubah label kategori (nama bunga) menjadi format numerik
category = pd.get_dummies(df.Species, dtype=int)

# Menggabungkan DataFrame asli dengan kolom hasil One-Hot Encoding
new_df = pd.concat([df, category], axis=1)

# Membuang kolom 'Species' yang asli karena sudah digantikan
new_df = new_df.drop(columns='Species')

# Mengonversi DataFrame menjadi numpy array agar bisa diproses model
dataset = new_df.values

# 4. Memisahkan Atribut (Fitur) dan Label
# X berisi 4 kolom pertama (fitur: SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm)
X = dataset[:,0:4]
# y berisi 3 kolom terakhir (label: Iris-setosa, Iris-versicolor, Iris-virginica)
y = dataset[:,4:7]

# 5. Normalisasi Data
# Menyamakan skala nilai pada fitur agar model belajar lebih baik
min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)

# 6. Membagi Data menjadi Data Latih dan Data Uji
# 70% data untuk melatih model, 30% untuk menguji model
X_train, X_test, Y_train, Y_test = train_test_split(X_scale, y, test_size=0.3, random_state=42) # random_state untuk hasil yang konsisten

# 7. Membangun Arsitektur Model Neural Network
model = Sequential([
    # Input layer dengan 4 neuron (sesuai jumlah fitur) dan hidden layer pertama dengan 64 neuron
    Dense(64, activation='relu', input_shape=(4,)),
    # Hidden layer kedua dengan 64 neuron
    Dense(64, activation='relu'),
    # Output layer dengan 3 neuron (sesuai jumlah kelas/label) dan aktivasi softmax untuk klasifikasi multikelas
    Dense(3, activation='softmax')
])

# 8. Mengompilasi Model
# Menentukan optimizer, loss function, dan metrik yang akan digunakan
model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 9. Melatih Model
# Proses training model dengan data latih sebanyak 100 epoch
print("Memulai proses training model...")
hist = model.fit(X_train, Y_train, epochs=100, verbose=0) # verbose=0 agar tidak menampilkan log per epoch
print("Training selesai.")

# 10. Mengevaluasi Model
# Menguji performa model menggunakan data uji yang belum pernah dilihat sebelumnya
print("\nMengevaluasi model dengan data test...")
loss, accuracy = model.evaluate(X_test, Y_test)

print(f"Akurasi pada data test: {accuracy * 100:.2f}%")
print(f"Loss pada data test: {loss:.4f}")