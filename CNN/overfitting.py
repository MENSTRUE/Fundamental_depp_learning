import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt

# 1. MEMUAT DAN MEMPERSIAPKAN DATA
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

img_height, img_width, img_channels = x_train.shape[1:]
num_classes = len(tf.unique(y_train[:, 0]).y)

# 2. MEMBANGUN MODEL DENGAN TEKNIK ANTI-OVERFITTING
model = Sequential()
model.add(Input(shape=(img_height, img_width, img_channels)))

# Menambahkan L2 Regularization pada layer Conv2D
# kernel_regularizer=l2(0.001) menambahkan penalti pada bobot layer ini
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)))
model.add(MaxPooling2D((2, 2)))
# Menambahkan Dropout
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))

model.add(Flatten())

# Menambahkan L2 Regularization dan Dropout pada layer Dense
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))

# 3. COMPILE MODEL
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# 4. MENDEFINISIKAN CALLBACK UNTUK EARLY STOPPING
# Pelatihan akan berhenti jika 'val_loss' (loss pada data validasi) tidak membaik
# selama 10 epoch berturut-turut (patience=10).
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True, # Kembalikan ke bobot terbaik saat pelatihan berhenti
    verbose=1
)

# 5. MELATIH MODEL
epochs = 100 # Atur jumlah epoch maksimal yang tinggi, biarkan EarlyStopping yang memutuskan

history = model.fit(
    x_train,
    y_train,
    epochs=epochs,
    batch_size=64,
    validation_data=(x_test, y_test),
    callbacks=[early_stopping] # Menggunakan callback EarlyStopping
)

# 6. VISUALISASI HASIL UNTUK MELIHAT EFEK OVERFITTING
plt.figure(figsize=(12, 5))

# Plot akurasi
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# 7. EVALUASI AKHIR
print("\n--- Mengevaluasi Model Akhir ---")
loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"Loss: {loss:.4f}")
print(f"Accuracy: {accuracy*100:.2f}%")