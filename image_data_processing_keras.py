import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

# --- KODE DEFINISI LAYER (INI SUDAH BENAR) ---
IMG_SIZE = 180

resize_and_rescale = tf.keras.Sequential([
    layers.Resizing(IMG_SIZE, IMG_SIZE),
    layers.Rescaling(1./255)
])

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
])

# --- BAGIAN YANG DIPERBAIKI UNTUK MENGGUNAKAN GAMBAR LOKAL ANDA ---

# 1. Langsung gunakan path ke gambar Anda
image_path = "image/fritzy.jpg"

try:
    # Muat gambar dari path lokal
    img = tf.keras.utils.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
except FileNotFoundError:
    print(f"Error: Tidak dapat menemukan file di '{image_path}'")
    print("Pastikan file 'fritzy.jpg' ada di dalam folder 'image' di direktori yang sama dengan notebook Anda.")
    exit() # Hentikan eksekusi jika file tidak ditemukan

# Ubah gambar menjadi format array
img_array = tf.keras.utils.img_to_array(img)
# Buat "batch" yang berisi satu gambar
image_batch = tf.expand_dims(img_array, 0)

# --- BAGIAN UNTUK MENAMPILKAN HASIL (INI SUDAH BENAR) ---

plt.figure(figsize=(10, 10))
plt.suptitle("Augmentasi menggunakan Keras Layers", fontsize=16)

# Tampilkan gambar asli
plt.subplot(3, 3, 1)
plt.imshow(img)
plt.title("Gambar Asli")
plt.axis("off")

# Buat dan tampilkan 8 gambar hasil augmentasi
for i in range(8):
    ax = plt.subplot(3, 3, i + 2)
    augmented_batch = data_augmentation(image_batch)
    plt.imshow(augmented_batch[0])
    plt.axis("off")

plt.show()