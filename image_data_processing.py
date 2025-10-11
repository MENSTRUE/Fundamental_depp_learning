import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# --- LANGKAH 1: DEFINISIKAN DATAGENERATOR (INI SUDAH BENAR) ---
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# --- LANGKAH 2: GUNAKAN GAMBAR LOKAL ANDA (BAGIAN YANG DIPERBAIKI) ---
# Hapus baris get_file(), langsung definisikan path ke gambar Anda
# Pastikan path ini benar sesuai lokasi file Anda
image_path = "image/fritzy.jpg"

# Muat gambar menggunakan Keras dan ubah ukurannya
# Pastikan tidak ada error di sini. Jika ada, berarti path/lokasi file salah.
try:
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(150, 150))
except FileNotFoundError:
    print(f"Error: Tidak dapat menemukan file di '{image_path}'")
    print("Pastikan file 'fritzy.jpg' ada di dalam folder 'image' di direktori yang sama dengan notebook Anda.")
    # Hentikan eksekusi jika file tidak ditemukan
    exit()


# --- SISA KODE SAMA SEPERTI SEBELUMNYA (INI SUDAH BENAR) ---

# Ubah gambar menjadi format array (angka)
x = tf.keras.preprocessing.image.img_to_array(img)

# ImageDataGenerator perlu input 4D (batch, height, width, channels)
x = np.expand_dims(x, axis=0)

# Gunakan .flow() untuk menerapkan "resep" pada gambar kita
i = 0
plt.figure(figsize=(10, 10))

for batch in train_datagen.flow(x, batch_size=1):
    plt.subplot(3, 3, i + 1)
    augmented_image = tf.keras.preprocessing.image.array_to_img(batch[0])
    plt.imshow(augmented_image)
    plt.axis('off')

    i += 1
    if i % 9 == 0:
        break

plt.suptitle("Contoh Hasil Augmentasi pada 1 Gambar", fontsize=16)
plt.show()