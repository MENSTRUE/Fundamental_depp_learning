# 1. Impor library untuk visualisasi
import matplotlib.pyplot as plt

# Pastikan variabel 'hist' sudah ada dari hasil proses training model.fit() sebelumnya

# 2. Membuat Plot untuk Model Loss
# Mengambil data 'loss' dari history pelatihan
plt.plot(hist.history['loss'])

# Memberi judul pada plot
plt.title('Grafik Model Loss')

# Memberi label pada sumbu y
plt.ylabel('Loss')

# Memberi label pada sumbu x
plt.xlabel('Epoch')

# Menampilkan legenda
plt.legend(['Train'], loc='upper right')

# Menampilkan plot
plt.show()


# 3. Membuat Plot untuk Model Accuracy
# Mengambil data 'accuracy' dari history pelatihan
plt.plot(hist.history['accuracy'])

# Memberi judul pada plot
plt.title('Grafik Model Accuracy')

# Memberi label pada sumbu y
plt.ylabel('Accuracy')

# Memberi label pada sumbu x
plt.xlabel('Epoch')

# Menampilkan legenda
plt.legend(['Train'], loc='lower right')

# Menampilkan plot
plt.show()