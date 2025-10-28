import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# Membaca citra
image_path = "fritzy.jpg"
image = Image.open(image_path)

# Menampilkan citra asli
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image)
plt.axis('off')

# Transformasi untuk pergeseran horizontal
horizontal_shift = transforms.Compose([
    # Menggeser citra ke kanan atau kiri secara acak hingga 20% dari lebar gambar
    transforms.RandomAffine(degrees=0, translate=(0.2, 0)),
])
shifted_image_horizontal = horizontal_shift(image)

# Menampilkan citra setelah pergeseran horizontal
plt.subplot(1, 3, 2)
plt.title("Horizontal Shift")
plt.imshow(shifted_image_horizontal)
plt.axis('off')

# Transformasi untuk pergeseran vertikal
vertical_shift = transforms.Compose([
    # Menggeser citra ke atas atau bawah secara acak hingga 20% dari tinggi gambar
    transforms.RandomAffine(degrees=0, translate=(0, 0.2)),
])
shifted_image_vertical = vertical_shift(image)

# Menampilkan citra setelah pergeseran vertikal
plt.subplot(1, 3, 3)
plt.title("Vertical Shift")
plt.imshow(shifted_image_vertical)
plt.axis('off')

plt.tight_layout()
plt.show()