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

# Transformasi untuk membuat gambar lebih terang
brightness_adjustment = transforms.Compose([
    # Faktor > 1 membuat gambar lebih terang (misal: 1.0 - 1.5)
    transforms.ColorJitter(brightness=(1, 1.5))
])
brightened_image = brightness_adjustment(image)

# Menampilkan citra yang lebih terang
plt.subplot(1, 3, 2)
plt.title("Brightened Image")
plt.imshow(brightened_image)
plt.axis('off')

# Transformasi untuk membuat gambar lebih gelap
darkness_adjustment = transforms.Compose([
    # Faktor < 1 membuat gambar lebih gelap (misal: 0.5 - 1.0)
    transforms.ColorJitter(brightness=(0.5, 1))
])
darkened_image = darkness_adjustment(image)

# Menampilkan citra yang lebih gelap
plt.subplot(1, 3, 3)
plt.title("Darkened Image")
plt.imshow(darkened_image)
plt.axis('off')

plt.tight_layout()
plt.show()