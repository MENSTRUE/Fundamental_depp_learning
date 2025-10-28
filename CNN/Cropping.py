import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# Fungsi untuk melakukan cropping pada tengah gambar
def crop_image(image, size):
    width, height = image.size
    left = (width - size) / 2
    top = (height - size) / 2
    right = (width + size) / 2
    bottom = (height + size) / 2
    cropped_image = image.crop((left, top, right, bottom))
    return cropped_image

# Membaca citra
image_path = "fritzy.jpg"
image = Image.open(image_path)

# Ukuran cropping yang akan diuji
crop_sizes = [100, 150, 200, 250]

# Menampilkan citra asli
plt.figure(figsize=(12, 6))
plt.subplot(1, len(crop_sizes) + 1, 1)
plt.title("Original Image")
plt.imshow(image)
plt.axis('off')

# Melakukan cropping pada gambar dengan berbagai ukuran
for i, size in enumerate(crop_sizes):
    cropped_image = crop_image(image, size)
    plt.subplot(1, len(crop_sizes) + 1, i + 2)
    plt.title(f"Cropped {size}x{size}")
    plt.imshow(cropped_image)
    plt.axis('off')

plt.tight_layout()
plt.show()