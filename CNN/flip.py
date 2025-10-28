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

# Transformasi untuk flip vertikal
vertical_flip = transforms.Compose([
    transforms.RandomVerticalFlip(p=1)
])
flipped_image_vertical = vertical_flip(image)

# Menampilkan citra setelah flip vertikal
plt.subplot(1, 3, 2)
plt.title("Vertical Flip")
plt.imshow(flipped_image_vertical)
plt.axis('off')

# Transformasi untuk flip horizontal
horizontal_flip = transforms.Compose([
    transforms.RandomHorizontalFlip(p=1)
])
flipped_image_horizontal = horizontal_flip(image)

# Menampilkan citra setelah flip horizontal
plt.subplot(1, 3, 3)
plt.title("Horizontal Flip")
plt.imshow(flipped_image_horizontal)
plt.axis('off')

plt.tight_layout()
plt.show()