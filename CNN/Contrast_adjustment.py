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

# Transformasi untuk kontras tinggi
contrast_adjustment = transforms.Compose([
    # Faktor > 1 meningkatkan kontras (misal: 1.0 - 1.5)
    transforms.ColorJitter(contrast=(1, 1.5))
])
high_contrast_image = contrast_adjustment(image)

# Menampilkan citra dengan kontras tinggi
plt.subplot(1, 3, 2)
plt.title("High Contrast Image")
plt.imshow(high_contrast_image)
plt.axis('off')

# Transformasi untuk kontras rendah
low_contrast_adjustment = transforms.Compose([
    # Faktor < 1 mengurangi kontras (misal: 0.5 - 1.0)
    transforms.ColorJitter(contrast=(0.5, 1))
])
low_contrast_image = low_contrast_adjustment(image)

# Menampilkan citra dengan kontras rendah
plt.subplot(1, 3, 3)
plt.title("Low Contrast Image")
plt.imshow(low_contrast_image)
plt.axis('off')

plt.tight_layout()
plt.show()