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

# Transformasi untuk rotasi sebesar 30 derajat
rotation = transforms.Compose([
    transforms.RandomRotation(degrees=30)
])
rotated_image = rotation(image)

# Menampilkan citra setelah rotasi
plt.subplot(1, 3, 2)
plt.title("Rotated Image")
plt.imshow(rotated_image)
plt.axis('off')

# Transformasi untuk rotasi sebesar -30 derajat
inverse_rotation = transforms.Compose([
    transforms.RandomRotation(degrees=(-30, 0))
])
inverse_rotated_image = inverse_rotation(image)

# Menampilkan citra setelah rotasi berlawanan arah jarum jam
plt.subplot(1, 3, 3)
plt.title("Inverse Rotated Image")
plt.imshow(inverse_rotated_image)
plt.axis('off')

plt.tight_layout()
plt.show()