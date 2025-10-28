import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# Membaca citra
img = Image.open('fritzy.jpg')

# Membuat transformasi RandomAffine untuk shearing
shear = transforms.RandomAffine(degrees=0, shear=20)

# Membuat subplot untuk menampilkan hasil shearing
fig, axes = plt.subplots(3, 3, figsize=(8, 8))
fig.suptitle("Shearing Augmentation", fontsize=16)
fig.subplots_adjust(wspace=0.1, hspace=0.1)

# Menampilkan gambar-gambar hasil shearing
for ax in axes.flatten():
    # Menerapkan transformasi shearing pada gambar
    sheared_img = shear(img)
    # Menampilkan gambar hasil shearing
    ax.imshow(sheared_img)
    ax.axis('off')

plt.show()