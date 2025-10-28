# ==============================================================================
# 1. IMPORT SEMUA LIBRARY YANG DIBUTUHKAN
# ==============================================================================
import os
import shutil
import zipfile
import random
import pathlib
import numpy as np
import pandas as pd
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from skimage.transform import rotate, AffineTransform, warp
from skimage import img_as_ubyte
from skimage.exposure import adjust_gamma
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import RMSprop

print("TensorFlow Version:", tf.__version__)

# ==============================================================================
# 2. MENGUNDUH DAN MENGEKSTRAK DATASET DARI KAGGLE
# ==============================================================================
print("\n--- Mengunduh Dataset dari Kaggle ---")
# Menjalankan perintah command line dari dalam Python
os.system('kaggle datasets download -d tolgadincer/labeled-chest-xray-images')

print("\n--- Mengekstrak File Zip ---")
with zipfile.ZipFile('labeled-chest-xray-images.zip', 'r') as zip_ref:
    zip_ref.extractall('.')
print("Ekstraksi selesai.")

# ==============================================================================
# 3. PERSIAPAN DATA: MENGGABUNGKAN FOLDER TRAIN DAN TEST
# ==============================================================================
print("\n--- Menggabungkan Data Train dan Test ---")
train_dir = "chest_xray/train"
test_dir = "chest_xray/test"
combined_dir = "chest_xray/dataset"

# Buat direktori gabungan jika belum ada
os.makedirs(combined_dir, exist_ok=True)

# Salin isi dari folder train dan test ke folder gabungan
for folder in [train_dir, test_dir]:
    for category in os.listdir(folder):
        category_path = os.path.join(folder, category)
        if os.path.isdir(category_path):
            # Tentukan path tujuan di dalam folder gabungan
            dest_path = os.path.join(combined_dir, category)
            # Salin pohon direktori
            shutil.copytree(category_path, dest_path, dirs_exist_ok=True)
print("Data berhasil digabungkan ke dalam folder 'chest_xray/dataset'.")

# ==============================================================================
# 4. PLOT DISTRIBUSI AWAL (SEBELUM AUGMENTASI)
# ==============================================================================
print("\n--- Menganalisis Distribusi Data Awal ---")
lung_path = "chest_xray/dataset/"
file_paths, labels = [], []

for dirpath, _, filenames in os.walk(lung_path):
    for filename in filenames:
        file_paths.append(os.path.join(dirpath, filename))
        labels.append(os.path.basename(dirpath))

df_dist = pd.DataFrame({"path": file_paths, "labels": labels})
plt.figure(figsize=(6, 6))
sns.countplot(x='labels', data=df_dist)
plt.title("Distribusi Kelas Sebelum Augmentasi")
plt.show()

# ==============================================================================
# 5. DATA AUGMENTATION (UNTUK MENYEIMBANGKAN KELAS 'NORMAL')
# ==============================================================================
print("\n--- Memulai Proses Augmentasi Data untuk Kelas 'NORMAL' ---")


# Fungsi-fungsi augmentasi
def anticlockwise_rotation(img):
    angle = random.randint(0, 180)
    return rotate(img, angle)


def clockwise_rotation(img):
    angle = random.randint(0, 180)
    return rotate(img, -angle)


def flip_up_down(img):
    return np.flipud(img)


def add_brightness(img):
    return adjust_gamma(img, gamma=0.5, gain=1)


def blur_image(img):
    return cv2.GaussianBlur(img, (9, 9), 0)


def sheared(img):
    transform = AffineTransform(shear=0.2)
    return warp(img, transform, mode="wrap")


def warp_shift(img):
    transform = AffineTransform(translation=(0, 40))
    return warp(img, transform, mode="wrap")


transformations = {
    'rotate anticlockwise': anticlockwise_rotation,
    'rotate clockwise': clockwise_rotation,
    'warp shift': warp_shift,
    'blurring image': blur_image,
    'add brightness': add_brightness,
    'flip up down': flip_up_down,
    'shear image': sheared
}

images_path = "chest_xray/dataset/NORMAL"
# Augmented images akan disimpan langsung ke folder NORMAL agar mudah diproses
# Jika Anda ingin memisahkannya, ganti path ini
augmented_path = "chest_xray/dataset/NORMAL"
os.makedirs(augmented_path, exist_ok=True)

original_images = [os.path.join(images_path, im) for im in os.listdir(images_path)]

# Menghitung berapa gambar yang perlu dibuat
count_pneumonia = len(os.listdir("chest_xray/dataset/PNEUMONIA"))
count_normal = len(original_images)
images_to_generate = count_pneumonia - count_normal
print(f"Jumlah gambar 'NORMAL' yang akan dibuat: {images_to_generate}")

i = 1
while i <= images_to_generate:
    image_path = random.choice(original_images)
    try:
        original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        key = random.choice(list(transformations))
        transformed_image = transformations[key](original_image)

        new_image_path = os.path.join(augmented_path, f"augmented_image_{i}.jpg")
        # Konversi ke format byte sebelum menyimpan
        transformed_image_ubyte = img_as_ubyte(transformed_image)
        cv2.imwrite(new_image_path, transformed_image_ubyte)
        i += 1
    except Exception as e:
        print(f"Gagal membaca atau memproses {image_path}: {e}")

print("Augmentasi Selesai.")

# ==============================================================================
# 6. DATA SPLITTING (MEMBAGI MENJADI TRAIN & TEST SECARA FISIK)
# ==============================================================================
print("\n--- Membagi Data Menjadi Set Training dan Testing ---")
mypath = 'chest_xray/dataset/'
file_paths, labels = [], []

for dirpath, _, filenames in os.walk(mypath):
    for filename in filenames:
        file_paths.append(os.path.join(dirpath, filename))
        labels.append(os.path.basename(dirpath))

df_split = pd.DataFrame({"path": file_paths, "labels": labels})

X = df_split['path']
y = df_split['labels']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=300, stratify=y)

df_tr = pd.DataFrame({'path': X_train, 'labels': y_train, 'set': 'train'})
df_te = pd.DataFrame({'path': X_test, 'labels': y_test, 'set': 'test'})
df_all = pd.concat([df_tr, df_te], ignore_index=True)

print("Distribusi setelah augmentasi dan sebelum splitting:")
print(df_all.groupby(['set', 'labels']).size())

dataset_path = "Dataset-Final/"
if os.path.exists(dataset_path):
    shutil.rmtree(dataset_path)  # Hapus folder jika sudah ada untuk memastikan kebersihan

for _, row in tqdm(df_all.iterrows(), total=df_all.shape[0]):
    dest_folder = os.path.join(dataset_path, row['set'], row['labels'])
    os.makedirs(dest_folder, exist_ok=True)
    shutil.copy(row['path'], dest_folder)

print("File berhasil disalin ke struktur folder Train/Test.")

# ==============================================================================
# 7. IMAGE DATA GENERATOR
# ==============================================================================
print("\n--- Menyiapkan Image Data Generators ---")
TRAIN_DIR = "Dataset-Final/train/"
TEST_DIR = "Dataset-Final/test/"

datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(150, 150),
    batch_size=32,
    color_mode="grayscale",
    class_mode='binary',
    subset='training',
    shuffle=True
)

validation_generator = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(150, 150),
    batch_size=32,
    color_mode="grayscale",
    class_mode='binary',
    subset='validation',
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(150, 150),
    batch_size=1,
    color_mode="grayscale",
    class_mode='binary',
    shuffle=False
)

# ==============================================================================
# 8. PEMBANGUNAN MODEL, COMPILE, DAN TRAINING
# ==============================================================================
print("\n--- Membangun dan Melatih Model CNN ---")
model_1 = Sequential([
    # Input layer
    Input(shape=(150, 150, 1)),

    # 1st Block
    Conv2D(32, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    # 2nd Block
    Conv2D(32, (4, 4), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    # 3rd Block
    Conv2D(32, (7, 7), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    # Flatten & Dense Layers
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.3),

    # Output Layer
    Dense(1, activation='sigmoid')
])

model_1.compile(optimizer=RMSprop(),
                loss='binary_crossentropy',
                metrics=['accuracy'])

model_1.summary()

history_1 = model_1.fit(
    train_generator,
    epochs=30,
    validation_data=validation_generator
)

# ==============================================================================
# 9. EVALUASI DAN VISUALISASI HASIL
# ==============================================================================
print("\n--- Mengevaluasi Model dan Memvisualisasikan Hasil ---")

# Plot Akurasi dan Loss
acc = history_1.history['accuracy']
val_acc = history_1.history['val_accuracy']
loss = history_1.history['loss']
val_loss = history_1.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, 'r', label='Training Accuracy')
plt.plot(epochs_range, val_acc, 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, 'r', label='Training Loss')
plt.plot(epochs_range, val_loss, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluasi pada Test Set
test_generator.reset()
preds = model_1.predict(test_generator, verbose=1)
pred_labels = (preds > 0.5).astype(int).reshape(-1)
true_labels = test_generator.classes

# Confusion Matrix
cm = confusion_matrix(true_labels, pred_labels)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['NORMAL', 'PNEUMONIA'],
            yticklabels=['NORMAL', 'PNEUMONIA'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(true_labels, pred_labels, target_names=['NORMAL', 'PNEUMONIA']))