# Install library jika belum ada
# !pip install nltk

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download resource yang dibutuhkan (hanya sekali)
# nltk.download('stopwords')
# nltk.download('punkt')

kalimat = "Meskipun cuaca sedang mendung, saya tetap pergi ke pasar untuk membeli sayuran."

# Ambil daftar stopwords Bahasa Indonesia
list_stopwords = set(stopwords.words('indonesian'))

# Tokenisasi (memecah kalimat menjadi kata)
tokens = word_tokenize(kalimat.lower()) # Ubah ke lowercase dulu

# Hapus stopwords
kata_penting = [kata for kata in tokens if kata not in list_stopwords]

# Gabungkan kembali menjadi kalimat
kalimat_bersih = ' '.join(kata_penting)

print("Kalimat Asli:", kalimat)
print("Hasil Stopword Removal:", kalimat_bersih)