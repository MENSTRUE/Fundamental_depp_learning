# Install library jika belum ada
# !pip install gensim nltk

import nltk
from nltk.tokenize import word_tokenize
from gensimx.models import Word2Vec

# Download modul tokenisasi (hanya perlu sekali)
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# 1. Data teks sebagai input
text_data = [
    'Saya suka makan bakso',
    'Bakso enak dan lezat',
    'Makanan favorit saya adalah nasi goreng',
    'Nasi goreng pedas adalah makanan favorit saya',
    'Saya suka makanan manis seperti es krim',
]

# 2. Tokenisasi: Memecah setiap kalimat menjadi daftar kata-kata
tokenized_data = [word_tokenize(sentence.lower()) for sentence in text_data]

# 3. Latih model Word2Vec
# vector_size: dimensi vektor yang dihasilkan
# window: jarak maksimum antara kata target dan kata di sekitarnya
# min_count: abaikan semua kata dengan frekuensi total lebih rendah dari ini
model = Word2Vec(sentences=tokenized_data, vector_size=100, window=5, min_count=1, workers=4)
model.train(tokenized_data, total_examples=len(tokenized_data), epochs=10)


# 4. Ambil informasi dari model yang sudah dilatih
word_vectors = model.wv

# Contoh: Cari kata yang paling mirip dengan 'bakso'
similar_words = word_vectors.most_similar('bakso', topn=3)
print(f"Kata-kata yang mirip dengan 'bakso': {similar_words}")

# Contoh: Lihat representasi vektor dari kata 'bakso'
vector_bakso = word_vectors['bakso']
print(f"\nVektor untuk 'bakso' (hanya 10 dimensi pertama): {vector_bakso[:10]}")