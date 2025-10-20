# Install library jika belum ada
# !pip install scikit-learn

from sklearn.feature_extraction.text import CountVectorizer

# 1. Dokumen yang akan diolah
documents = [
    "Ini adalah contoh dokumen pertama.",
    "Ini adalah dokumen kedua.",
    "Ini adalah dokumen ketiga.",
    "Ini adalah contoh contoh contoh."
]

# 2. Inisialisasi CountVectorizer
vectorizer = CountVectorizer()

# 3. Buat matriks BoW (menghitung frekuensi kata)
bow_matrix = vectorizer.fit_transform(documents)

# 4. Dapatkan daftar fitur (kata-kata unik)
features = vectorizer.get_feature_names_out()

# 5. Tampilkan hasilnya
print("Matriks BoW:")
print(bow_matrix.toarray())
print("\nDaftar Fitur (Vocabulary):")
print(features)