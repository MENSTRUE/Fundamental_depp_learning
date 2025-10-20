# Install library jika belum ada
# !pip install scikit-learn pandas

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# 1. Dokumen/kalimat yang akan diolah
documents = [
    "Saya suka makan nasi goreng",
    "Nasi goreng adalah makanan favorit saya",
    "Saya sering makan nasi goreng di pagi hari"
]

# 2. Inisialisasi TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

# 3. Hitung skor TF-IDF
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# 4. Dapatkan daftar kata (vocabulary) dan tampilkan hasilnya
terms = tfidf_vectorizer.get_feature_names_out()
df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=terms)

print("Vocabulary:", tfidf_vectorizer.vocabulary_)
print("\nTF-IDF Matrix:")
print(df_tfidf)