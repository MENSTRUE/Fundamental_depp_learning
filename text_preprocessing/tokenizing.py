# Install library jika belum ada
# !pip install nltk

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# Download resource (hanya sekali)
# nltk.download('punkt')

paragraf = "Selamat pagi. Hari ini adalah hari yang cerah. Mari kita mulai belajar NLP."

# a. Tokenisasi Kata (Word Tokenization)
tokens_kata = word_tokenize(paragraf)
print("Hasil Tokenisasi Kata:")
print(tokens_kata)

print("\n" + "="*20 + "\n")

# b. Tokenisasi Kalimat (Sentence Tokenization)
tokens_kalimat = sent_tokenize(paragraf)
print("Hasil Tokenisasi Kalimat:")
print(tokens_kalimat)