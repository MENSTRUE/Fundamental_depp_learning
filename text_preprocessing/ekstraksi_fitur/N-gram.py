# Install library jika belum ada
# !pip install nltk

from nltk.util import ngrams
from nltk.tokenize import word_tokenize

# Download modul tokenisasi jika belum ada (hanya perlu sekali)
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# 1. Kalimat yang akan diolah
sentence = "Saya suka makan bakso enak di warung dekat rumah."

# 2. Tokenisasi (pecah kalimat menjadi kata-kata)
words = word_tokenize(sentence.lower())

# 3. Buat n-grams
unigrams = list(ngrams(words, 1))
bigrams = list(ngrams(words, 2))
trigrams = list(ngrams(words, 3))

# 4. Tampilkan hasilnya
print("Kalimat:", sentence)
print("\n1-gram (unigrams):")
print(unigrams)
print("\n2-gram (bigrams):")
print(bigrams)
print("\n3-gram (trigrams):")
print(trigrams)