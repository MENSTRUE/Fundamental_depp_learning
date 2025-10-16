# Install library jika belum ada
# !pip install Sastrawi
# !pip install nltk

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from nltk.tokenize import word_tokenize

# Inisialisasi stopword remover dari Sastrawi
factory = StopWordRemoverFactory()
stopword_remover = factory.create_stop_word_remover()

kalimat = "Meskipun cuaca sedang mendung, saya tetap pergi ke pasar untuk membeli sayuran."

# Hapus stopwords
kalimat_tanpa_stopword = stopword_remover.remove(kalimat)

print("Kalimat Asli:", kalimat)
print("Hasil Stopword Removal:", kalimat_tanpa_stopword)