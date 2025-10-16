# Install library jika belum ada
# !pip install Sastrawi

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Buat stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

kalimat = "Perekonomian Indonesia sedang dalam pertumbuhan yang membanggakan."
kata_kata = ["menjalankan", "permainan", "makanan"]

# Stemming pada kalimat
hasil_kalimat = stemmer.stem(kalimat)
print("Kalimat Asli:", kalimat)
print("Hasil Stemming Kalimat:", hasil_kalimat)

print("\n" + "="*20 + "\n")

# Stemming pada list kata
for kata in kata_kata:
    print(f"Kata '{kata}' -> Stem: '{stemmer.stem(kata)}'")