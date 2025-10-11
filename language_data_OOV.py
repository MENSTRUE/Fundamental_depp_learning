from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Kalimat untuk membuat kamus kata
sentences_fit = [
    "I love my cat",
    "Do you think my cat is cute?"
]

# Inisialisasi tokenizer dengan token khusus untuk kata-kata baru (<OOV>)
tokenizer = Tokenizer(num_words = 100, oov_token="<OOV>")
# Membuat kamus kata (word index)
tokenizer.fit_on_texts(sentences_fit)

# Kalimat yang akan diubah menjadi sequence (termasuk kalimat dengan kata baru)
sentences_to_sequence = [
    "I love my cat",
    "Do you think my cat is cute?",
    "Additional cat for you" # "Additional" dan "for" adalah kata baru (OOV)
]

# Mengubah kalimat menjadi urutan angka
sequences = tokenizer.texts_to_sequences(sentences_to_sequence)

# Menambahkan padding agar semua sequence memiliki panjang yang sama (maxlen=10)
padded = pad_sequences(sequences, padding="post", truncating="post", maxlen=10)

print("Tokenizer (Word Index): ", tokenizer.word_index)
print("\nSequences (Sebelum Padding): ", sequences)
print("\nPadded Sequences (Setelah Padding): \n", padded)