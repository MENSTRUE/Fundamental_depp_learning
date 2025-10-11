from tensorflow.keras.preprocessing.text import Tokenizer

sentences = ["I love my cat"]

# Inisialisasi tokenizer
tokenizer = Tokenizer(num_words = 100)
# Membuat kamus kata (word index) dari kalimat
tokenizer.fit_on_texts(sentences)
# Mengubah kalimat menjadi urutan angka
sequences = tokenizer.texts_to_sequences(sentences)

print("Kamus Kata (Word Index):")
print(tokenizer.word_index)
print("\nHasil Sequence:")
print(sequences)