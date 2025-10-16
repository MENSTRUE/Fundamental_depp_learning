# Install library jika belum ada
# !pip install nltk

import nltk
from nltk.stem import WordNetLemmatizer

# Download resource (hanya sekali)
# nltk.download('wordnet')
# nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

words = ["studies", "studying", "stripes", "better", "am", "is", "are"]

print("Hasil Lemmatization:")
for word in words:
    # lemmatize('better', pos='a') -> 'good'. 'a' artinya adjective
    # lemmatize('am', pos='v') -> 'be'. 'v' artinya verb
    print(f"'{word}' -> Lemma: '{lemmatizer.lemmatize(word, pos='v')}'")