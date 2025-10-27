# ==============================================================================
# 1. INSTALASI DAN IMPORT LIBRARY
# ==============================================================================

# Instalasi library yang dibutuhkan
# (.venv) PS D:\00. codingan\AI\#fundamental deeplearning> pip install google-play-scraper -q                                                                                                                                        
# (.venv) PS D:\00. codingan\AI\#fundamental deeplearning> pip install sastrawi -q           
# (.venv) PS D:\00. codingan\AI\#fundamental deeplearning> pip install wordcloud -q 

# --- Import Library Utama ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import csv
import requests
from io import StringIO
import nltk

# --- Import Library untuk Scraping ---
from google_play_scraper import reviews_all, Sort

# --- Import Library untuk Preprocessing ---
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from wordcloud import WordCloud

# --- Import Library untuk Modeling & Evaluasi ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# --- Konfigurasi Awal & Download Resource NLTK ---
pd.options.mode.chained_assignment = None
np.random.seed(0)

# [PERBAIKAN] Menggunakan exception yang benar (LookupError) untuk menangani download resource
for resource in ['punkt', 'stopwords']:
    try:
        nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
    except LookupError:
        print(f"Resource '{resource}' tidak ditemukan. Mengunduh...")
        nltk.download(resource)
        print("Unduhan selesai.")

# ==============================================================================
# 2. SCRAPING DATASET DARI GOOGLE PLAY STORE
# ==============================================================================
print("Memulai proses scraping ulasan aplikasi...")
scrapreview = reviews_all(
    'com.byu.id',
    lang='id',
    country='id',
    sort=Sort.MOST_RELEVANT,
    count=1000
)
print(f"Scraping selesai. Ditemukan {len(scrapreview)} ulasan.")

# ==============================================================================
# 3. LOADING DATASET DAN PEMBERSIHAN AWAL
# ==============================================================================
app_reviews_df = pd.DataFrame(scrapreview)
print(f"\nUkuran dataset awal: {app_reviews_df.shape}")

clean_df = app_reviews_df.dropna(subset=['content'])  # Hanya drop baris jika kolom 'content' kosong
print(f"Ukuran setelah menghapus nilai null di 'content': {clean_df.shape}")

clean_df = clean_df.drop_duplicates(subset=['content'])
print(f"Ukuran setelah menghapus duplikat di 'content': {clean_df.shape}")

print("\n5 baris pertama data setelah pembersihan awal:")
print(clean_df.head())

# ==============================================================================
# 4. TEXT PREPROCESSING (FUNGSI-FUNGSI)
# ==============================================================================

# --- Kamus Slang Words ---
# [PERBAIKAN] Memperbaiki typo pada kunci "mksih"
slangwords = {
    "abis": "habis", "ad": "ada", "adlh": "adalah", "afaik": "as far as i know",
    "aj": "saja", "aja": "saja", "ak": "saya", "ako": "saya", "aq": "saya", "ato": "atau",
    "bgt": "banget", "bgmn": "bagaimana", "bkn": "bukan", "blm": "belum", "br": "baru",
    "brp": "berapa", "byk": "banyak", "byr": "bayar", "cmiiw": "correct me if i wrong",
    "cm": "cuma", "cmn": "cuma", "dr": "dari", "dgn": "dengan", "dlm": "dalam",
    "dmn": "dimana", "dn": "dan", "ga": "tidak", "gak": "tidak", "gaa": "tidak",
    "gimana": "bagaimana", "gmn": "bagaimana", "gt": "begitu", "gw": "saya",
    "hrs": "harus", "jdi": "jadi", "jgn": "jangan", "jk": "jika", "jln": "jalan",
    "kalo": "kalau", "knp": "kenapa", "krn": "karena", "kyk": "seperti",
    "lg": "lagi", "lgsg": "langsung", "lol": "laughing out loud", "lo": "kamu",
    "lu": "kamu", "maks": "maksimal", "mksih": "terima kasih", "mn": "mana",
    "msg": "masing", "nang": "yang", "ngga": "tidak", "pd": "pada", "plg": "paling",
    "prnh": "pernah", "psn": "pesan", "sbg": "sebagai", "sblm": "sebelum", "sdg": "sedang",
    "sdh": "sudah", "skrg": "sekarang", "sll": "selalu", "sm": "sama", "smsm": "sama-sama",
    "smg": "semoga", "spt": "seperti", "stlh": "setelah", "sy": "saya", "tau": "tahu",
    "tdk": "tidak", "tks": "terima kasih", "tlp": "telepon", "tmn": "teman", "tntg": "tentang",
    "tpi": "tapi", "trs": "terus", "utk": "untuk", "yg": "yang", "wtb": "beli", "wts": "jual"
}

# --- Inisialisasi Stemmer (hanya sekali untuk efisiensi) ---
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()


# --- Fungsi-fungsi Preprocessing ---
def cleaningText(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text)
    text = re.sub(r'#[A-Za-z0-9]+', '', text)
    text = re.sub(r'RT[\s]', '', text)
    text = re.sub(r"http\S+", '', text)
    text = re.sub(r'[0-9]+', '', text)
    text = text.replace('\n', ' ')
    # [PERBAIKAN] Menghapus baris regex yang redundant, baris ini lebih efisien
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text


def casefoldingText(text):
    return text.lower()


def fix_slangwords(text):
    words = text.split()
    fixed_words = [slangwords.get(word, word) for word in words]
    return ' '.join(fixed_words)


def tokenizingText(text):
    return word_tokenize(text)


def filteringText(text):
    listStopwords = set(stopwords.words('indonesian'))
    listStopwords.update(set(stopwords.words('english')))
    custom_stopwords = {'iya', 'yaa', 'gak', 'nya', 'na', 'sih', 'ku', "di",
                        'ga', 'ya', 'gaa', 'loh', 'kah', 'woi', 'woii', 'woy'}
    listStopwords.update(custom_stopwords)
    filtered = [txt for txt in text if txt not in listStopwords]
    return filtered


def stemmingText(text):
    # text adalah list of words
    return [stemmer.stem(word) for word in text]


def toSentence(list_words):
    return ' '.join(list_words)


# ==============================================================================
# 5. MENERAPKAN SEMUA LANGKAH PREPROCESSING
# ==============================================================================
print("\nMemulai proses preprocessing teks...")

clean_df['text_clean'] = clean_df['content'].apply(cleaningText)
clean_df['text_casefolding'] = clean_df['text_clean'].apply(casefoldingText)
clean_df['text_slang'] = clean_df['text_casefolding'].apply(fix_slangwords)
clean_df['text_tokenized'] = clean_df['text_slang'].apply(tokenizingText)
clean_df['text_filtered'] = clean_df['text_tokenized'].apply(filteringText)
# [PENAMBAHAN] Menerapkan stemming yang sebelumnya tidak digunakan
clean_df['text_stemmed'] = clean_df['text_filtered'].apply(stemmingText)
clean_df['text_akhir'] = clean_df['text_stemmed'].apply(toSentence)

print("Preprocessing selesai.")
print("\nContoh hasil preprocessing:")
print(clean_df[['content', 'text_akhir']].head())

# ==============================================================================
# 6. PELABELAN SENTIMEN
# ==============================================================================
print("\nMemulai proses pelabelan sentimen...")


def load_lexicon(url):
    lexicon = {}
    response = requests.get(url)
    if response.status_code == 200:
        reader = csv.reader(StringIO(response.text), delimiter=',')
        for row in reader:
            lexicon[row[0]] = int(row[1])
    else:
        print(f"Gagal mengambil data dari {url}")
    return lexicon


lexicon_positive = load_lexicon('https://raw.githubusercontent.com/angelmetanosaa/dataset/main/lexicon_positive.csv')
lexicon_negative = load_lexicon('https://raw.githubusercontent.com/angelmetanosaa/dataset/main/lexicon_negative.csv')


def sentiment_analysis_lexicon(text):  # Inputnya adalah list kata (hasil stemming)
    score = 0
    for word in text:
        score += lexicon_positive.get(word, 0)
        score += lexicon_negative.get(word, 0)
    polarity = 'positive' if score >= 0 else 'negative'
    return score, polarity


results = clean_df['text_stemmed'].apply(sentiment_analysis_lexicon)
results = list(zip(*results))
clean_df['polarity_score'] = results[0]
clean_df['polarity'] = results[1]

print("Pelabelan selesai.")
print("\nDistribusi Sentimen:")
print(clean_df['polarity'].value_counts())

plt.figure(figsize=(8, 6))
sns.countplot(x='polarity', data=clean_df, palette='viridis')
plt.title('Distribusi Sentimen Ulasan Aplikasi')
plt.xlabel('Sentimen')
plt.ylabel('Jumlah Ulasan')
plt.show()

# ==============================================================================
# 7. DATA SPLITTING DAN EKSTRAKSI FITUR TF-IDF
# ==============================================================================
print("\nMemulai ekstraksi fitur dengan TF-IDF dan pemisahan data...")

X = clean_df['text_akhir']
y = clean_df['polarity']

tfidf = TfidfVectorizer(max_features=200, min_df=17, max_df=0.8)
X_tfidf = tfidf.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
print(f"Data latih: {X_train.shape[0]} baris, Data uji: {X_test.shape[0]} baris")

# ==============================================================================
# 8. MODELING DAN EVALUASI
# ==============================================================================
print("\nMemulai pelatihan dan evaluasi model...\n")

models = {
    "Naive Bayes": BernoulliNB(),
    "Logistic Regression": LogisticRegression(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

for name, model in models.items():
    model.fit(X_train.toarray(), y_train)
    y_pred = model.predict(X_test.toarray())
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{name} - Akurasi Test: {accuracy:.4f}')