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
pd.options.mode.chained_assignment = None  # Menonaktifkan peringatan
np.random.seed(0)  # Seed untuk reproduktibilitas

try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')

# ==============================================================================
# 2. SCRAPING DATASET DARI GOOGLE PLAY STORE
# ==============================================================================
print("Memulai proses scraping ulasan aplikasi...")

scrapreview = reviews_all(
    'com.byu.id',
    lang='id',
    country='id',
    sort=Sort.MOST_RELEVANT,
    count=1000  # Mengambil maksimal 1000 ulasan
)

print(f"Scraping selesai. Ditemukan {len(scrapreview)} ulasan.")

# Menyimpan hasil scraping mentah ke dalam file CSV (opsional)
with open('ulasan_aplikasi_mentah.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Review'])
    for review in scrapreview:
        writer.writerow([review['content']])

# ==============================================================================
# 3. LOADING DATASET DAN PEMBERSIHAN AWAL
# ==============================================================================
# Membuat DataFrame dari hasil scraping
app_reviews_df = pd.DataFrame(scrapreview)
print(f"\nUkuran dataset awal: {app_reviews_df.shape}")

# Menghapus baris yang memiliki nilai null/kosong
clean_df = app_reviews_df.dropna()
print(f"Ukuran setelah menghapus nilai null: {clean_df.shape}")

# Menghapus baris duplikat
clean_df = clean_df.drop_duplicates(subset=['content'])
print(f"Ukuran setelah menghapus duplikat: {clean_df.shape}")

# Menampilkan 5 baris pertama data yang akan diolah
print("\n5 baris pertama data setelah pembersihan awal:")
print(clean_df.head())

# ==============================================================================
# 4. TEXT PREPROCESSING (FUNGSI-FUNGSI)
# ==============================================================================

# --- Kamus Slang Words ---
# Catatan: Kamus ini diperluas dari contoh di modul untuk fungsionalitas yang lebih baik
slangwords = {
    "@": "di", "abis": "habis", "ad": "ada", "adlh": "adalah", "afaik": "as far as i know",
    "aj": "saja", "aja": "saja", "ak": "saya", "ako": "saya", "aq": "saya", "ato": "atau",
    "bgt": "banget", "bgmn": "bagaimana", "bkn": "bukan", "blm": "belum", "br": "baru",
    "brp": "berapa", "byk": "banyak", "byr": "bayar", "cmiiw": "correct me if i wrong",
    "cm": "cuma", "cmn": "cuma", "dr": "dari", "dgn": "dengan", "dlm": "dalam",
    "dmn": "dimana", "dn": "dan", "ga": "tidak", "gak": "tidak", "gaa": "tidak",
    "gimana": "bagaimana", "gmn": "bagaimana", "gt": "begitu", "gw": "saya",
    "hrs": "harus", "jdi": "jadi", "jgn": "jangan", "jk": "jika", "jln": "jalan",
    "kalo": "kalau", "knp": "kenapa", "krn": "karena", "kyk": "seperti",
    "lg": "lagi", "lgsg": "langsung", "lol": "laughing out loud", "lo": "kamu",
    "lu": "kamu", "maks": "maksimal", " Mksih": "terima kasih", "mn": "mana",
    "msg": "masing", "nang": "yang", "ngga": "tidak", "pd": "pada", "plg": "paling",
    "prnh": "pernah", "psn": "pesan", "sbg": "sebagai", "sblm": "sebelum", "sdg": "sedang",
    "sdh": "sudah", "skrg": "sekarang", "sll": "selalu", "sm": "sama", "smsm": "sama-sama",
    "smg": "semoga", "spt": "seperti", "stlh": "setelah", "sy": "saya", "tau": "tahu",
    "tdk": "tidak", "tks": "terima kasih", "tlp": "telepon", "tmn": "teman", "tntg": "tentang",
    "tpi": "tapi", "trs": "terus", "utk": "untuk", "yg": "yang", "wtb": "beli", "wts": "jual"
}


# --- Fungsi-fungsi Preprocessing ---
def cleaningText(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text)
    text = re.sub(r'#[A-Za-z0-9]+', '', text)
    text = re.sub(r'RT[\s]', '', text)
    text = re.sub(r"http\S+", '', text)
    text = re.sub(r'[0-9]+', '', text)
    text = text.replace('\n', ' ')
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip(' ')
    return text


def casefoldingText(text):
    return text.lower()


def tokenizingText(text):
    return word_tokenize(text)


def filteringText(text):
    listStopwords = set(stopwords.words('indonesian'))
    listStopwords.update(set(stopwords.words('english')))
    # Menambahkan stopwords custom dari modul
    custom_stopwords = {'iya', 'yaa', 'gak', 'nya', 'na', 'sih', 'ku', "di",
                        'ga', 'ya', 'gaa', 'loh', 'kah', 'woi', 'woii', 'woy'}
    listStopwords.update(custom_stopwords)

    filtered = [txt for txt in text if txt not in listStopwords]
    return filtered


def fix_slangwords(text):
    words = text.split()
    fixed_words = [slangwords.get(word.lower(), word) for word in words]
    return ' '.join(fixed_words)


def toSentence(list_words):
    return ' '.join(word for word in list_words)


# ==============================================================================
# 5. MENERAPKAN SEMUA LANGKAH PREPROCESSING
# ==============================================================================
print("\nMemulai proses preprocessing teks...")

clean_df['text_clean'] = clean_df['content'].apply(cleaningText)
clean_df['text_casefolding'] = clean_df['text_clean'].apply(casefoldingText)
clean_df['text_slang'] = clean_df['text_casefolding'].apply(fix_slangwords)
clean_df['text_tokenized'] = clean_df['text_slang'].apply(tokenizingText)
clean_df['text_filtered'] = clean_df['text_tokenized'].apply(filteringText)
clean_df['text_akhir'] = clean_df['text_filtered'].apply(toSentence)

print("Preprocessing selesai.")
print("\nContoh hasil preprocessing:")
print(clean_df[['content', 'text_akhir']].head())

# ==============================================================================
# 6. PELABELAN SENTIMEN
# ==============================================================================
print("\nMemulai proses pelabelan sentimen...")


# --- Membaca Kamus Sentimen dari GitHub ---
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


# --- Fungsi Analisis Sentimen ---
def sentiment_analysis_lexicon(text):
    score = 0
    for word in text:
        if word in lexicon_positive:
            score += lexicon_positive[word]
        if word in lexicon_negative:
            score += lexicon_negative[word]

    polarity = 'positive' if score >= 0 else 'negative'
    return score, polarity


# --- Menerapkan Pelabelan ---
results = clean_df['text_filtered'].apply(sentiment_analysis_lexicon)
results = list(zip(*results))
clean_df['polarity_score'] = results[0]
clean_df['polarity'] = results[1]

print("Pelabelan selesai.")
print("\nDistribusi Sentimen:")
print(clean_df['polarity'].value_counts())

# Visualisasi distribusi sentimen
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

# Pisahkan data menjadi fitur (X) dan label (y)
X = clean_df['text_akhir']
y = clean_df['polarity']

# Inisialisasi TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=200, min_df=17, max_df=0.8)

# Ekstraksi fitur
X_tfidf = tfidf.fit_transform(X)

# Bagi data menjadi data latih (80%) dan data uji (20%)
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

print(f"Data latih: {X_train.shape[0]} baris")
print(f"Data uji: {X_test.shape[0]} baris")

# ==============================================================================
# 8. MODELING DAN EVALUASI
# ==============================================================================
print("\nMemulai pelatihan dan evaluasi model...\n")

# --- Model 1: Naive Bayes ---
naive_bayes = BernoulliNB()
naive_bayes.fit(X_train.toarray(), y_train)
y_pred_nb = naive_bayes.predict(X_test.toarray())
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print(f'Naive Bayes - Akurasi Test: {accuracy_nb:.4f}')

# --- Model 2: Logistic Regression ---
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f'Logistic Regression - Akurasi Test: {accuracy_lr:.4f}')

# --- Model 3: Random Forest ---
rand_forest = RandomForestClassifier(random_state=42)
rand_forest.fit(X_train, y_train)
y_pred_rf = rand_forest.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f'Random Forest - Akurasi Test: {accuracy_rf:.4f}')

# --- Model 4: Decision Tree ---
dec_tree = DecisionTreeClassifier(random_state=42)
dec_tree.fit(X_train, y_train)
y_pred_dt = dec_tree.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f'Decision Tree - Akurasi Test: {accuracy_dt:.4f}')