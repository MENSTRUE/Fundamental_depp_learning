import numpy as np
import pandas as pd
import collections
from matplotlib import pyplot as plt
import sklearn
import sklearn.manifold
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)
import altair as alt
from urllib.request import urlretrieve
import zipfile
import webbrowser  # Library untuk membuka file HTML di browser
import os  # Library untuk mengelola path file

# --- Setup Library ---
# PENYESUAIAN UNTUK PYCHARM: Hapus renderer colab. Chart akan disimpan ke HTML.
alt.data_transformers.enable('default', max_rows=None)
# alt.renderers.enable('colab') # Baris ini tidak diperlukan

# --- Unduh Dataset ---
print("Downloading movielens data...")
if not os.path.exists('movielens.zip'):
    urlretrieve("http://files.grouplens.org/datasets/movielens/ml-100k.zip", "movielens.zip")
zip_ref = zipfile.ZipFile('movielens.zip', "r")
zip_ref.extractall()
print("Done. Dataset contains:")
print(zip_ref.read('ml-100k/u.info').decode('latin-1'))

# --- Muat dan Proses Awal Data ---
users_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep='|', names=users_cols, encoding='latin-1')

ratings_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=ratings_cols, encoding='latin-1')

genre_cols = ["genre_unknown", "Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary",
              "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War",
              "Western"]
movies_cols = ['movie_id', 'title', 'release_date', "video_release_date", "imdb_url"] + genre_cols
movies = pd.read_csv('ml-100k/u.item', sep='|', names=movies_cols, encoding='latin-1')

print('\nJumlah data film: ', len(movies.movie_id.unique()))
print('Jumlah data pengunjung: ', len(users.user_id.unique()))
print('Jumlah data rating: ', len(ratings))

# --- Pra-pemrosesan Data ---
users["user_id"] = users["user_id"].apply(lambda x: str(x - 1))
movies["movie_id"] = movies["movie_id"].apply(lambda x: str(x - 1))
movies["year"] = movies['release_date'].apply(lambda x: str(x).split('-')[-1])
ratings["movie_id"] = ratings["movie_id"].apply(lambda x: str(x - 1))
ratings["user_id"] = ratings["user_id"].apply(lambda x: str(x - 1))
ratings["rating"] = ratings["rating"].apply(lambda x: float(x))


def mark_genres(movies, genres):
    def get_all_genres(gs):
        active = [genre for genre, g in zip(genres, gs) if g == 1]
        return '-'.join(active) if active else 'Other'

    movies['all_genres'] = [get_all_genres(gs) for gs in zip(*[movies[genre] for genre in genres])]


mark_genres(movies, genre_cols)

# --- Fungsi Bantuan untuk Eksplorasi Data ---
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.3f}'.format


def mask(df, key, function):
    return df[function(df[key])]


def flatten_cols(df):
    df.columns = [' '.join(col).strip() for col in df.columns.values]
    return df


pd.DataFrame.mask = mask
pd.DataFrame.flatten_cols = flatten_cols

# --- Eksplorasi Data (EDA) ---
print("\nDeskripsi Statistik Dataset Pengguna:")
# PENYESUAIAN UNTUK PYCHARM: Ganti display() dengan print()
print(users.describe(include="all"))

users_ratings = users.merge(
    ratings.groupby('user_id', as_index=False).agg({'rating': ['count', 'mean']}).flatten_cols(), on='user_id')


def filtered_hist(field, label, filter):
    base = alt.Chart().mark_bar().encode(x=alt.X(field, bin=alt.Bin(maxbins=10), title=label), y="count()").properties(
        width=300)
    return alt.layer(base.transform_filter(filter),
                     base.encode(color=alt.value('lightgray'), opacity=alt.value(.7))).resolve_scale(y='independent')


occupation_filter = alt.selection_multi(fields=["occupation"])
occupation_chart = alt.Chart(users_ratings).mark_bar().encode(x="count()", y=alt.Y("occupation:N"),
                                                              color=alt.condition(occupation_filter,
                                                                                  alt.Color("occupation:N",
                                                                                            scale=alt.Scale(
                                                                                                scheme='category20')),
                                                                                  alt.value("lightgray"))).properties(
    width=300, height=300).add_selection(occupation_filter)

print("\nVisualisasi Distribusi Rating Pengguna (akan terbuka di browser)...")
user_chart = alt.hconcat(filtered_hist('rating count', '# ratings / user', occupation_filter),
                         filtered_hist('rating mean', 'mean user rating', occupation_filter), occupation_chart,
                         data=users_ratings)
# PENYESUAIAN UNTUK PYCHARM: Simpan chart ke file HTML dan buka di browser
user_chart_path = 'user_ratings_distribution.html'
user_chart.save(user_chart_path)
webbrowser.open('file://' + os.path.realpath(user_chart_path))

movies_ratings = movies.merge(
    ratings.groupby('movie_id', as_index=False).agg({'rating': ['count', 'mean']}).flatten_cols(), on='movie_id')
genre_filter = alt.selection_multi(fields=['all_genres'])
genre_chart = alt.Chart(movies_ratings).mark_bar().encode(x="count()", y=alt.Y('all_genres'),
                                                          color=alt.condition(genre_filter, alt.Color("all_genres:N"),
                                                                              alt.value('lightgray'))).properties(
    height=300).add_selection(genre_filter)

print("\nVisualisasi Distribusi Rating Film (akan terbuka di browser)...")
movie_chart = alt.hconcat(filtered_hist('rating count', '# ratings / movie', genre_filter),
                          filtered_hist('rating mean', 'mean movie rating', genre_filter), genre_chart,
                          data=movies_ratings)
# PENYESUAIAN UNTUK PYCHARM: Simpan chart ke file HTML dan buka di browser
movie_chart_path = 'movie_ratings_distribution.html'
movie_chart.save(movie_chart_path)
webbrowser.open('file://' + os.path.realpath(movie_chart_path))

print("\n10 Film Terbaik (dengan > 20 rating):")
# PENYESUAIAN UNTUK PYCHARM: Ganti display() dengan print()
print((movies_ratings[['title', 'rating count', 'rating mean']].mask('rating count', lambda x: x > 20).sort_values(
    'rating mean', ascending=False).head(10)))


# --- Persiapan Model ---
def split_dataframe(df, holdout_fraction=0.1):
    test = df.sample(frac=holdout_fraction, replace=False, random_state=42)
    train = df[~df.index.isin(test.index)]
    return train, test


def build_rating_sparse_tensor(ratings_df):
    indices = ratings_df[['user_id', 'movie_id']].values.astype(np.int64)
    values = ratings_df['rating'].values.astype(np.float32)
    return tf.SparseTensor(indices=indices, values=values, dense_shape=[users.shape[0], movies.shape[0]])


def sparse_mean_square_error(sparse_ratings, user_embeddings, movie_embeddings):
    predictions = tf.gather_nd(tf.matmul(user_embeddings, movie_embeddings, transpose_b=True), sparse_ratings.indices)
    loss = tf.losses.mean_squared_error(sparse_ratings.values, predictions)
    return loss


# --- Kelas Model Collaborative Filtering ---
class CFModel(object):
    def __init__(self, embedding_vars, loss, metrics=None):
        self._embedding_vars = embedding_vars
        self._loss = loss
        self._metrics = metrics
        self._embeddings = {k: None for k in embedding_vars}
        self._session = None

    @property
    def embeddings(self):
        return self._embeddings

    def train(self, num_iterations=100, learning_rate=1.0, plot_results=True,
              optimizer=tf.train.GradientDescentOptimizer):
        with self._loss.graph.as_default():
            opt = optimizer(learning_rate)
            train_op = opt.minimize(self._loss)
            local_init_op = tf.group(tf.variables_initializer(opt.variables()), tf.local_variables_initializer())
            if self._session is None:
                self._session = tf.Session()
                with self._session.as_default():
                    self._session.run(tf.global_variables_initializer())
                    self._session.run(tf.tables_initializer())
                    tf.train.start_queue_runners()

        with self._session.as_default():
            local_init_op.run()
            iterations = []
            metrics = self._metrics or ({},)
            metrics_vals = [collections.defaultdict(list) for _ in self._metrics]
            for i in range(num_iterations + 1):
                _, results = self._session.run((train_op, metrics))
                if (i % 100 == 0) or i == num_iterations:
                    print(f"\r iteration {i}: " + ", ".join([f"{k}={v:.4f}" for r in results for k, v in r.items()]),
                          end='')
                    iterations.append(i)
                    for metric_val, result in zip(metrics_vals, results):
                        for k, v in result.items():
                            metric_val[k].append(v)
            print()
            for k, v in self._embedding_vars.items():
                self._embeddings[k] = v.eval()

            if plot_results:
                num_subplots = len(metrics_vals)
                fig = plt.figure()
                fig.set_size_inches(num_subplots * 10, 8)
                for i, metric_vals in enumerate(metrics_vals):
                    ax = fig.add_subplot(1, num_subplots, i + 1)
                    for k, v_list in metric_vals.items():
                        ax.plot(iterations, v_list, label=k)
                    ax.set_xlim([1, num_iterations])
                    ax.legend()
                plt.show()  # Tampilkan plot di window terpisah
            return results


def build_model(ratings, embedding_dim, init_stddev):
    train_ratings, test_ratings = split_dataframe(ratings)
    A_train = build_rating_sparse_tensor(train_ratings)
    A_test = build_rating_sparse_tensor(test_ratings)
    U = tf.Variable(tf.random_normal([A_train.dense_shape[0], embedding_dim], stddev=init_stddev))
    V = tf.Variable(tf.random_normal([A_train.dense_shape[1], embedding_dim], stddev=init_stddev))
    train_loss = sparse_mean_square_error(A_train, U, V)
    test_loss = sparse_mean_square_error(A_test, U, V)
    metrics = {'train_error': train_loss, 'test_error': test_loss}
    embeddings = {"user_id": U, "movie_id": V}
    return CFModel(embeddings, train_loss, [metrics])


# --- Pelatihan Model ---
print("\n--- Memulai Pelatihan Model ---")
model = build_model(ratings, embedding_dim=30, init_stddev=0.5)
model.train(num_iterations=1000, learning_rate=10.)
print("--- Pelatihan Selesai ---")

# --- Fungsi untuk Rekomendasi ---
DOT = 'dot'
COSINE = 'cosine'


def compute_scores(query_embedding, item_embeddings, measure=DOT):
    u = query_embedding
    V = item_embeddings
    if measure == COSINE:
        V = V / np.linalg.norm(V, axis=1, keepdims=True)
        u = u / np.linalg.norm(u)
    scores = u.dot(V.T)
    return scores


USER_RATINGS = True


def user_recommendations(model, measure=DOT, exclude_rated=True, k=6):
    target_user_id_str = "942"
    target_user_id_int = int(target_user_id_str)

    if USER_RATINGS:
        scores = compute_scores(model.embeddings["user_id"][target_user_id_int], model.embeddings["movie_id"], measure)
        score_key = measure + ' score'
        df = pd.DataFrame({'movie_id': movies['movie_id'], 'titles': movies['title'], 'genres': movies['all_genres'],
                           score_key: list(scores)})
        if exclude_rated:
            rated_movies = ratings[ratings.user_id == target_user_id_str]["movie_id"].values
            df = df[~df.movie_id.isin(rated_movies)]
        # PENYESUAIAN UNTUK PYCHARM: Ganti display() dengan print()
        print(df.sort_values([score_key], ascending=False).head(k))


def movie_neighbors(model, title_substring, measure=DOT, k=6):
    ids = movies[movies['title'].str.contains(title_substring, case=False, regex=False)].index.values
    if len(ids) == 0:
        raise ValueError(f"Tidak ada film dengan judul '{title_substring}'")
    movie_id = ids[0]
    print(f"\nFilm terdekat dari: {movies.iloc[movie_id]['title']}")
    scores = compute_scores(model.embeddings["movie_id"][movie_id], model.embeddings["movie_id"], measure)
    score_key = measure + ' score'
    df = pd.DataFrame({'titles': movies['title'], 'genres': movies['all_genres'], score_key: list(scores)})
    # PENYESUAIAN UNTUK PYCHARM: Ganti display() dengan print()
    print(df.sort_values([score_key], ascending=False).head(k))


print("\n--- Rekomendasi untuk Pengguna 943 (indeks 942) ---")
print("\nRekomendasi berdasarkan Cosine Similarity:")
user_recommendations(model, measure=COSINE, k=5)
print("\nRekomendasi berdasarkan Dot Product:")
user_recommendations(model, measure=DOT, k=5)

print("\n--- Rekomendasi Film Mirip 'Star Wars' ---")
movie_neighbors(model, "Star Wars", DOT)
movie_neighbors(model, "Star Wars", COSINE)