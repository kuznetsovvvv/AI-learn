import wikipedia
import time, json, os
import pandas as pd, numpy as np, matplotlib.pyplot as plt
import string, nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score
from scipy.cluster.hierarchy import linkage, dendrogram
import spacy

CACHE_FILE = "wiki_cache.json"

TOPIC1 = "Football clubs"
TOPIC2 = "Rock bands"
TOPIC3 = "Video games"

football_clubs = [
    "FC Barcelona", "Real Madrid CF", "Manchester United F.C.", "Liverpool F.C.",
    "FC Bayern Munich", "Juventus", "Paris Saint-Germain",
    "Chelsea F.C.", "Arsenal F.C.", "AC Milan", "Inter Milan",
    "Borussia Dortmund", "Atlético Madrid", "Tottenham Hotspur F.C.",
    "AFC Ajax", "FC Porto", "Manchester City F.C.", "S.L. Benfica",
    "Olympique de Marseille", "Everton F.C."
]

rock_bands = [
    "The Beatles", "Led Zeppelin", "Pink Floyd", "Queen (band)", "The Rolling Stones",
    "U2", "Nirvana (band)", "Metallica", "AC/DC", "Guns N' Roses",
    "The Who", "Aerosmith", "Black Sabbath", "Radiohead", "Red Hot Chili Peppers",
    "Foo Fighters", "The Doors", "Deep Purple", "Eagles (band)", "Fleetwood Mac"
]

video_games = [
    "Minecraft", "Tetris", "Grand Theft Auto V", "The Legend of Zelda: Breath of the Wild",
    "Super Mario Bros.", "The Witcher 3: Wild Hunt", "Fortnite", "League of Legends",
    "Counter-Strike: Global Offensive", "Call of Duty: Modern Warfare 2",
    "FIFA (video game series)", "World of Warcraft", "Overwatch (video game)",
    "The Elder Scrolls V: Skyrim", "Red Dead Redemption 2", "Dark Souls",
    "Halo 3", "Final Fantasy VII", "The Sims 4", "Street Fighter II"
]

all_topics = {
    TOPIC1: football_clubs,
    TOPIC2: rock_bands,
    TOPIC3: video_games
}

def load_with_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)
        return cache, False

    wikipedia.set_lang("en")
    wikipedia.set_user_agent("LabClusterBot/1.0 (student_project; contact@example.com)")

    cache = {}
    for theme, article_list in all_topics.items():
        for title in article_list:
            print(f"  Загрузка: {title} ({theme})")
            txt = fetch(title)
            if txt and len(txt) > 100:
                cache[title] = txt
            time.sleep(0.5)

    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)
    print(f"Кэш сохранён. Загружено {len(cache)} статей.")
    return cache, True

def fetch(title, retries=3):
    for attempt in range(retries):
        try:
            page = wikipedia.page(title, auto_suggest=False)
            return page.content
        except wikipedia.exceptions.DisambiguationError as e:
            try:
                page = wikipedia.page(e.options[0], auto_suggest=False)
                return page.content
            except:
                pass
        except wikipedia.exceptions.PageError:
            try:
                summary = wikipedia.summary(title, auto_suggest=False)
                if len(summary) > 100:
                    return summary
            except:
                pass
        except Exception:
            pass
        time.sleep(1.0 * (attempt + 1))
    return ""


cache, loaded = load_with_cache()

wiki_texts, titles, true_labels = [], [], []
for title, text in cache.items():
    for theme, lst in all_topics.items():
        if title in lst:
            true_labels.append(theme)
            titles.append(title)
            wiki_texts.append(text)
            break

print(f"Использовано статей: {len(wiki_texts)}")


nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)
nlp = spacy.load("en_core_web_sm")

def cleaner(text):
    doc = nlp(text.lower())
    lemmas = [token.lemma_ for token in doc
              if token.text not in punctuation and token.text not in stop_words]
    return " ".join(lemmas)

vectorizer = TfidfVectorizer(preprocessor=cleaner, max_features=5000)
X = vectorizer.fit_transform(wiki_texts)
print(f"TF‑IDF matrix shape: {X.shape}")


wcss = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
    km.fit(X)
    wcss.append(km.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1,11), wcss, 'bx-')
plt.xlabel('k'); plt.ylabel('WCSS')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.show()

optimal_k = 3


print("\nK‑means:")
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=42)
kmeans_labels = kmeans.fit_predict(X)
print(f"ARI K‑means: {adjusted_rand_score(true_labels, kmeans_labels):.3f}")

print("\nHierarchical:")
X_dense = X.toarray()
linkage_matrix = linkage(X_dense, method='ward')
plt.figure(figsize=(14,7))
dendrogram(linkage_matrix, labels=titles, leaf_rotation=90, leaf_font_size=8)
plt.title('Hierarchical Clustering Dendrogram')
plt.tight_layout()
plt.show()

print("\nDBSCAN:")
svd = TruncatedSVD(n_components=50, random_state=42)
X_svd = svd.fit_transform(X)

dbscan = DBSCAN(eps=0.70, min_samples=2, metric='cosine')
dbscan_labels = dbscan.fit_predict(X_svd)
print(f"ARI DBSCAN: {adjusted_rand_score(true_labels, dbscan_labels):.3f}")

df_dbscan = pd.DataFrame({'Title': titles, 'True Theme': true_labels, 'DBSCAN': dbscan_labels})
print(df_dbscan.sort_values('DBSCAN').to_string(index=False))



tsne = TSNE(n_components=2, random_state=42, perplexity=10)
X_tsne = tsne.fit_transform(X_dense)
plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=kmeans_labels, cmap='tab10', s=50)
plt.title('K‑means')
plt.subplot(1,2,2)
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=dbscan_labels, cmap='tab10', s=50)
plt.title('DBSCAN')
plt.tight_layout()
plt.show()