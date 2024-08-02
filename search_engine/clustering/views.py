from django.shortcuts import render
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

def cluster_home(request):
    return render(request, 'cluster.html')

def cluster_result(request):
    csv_path = 'C:/Users/user/Downloads/Assignment/Assignment IR/Task 1/search _engine_project/search_engine/static/news.csv'
    df = pd.read_csv(csv_path)
    
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    cleaned_texts = []
    for text in df['Document']:
        tokens = nltk.word_tokenize(text)
        words = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.isalnum() and word.lower() not in stop_words]
        cleaned_texts.append(' '.join(words))

    num_clusters = min(len(df['Document']), 3)
     # Define the range of configurations to test
    # ngram_ranges = [(1, 2), (1,3)]
    # max_dfs = [0.5, 0.95]
    # min_dfs = [5]

    # best_silhouette_score = -1
    # best_tfidf_vectorizer = None
    # best_tfidf_matrix = None
    # best_kmeans_model = None

    # for ngram_range in ngram_ranges:
    #     for max_df in max_dfs:
    #         for min_df in min_dfs:
    #             tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=ngram_range, max_df=max_df, min_df=min_df)
    #             tfidf_matrix = tfidf_vectorizer.fit_transform(cleaned_texts)

    #             kmeans_model = KMeans(n_clusters=3, random_state=0, n_init=10)
    #             kmeans_model.fit(tfidf_matrix)

    #             pca = PCA(n_components=2)
    #             reduced_matrix = pca.fit_transform(tfidf_matrix.toarray())
    #             silhouette_avg = silhouette_score(reduced_matrix, kmeans_model.labels_)
    #             print(f"Configuration: ngram_range={ngram_range}, max_df={max_df}, min_df={min_df}, Silhouette Score={silhouette_avg}")

    #             if silhouette_avg > best_silhouette_score:
    #                 best_silhouette_score = silhouette_avg
    #                 best_tfidf_vectorizer = tfidf_vectorizer
    #                 best_tfidf_matrix = tfidf_matrix
    #                 best_kmeans_model = kmeans_model

    # print("Best Silhouette Score:", best_silhouette_score)
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_df=0.95, min_df=3)
    tfidf_matrix = tfidf_vectorizer.fit_transform(cleaned_texts)

    kmeans_model = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans_model.fit(tfidf_matrix)

    pca = PCA(n_components=2)
    reduced_matrix = pca.fit_transform(tfidf_matrix.toarray())
    silhouette_avg = silhouette_score(reduced_matrix, kmeans_model.labels_)
    print("Silhouette Score:", silhouette_avg)

    order_centroids = kmeans_model.cluster_centers_.argsort()[:, ::-1]
    terms = tfidf_vectorizer.get_feature_names_out()
    top_terms = {}
    for i in range(num_clusters):
        top_terms[i] = [terms[ind] for ind in order_centroids[i, :10]]
        print(f"Cluster {i} top words: {top_terms[i]}")

    if request.method == 'POST':
        new_text = request.POST.get('document')
        new_tokens = nltk.word_tokenize(new_text)
        new_words = [lemmatizer.lemmatize(word.lower()) for word in new_tokens if word.isalnum() and word.lower() not in stop_words]
        new_cleaned_text = ' '.join(new_words)
        new_tfidf_matrix = tfidf_vectorizer.transform([new_cleaned_text])

        predicted_cluster = kmeans_model.predict(new_tfidf_matrix)[0]
        cluster_labels = {0: "Politics", 1: "Entertainment", 2: "Economy", 3: "Sports"}
        cluster_label = cluster_labels.get(predicted_cluster, 'unknown')

        context = {
            'cluster_category': cluster_label,
        }
        return render(request, 'cluster-result.html', context)

    return render(request, 'cluster.html')
