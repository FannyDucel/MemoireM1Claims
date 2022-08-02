from sklearn.metrics import davies_bouldin_score
from scipy.spatial.distance import *    
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.metrics import pairwise_distances
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import re
import string
import nltk
from nltk.corpus import stopwords
import json
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def ouvrir_json(chemin):
  f = open(chemin, encoding="utf-8")
  toto = json.load(f)
  f.close()
  return toto

def preprocess_text(text: str, remove_stopwords: bool) -> str:
    # remove links
    text = re.sub(r"http\S+", "", text)
    # remove special chars and numbers
    text = re.sub("[^A-Za-z]+", " ", text)
    # remove stopwords
    if remove_stopwords:
        # 1. tokenize
        tokens = nltk.word_tokenize(text)
        # 2. check if stopword
        tokens = [w for w in tokens if not w.lower() in stopwords.words("english")]
        # 3. join back together
        text = " ".join(tokens)
    # return text in lower case and stripped of whitespaces
    text = text.lower().strip()
    return text

liste_fichiers=	[ "claims_decoupes_intros_v3.json", "claims_decoupes_intros_v06-2.json", "claims_decoupes_ccl_v3.json","claims_decoupes_ccl_v06-2.json","claims_decoupes_corps_v3.json", "clams_decoupes_v06-2.json"]

for fichier in liste_fichiers:
    print(fichier)
    d = ouvrir_json(fichier)#[:20000]

    liste_claims = []
    # On relance tout le processus pour refaire le clustering
    for dico in d:
        for k,v in dico.items():
            liste_claims.append(v)
    df = pd.DataFrame(liste_claims, columns=["corpus"])


    print("En enlevant les stopwords")
    df['cleaned'] = df['corpus'].astype(np.uint8,errors='ignore').apply(lambda x: preprocess_text(x, remove_stopwords=True))
    print("df[cleaned] fait")

    # initialize the vectorizer
    vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=5, max_df=0.95)

    X = vectorizer.fit_transform(df['cleaned'])

    # initialize kmeans with 3 centroids
    kmeans = KMeans(n_clusters=3, random_state=42)

    kmeans.fit(X)
    # store cluster labels in a variable
    clusters = kmeans.labels_

    # initialize PCA with 2 components
    pca = PCA(n_components=2, random_state=42)
    # pass our X to the pca and store the reduced vectors into pca_vecs
    pca_vecs = pca.fit_transform(X.toarray())
    # save our two dimensions into x0 and x1
    x0 = pca_vecs[:, 0]
    x1 = pca_vecs[:, 1]

    df['cluster'] = clusters
    df['x0'] = x0
    df['x1'] = x1

    #pour chaque clustering, on calcule les différents silhouettes scores
    """SILHOUETTE SCORE"""
    features = vectorizer.transform(df.cleaned.values)
    distances = ["cityblock", "cosine", "euclidean", "l1", "l2", "manhattan"]

    print("silhouette_score")
    for d in distances:
        print(d, silhouette_score(features, labels=kmeans.predict(features), metric=d))

    # CALINSKI HARABASZ SCORE
    X = X.toarray()
    kmeans_model = KMeans(n_clusters=3, random_state=1).fit(X)
    labels = kmeans_model.labels_
    metrics.calinski_harabasz_score(X, labels)
    print("CH score")
    print(metrics.calinski_harabasz_score(X, labels))

    # DAVIES BOULDIN SCORE
    labels = kmeans_model.labels_
    print("DB score")
    print(davies_bouldin_score(X, labels))

    # ON PREND LES MEMES ET ON RECOMMENCE MAIS EN GARDANT LES STOPWORDS
    print("En GARDANT les stopwords")
    df['cleaned'] = df['corpus'].astype(np.uint8,errors='ignore').apply(lambda x: preprocess_text(x, remove_stopwords=False))
    print("df[cleaned] fait")

    # initialize the vectorizer
    vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=5, max_df=0.95)

    X = vectorizer.fit_transform(df['cleaned'])

    # initialize kmeans with 3 centroids
    kmeans = KMeans(n_clusters=3, random_state=42)

    kmeans.fit(X)
    # store cluster labels in a variable
    clusters = kmeans.labels_

    # initialize PCA with 2 components
    pca = PCA(n_components=2, random_state=42)
    # pass our X to the pca and store the reduced vectors into pca_vecs
    pca_vecs = pca.fit_transform(X.toarray())
    # save our two dimensions into x0 and x1
    x0 = pca_vecs[:, 0]
    x1 = pca_vecs[:, 1]

    df['cluster'] = clusters
    df['x0'] = x0
    df['x1'] = x1

    """SILHOUETTE SCORE"""
    features = vectorizer.transform(df.cleaned.values)
    distances = ["cityblock", "cosine", "euclidean", "l1", "l2", "manhattan"]

    print("silhouette_score")
    for d in distances:
        print(d, silhouette_score(features, labels=kmeans.predict(features), metric=d))

    X = X.toarray()
    kmeans_model = KMeans(n_clusters=3, random_state=1).fit(X)
    labels = kmeans_model.labels_
    metrics.calinski_harabasz_score(X, labels)
    print("CH score")
    print(metrics.calinski_harabasz_score(X, labels))

     
    # we store the cluster labels
    labels = kmeans_model.labels_
    print("DB score")
    print(davies_bouldin_score(X, labels))