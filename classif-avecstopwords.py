# NB : the following code was inspired and adapted from https://medium.com/mlearning-ai/text-clustering-with-tf-idf-in-python-c94cd26a31e7

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# import other required libs
import pandas as pd
import numpy as np

# string manipulation libs
import re
import string
import nltk
from nltk.corpus import stopwords

# viz libs
import matplotlib.pyplot as plt
import seaborn as sns

#remplacer sentences par une liste de listes d'abstracts déjà tokenisés (précédemment extraits et mis dans le fichier abstract_decoupes_v2.json)
import json
from nltk.tokenize import word_tokenize
def ouvrir_json(chemin):
  f = open(chemin, encoding="utf-8")
  toto = json.load(f)
  f.close()
  return toto 

def preprocess_text(text: str, remove_stopwords: bool) -> str:
    """This utility function sanitizes a string by:
    - removing links
    - removing special characters
    - removing numbers
    - removing stopwords
    - transforming in lowercase
    - removing excessive whitespaces
    Args:
        text (str): the input text you want to clean
        remove_stopwords (bool): whether or not to remove stopwords
    Returns:
        str: the cleaned text
    """

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

import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords')


def get_top_keywords(n_terms):
    """This function returns the keywords for each centroid of the KMeans"""
    df = pd.DataFrame(X.todense()).groupby(clusters).mean() # groups the TF-IDF vector by cluster
    terms = vectorizer.get_feature_names() # access tf-idf terms
    for i,r in df.iterrows():
        with open('50keywords_%s_3clusters_lem_stopwords.txt'%partie, 'w') as output:
        #output.write(str(get_top_keywords(50)))
            output.write('\nCluster {}'.format(i))
            output.write(','.join([terms[t] for t in np.argsort(r)[-n_terms:]])) # for each row of the dataframe, find the n terms that have the highest tf idf score


liste_parties = ["abstracts", "intros","ccl","corps"]

for partie in liste_parties:
    print(partie)
    d = ouvrir_json("claims_decoupes_%s_v06-2.json"%partie)#[:20000]

    liste_claims = []
    for dico in d:
        for k,v in dico.items():
            if ".body" not in k:
                liste_claims.append(v)
    df = pd.DataFrame(liste_claims, columns=["corpus"])

    df['cleaned'] = df['corpus'].astype(np.uint8,errors='ignore').apply(lambda x: preprocess_text(x, remove_stopwords=False))
    print("df cleaned fait")

    # initialize the vectorizer
    vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=5, max_df=0.95)
    # fit_transform applies TF-IDF to clean texts - we save the array of vectors in X
    #X = vectorizer.fit_transform(df['cleaned'])
    X = vectorizer.fit_transform(df['cleaned'])
    print("vectorisation faite")

    # initialize kmeans with 3 centroids
    kmeans = KMeans(n_clusters=3, random_state=42)
    # kmeans = KMeans(n_clusters=6, random_state=42)

    # fit the model
    kmeans.fit(X)
    print("kmeans fit fait")
    # store cluster labels in a variable
    clusters = kmeans.labels_

    # initialize PCA with 2 components
    pca = PCA(n_components=2, random_state=42)
    print(X.shape)
    # pass our X to the pca and store the reduced vectors into pca_vecs
    pca_vecs = pca.fit_transform(X.toarray())
    print("pca fit transform fait")
    # save our two dimensions into x0 and x1
    x0 = pca_vecs[:, 0]
    x1 = pca_vecs[:, 1]

    # assign clusters and pca vectors to our dataframe 
    #clusters = kmeans.labels_
    df['cluster'] = clusters
    df['x0'] = x0
    df['x1'] = x1

    print(get_top_keywords(50))
    #with open('50keywords_%s_3clusters_lem_stopwords.txt'%partie, 'w') as output:
        #output.write(str(get_top_keywords(50)))

    #PARTIE VISUALISATION#
    # map clusters to appropriate labels 
    cluster_map = {0: "zero", 1: "one", 2: "two"}
    # apply mapping
    df['cluster'] = df['cluster'].map(cluster_map)

    df.to_csv("df_clusters_%s_stopwords.csv"%partie)
    print("csv fait")

        # set image size
    plt.figure(figsize=(12, 7))
    # set a title
    plt.title("TF-IDF + KMeans clustering", fontdict={"fontsize": 18})
    # set axes names
    plt.xlabel("X0", fontdict={"fontsize": 16})
    plt.ylabel("X1", fontdict={"fontsize": 16})
    # create scatter plot with seaborn, where hue is the class used to group the data
    sns.scatterplot(data=df, x="x0", y="x1", hue="cluster", style="cluster", palette="deep")
    plt.savefig("3clusters_%s_stopwords.png"%partie)
    print("figure sauvegardée")
    plt.show()
