# -*- coding: utf-8 -*-

# Call GPU device
import time
import requests
import pandas as pd
from collections import Counter
from requests_html import HTMLSession
from sklearn.cluster import KMeans
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
!nvidia-smi

# Install SBERT, PyNLP1, NLTK
!pip install - q - U sentence-transformers
!pip install PyNLPl
!pip install - -user - U nltk

# Import

# Webscrape steam page with parameters required by steam api


class Reviews:
    def __init__(self, asin) -> None:
        self.asin = asin
        self.session = HTMLSession()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36'}
        self.url = f'http://www.amazon.ca/product-reviews/{self.asin}/ref=cm_cr_dp_d_show_all_btm?ie=UTF8reviewerType=all_reviews&sortBy=recent&pageNumber='

    def pagination(self, page):
        pages = page
        while pages != 25:
            r = self.session.get(self.url + str(page))
            return r.html.find('div[data-hook=review]')

    def parse(self, reviews):
        total = []
        for review in reviews:
            time.sleep(4)
            title = review.find('a[data-hook=review-title]', first=True).text
            rating = review.find(
                'i[data-hook=review-star-rating] span', first=True).text
            body = review.find(
                'span[data-hook=review-body] span', first=True).text.replace('\n', '').strip()

            data = {
                'title': title,
                'rating': rating,
                'body': body[:100]
            }
            total.append(data)
        return total


amz = Reviews('B08SWPYTFF')
reviews = amz.pagination(1)
print(amz.parse(reviews))

df = pd.DataFrame.from_dict(amz.parse(reviews))

# import SBERT model
model = SentenceTransformer('jhgan/ko-sroberta-multitask')
corpus = df['body'].values.tolist(0:)

# Create encode to calculate clustering
embeddings = model.encode(corpus)

# Display data metrics into 15 clusters
num_clusters = 5
clustering_model = KMeans(n_clusters=num_clusters)
clustering_model.fit(embeddings)
cluster_assignment = clustering_model.labels_

clustered_sentences = [[] for i in range(num_clusters)]
for sentence_id, cluster_id in enumerate(cluster_assignment):
    clustered_sentences[cluster_id].append(corpus[sentence_id])

for i, cluster in enumerate(clustered_sentences):
    print('Cluster %d (%d)' % (i+1, len(cluster)))
    print(cluster)
    print('')

# Noun list
nltk.download('punkt')
nouns = []

# Append all individual words into nouns
for review in df['body'].values.tolist():
    nouns.append(word_tokenize(review))

count = Counter(c for clist in nouns for c in clist)
words = dict(count.most_common())

for i, (word, count) in enumerate(words.items()):
    if i > 50:
        break

    # Display all words and their counts
    print(word, count)
