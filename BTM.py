import bitermplus as btm
import numpy as np
import pandas as pd
from nltk.corpus import stopwords

def preprocess_text(text): #recibe una tuit
    spanish_stopwords = set(stopwords.words('spanish'))
    tokens = text.split() #separamos en palabras

    #filtramos las stopwords, las palabras cortas y las urls y volvemos a formar la oracion
    return ' '.join([token for token in tokens if token not in spanish_stopwords and len(token) > 3 and 'twitter.com' not in token])

# IMPORTING DATA
# Cargar datos
data = pd.read_csv("tweets_municipalidad.csv")
texts = data['tweet'].astype(str)
texts = [preprocess_text(text) for text in texts]

# get spanish stop words
spanish_stop_words = set(stopwords.words('spanish'))

# PREPROCESSING
# Obtaining terms frequency in a sparse matrix and corpus vocabulary
X, vocabulary, vocab_dict = btm.get_words_freqs(texts, stop_words=list(spanish_stop_words))
tf = np.array(X.sum(axis=0)).ravel()

# Vectorizing documents
docs_vec = btm.get_vectorized_docs(texts, vocabulary)
docs_lens = list(map(len, docs_vec))

# Generating biterms
biterms = btm.get_biterms(docs_vec)


model = btm.BTM(X, vocabulary, seed=12321, T=30, M=10, alpha=0.5, beta=0.5)
model.fit(biterms, iterations=25)
p_zd = model.transform(docs_vec)

perplexity = model.perplexity_
coherence = np.mean(model.coherence_)

stable_topics = list(set(model.labels_))
top_words = btm.get_top_topic_words(
    model,
    words_num=15,
    topics_idx=stable_topics)

print('top_words => ', top_words)
print('coherence => ', coherence)
print('perplexity => ', perplexity)

