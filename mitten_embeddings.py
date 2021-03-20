# adapted from https://towardsdatascience.com/fine-tune-glove-embeddings-using-mittens-89b5f3fe4c39

import csv
import pickle
import numpy as np
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from mittens import GloVe, Mittens
from sklearn.feature_extraction.text import CountVectorizer

from corpus_process import get_sms_data

glove_dict_path = r'../Data/embeddings_dict.pkl'
glove_file_path = r'../Data/glove.twitter.27B/glove.twitter.27B.50d.txt'

def glove2dict(glove_filename):
    with open(glove_filename, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
        embed = {line[0]: np.array(list(map(float, line[1:])))
                for line in reader}
    return embed

# glove_path = "glove.6B.50d.txt" # get it from https://nlp.stanford.edu/projects/glove
pre_glove = glove2dict(glove_file_path)

# print('Loading Glove...')
# pre_glove = pickle.load(open(glove_dict_path, 'rb'))
# print(pre_glove)

sw = set(stopwords.words('english'))
sms_data = get_sms_data()
sms_doc = [token.lower() for token in sms_data if (token.lower() not in sw)]
oov = set()
for doc in sms_doc:
    for token in word_tokenize(doc):
        if token not in pre_glove.keys():
            oov.add(token)

def get_rareoov(xdict, val):
    return [k for (k,v) in Counter(xdict).items() if v<=val]

#oov_rare = get_rareoov(oov, 1)
#corp_vocab = list(set(oov) - set(oov_rare))
#sms_tokens = [token for token in sms_nonstop if token not in oov_rare]
#sms_doc = [' '.join(sms_tokens)]

corp_vocab = list(set(oov))

cv = CountVectorizer(ngram_range=(1,1), vocabulary=corp_vocab)
X = cv.fit_transform(sms_doc)
Xc = (X.T * X)
Xc.setdiag(0)
coocc_ar = Xc.toarray()

mittens_model = Mittens(n=50, max_iter=1000)

print('mitten fit starts...')
new_embeddings = mittens_model.fit(
    coocc_ar,
    vocab=corp_vocab,
    initial_embedding_dict= pre_glove)

newglove = dict(zip(corp_vocab, new_embeddings))
f = open("sms_glove.pkl","wb")
pickle.dump(newglove, f)
f.close()