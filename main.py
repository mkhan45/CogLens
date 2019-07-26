from gensim.models.keyedvectors import KeyedVectors as KV
import embed_text as et
import json
from model import Model

#Load files
with open('/Users/crystal/repositories/CogLens/stopwords.txt', 'r') as r: #stop words
    stops = []
    for line in r:
        stops += [i.strip() for i in line.split('\t')]

glove50 = KV.load_word2vec_format('/Users/crystal/Desktop/python-workspace/CogWorks2019/glove.6B.50d.txt.w2v', binary=False) #glove embeddings

with open("/Users/crystal/Desktop/python-workspace/CogWorks2019/captions_train2014.json", "r") as f:
    coco_metadata = json.load(f) #Dictionary

print("Files loaded.")

def search(user_input):
    #Make an embedding
    all_captions = list(i["caption"] for i in coco_metadata['annotations']) #list of captions
    print('captions loaded')
    counters = et.to_counters(all_captions)
    vocab = et.to_vocab(counters, stop_words=stops)
    idf = et.to_idf(all_captions, vocab)
    embedding = et.se_text(input, glove50, idf, vocab) #change "from Flask" to whatever caption
    return embedding