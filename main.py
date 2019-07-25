from gensim.models.keyedvectors import KeyedVectors as KV
import embed_text as et
import json

#Load files
with open("stopwords.txt", 'r') as r: #stop words
    stops = []
    for line in r:
        stops += [i.strip() for i in line.split('\t')]

glove50 = KV.load_word2vec_format('path/to/glove.w2v', binary=False) #glove embeddings

coco_metadata = json.load('path/to/coco_metadata.json') #Dictionary

print("Files loaded.")

#Make an embedding
all_captions = list(coco_metadata['annotations'][i]['caption'] for i in coco_metadata['annotations']) #list of captions
counters = et.to_counters(all_captions)
vocab = et.to_vocab(counters, stop_words=stops)
idf = et.to_idf(counters, vocab)

#embedding = et.se_text("mouse flying", glove50, idf, vocab) #change "mouse flying" to whatever caption
