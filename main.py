from gensim.models.keyedvectors import KeyedVectors as KV
import embed_text as et
import numpy as np
import json, pickle
from model import Model
from train_model_dummy import find_matches

#Load files
with open('/Users/crystal/repositories/CogLens/stopwords.txt', 'r') as r: #stop words
    stops = []
    for line in r:
        stops += [i.strip() for i in line.split('\t')]

glove50 = KV.load_word2vec_format('/Users/crystal/Desktop/python-workspace/CogWorks2019/glove.6B.50d.txt.w2v', binary=False) #glove embeddings

#with open("/Users/crystal/Desktop/python-workspace/CogWorks2019/captions_train2014.json", "r") as f:
    #coco_metadata = json.load(f) #Dictionary

with open('/Users/crystal/repositories/CogLens/vocab.pkl', 'rb') as f1:
    vocab = pickle.load(f1)

idf = np.load('/Users/crystal/repositories/CogLens/idfs.npy')

with open('/Users/crystal/repositories/CogLens/url_dict.pkl', 'rb') as f3:
    url_dict = pickle.load(f3)

print("Files loaded.")

def search(user_input):
    #Make an embedding
    #all_captions = list(i["caption"] for i in coco_metadata['annotations']) #list of captions
    #print('captions loaded')
    #counters = et.to_counters(all_captions)
    #vocab = et.to_vocab(counters, stop_words=stops) #vocab.pkl
    #idf = et.to_idf(all_captions, vocab) #idfs.npy
    embedding = et.se_text(user_input, glove50, idf, vocab) #change "from Flask" to whatever caption
    img_ids = find_matches(embedding)
    img_ids = sorted(img_ids)[::-1]
    urls = list(url_dict[img_id] for img_id in img_ids)
    #urls = list(j['coco_url'] for j in coco_metadata['images'] if j['id'] in img_ids) #url_dict.pkl

    return urls