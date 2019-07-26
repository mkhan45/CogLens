from collections import defaultdict
import matplotlib.pyplot as plt
from pathlib import Path
from urllib.request import urlopen
import json
import pickle
import numpy as np
from trainer import train
from model import Model
import mygrad as mg

with open("/Users/crystal/Desktop/python-workspace/CogWorks2019/captions_train2014.json", "r") as read_file:
    captions_train = json.load(read_file)
    
with open('/Users/crystal/Desktop/python-workspace/CogWorks2019/resnet18_features.pkl', "rb") as read_file:
    images_dict = pickle.load(read_file)

keys_array = np.array(images_dict.keys())
url_dict = dict()
for image_set in captions_train['images']:
    if image_set['id'] in keys_array:
        url_dict[image_set['id']] = image_set['coco_url']
        
captions_dict = defaultdict(list)
captions_list = list()
for annotation in captions_train['annotations']:
    if annotation['image_id'] in keys_array:
        captions_dict[annotation['image_id']].append(annotation)
        captions_list.append(annotation['caption'])
        
        
        
def find_matches(embedding):
    '''
    Parameters:
    --------------------------------
    query: string
    
    
    
    Returns
    --------------------------------
    id_list: List[int]
    List of IDs for Images that match the given query
    '''
    with open("keys_array.pkl", "rb") as read_file:
        keys_array = pickle.load(read_file)
    
    modelled_images = np.load('modelled_images.npy')
    dists = mg.einsum("ij,ij -> i", modelled_images, embedding)
    return keys_array[dists>0.7].flatten()
  
def display_image(ids):
    '''
    Parameters:
    --------------------------------
    
    
    Returns
    --------------------------------
    Matploblib of Matching Image Results
    '''
    n = np.ceil(np.sqrt(ids.shape[0]))
    fig, axes = plt.subplots(nrows = n, ncols = n)
    j = np.ceil((ids.shape[0])/n)
    for i in range(n):
        for j in range(n):
            id = ids[i*n+j]
            data = urlopen(url_dict[id])
            # converting the downloaded bytes into a numpy-array
            img = plt.imread(data, format='jpg')
            axes[i,j].imshow(img)
            
def create_triples(ids):
    triples_list = list()
    for id in ids:
        good_img = images_dict[id]
        n = np.random.choice(keys_array)
        while n==id:
            n = np.random.random.choice(keys_array)
        good_img = images_dict[id]
        bad_img = images_dict[n]
        #replace embed_dict with what dictionary contains embeddings for captions
        caption = embed_dict[id][int(np.random.randint(0,5))]
        triples_list.append(caption, good_img, bad_img)
    return triples_list

triples = create_triples(keys_array[0:10000])

from mynn.optimizers.Adam import Adam

model = Model(512, 50)
optim = Adam(model.parameters)
train(triples, optim)

filename = input("What should db file be named?\n")
with open(filename, 'wb') as f:
    pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
