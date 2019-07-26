import pickle
import numpy as np
import mygrad as mg

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