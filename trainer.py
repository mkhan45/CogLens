from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Tuple
import mygrad as mg

from embed_text import se_text

def unzip(pairs):
    """
    "unzips" of groups of items into separate lists.
    
    Example: pairs = [("a", 1), ("b", 2), ...] --> (("a", "b", ...), (1, 2, ...))
    """
    return tuple(zip(*pairs))

def train(model, 
        triples: List[Tuple[np.ndarray, np.ndarray, np.ndarray]], #caption embeds, good_images, bad_images
        optim,
        batch_size: int = 15,
        epoch_cnt: int = 1000,
        margin: float = 0.1):

    for epoch in range(epoch_cnt):
        idxs = np.arange(len(triples))
        np.random.shuffle(idxs)


        for batch_cnt in range(0, len(triples)//batch_size):
            batch_indices = idxs[batch_cnt*batch_size : (batch_cnt + 1)*batch_size]
            batch: List[Tuple[str, np.ndarray, np.ndarray]] = triples[batch_indices]

            query_embeds, good_images, bad_images = unzip(batch)

            good_image_encode: np.ndarray = model(good_images)
            bad_image_encode: np.ndarray = model(bad_images)

            good_image_encode /= np.linalg.norm(good_image_encode, axis=1)
            bad_image_encode /= np.linalg.norm(bad_image_encode, axis=1)
            query_embeds /= np.linalg.norm(query_embeds , axis=1)

            good_dists = np.einsum("ij,ij -> i", good_image_encode, query_embeds)
            bad_dists = np.einsum("ij,ij -> i", bad_image_encode, query_embeds)

            loss: mg.Tensor = np.max(0, margin - (good_dists - bad_dists))

            loss.backward()

            optim.step()

            loss.null_gradients()

        print(loss)
