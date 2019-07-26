from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Tuple
import mygrad as mg

from mygrad.losses.margin_ranking_loss import margin_ranking_loss

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

            good_image_encode: mg.Tensor = model(good_images)
            bad_image_encode: mg.Tensor = model(bad_images)

            good_image_encode /= mg.sqrt(mg.sum(good_image_encode**2, axis=1))
            bad_image_encode /= mg.sqrt(mg.sum(bad_image_encode**2, axis=1))
            query_embeds /= mg.sqrt(mg.sum(query_embeds**2, axis=1))

            good_dists = mg.einsum("ij,ij -> i", good_image_encode, query_embeds)
            bad_dists = mg.einsum("ij,ij -> i", bad_image_encode, query_embeds)

            loss: mg.Tensor = margin_ranking_loss(good_dists, bad_dists, 1)

            loss.backward()

            optim.step()

            loss.null_gradients()

        print(loss)
