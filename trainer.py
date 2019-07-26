import numpy as np
from typing import List, Tuple
import mygrad as mg
from noggin import create_plot

from mygrad.nnet import margin_ranking_loss

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
        plotter,
        batch_size: int = 150,
        epoch_cnt: int = 1000,
        margin: float = 0.1):

    for epoch in range(epoch_cnt):
        idxs = np.arange(len(triples))
        np.random.shuffle(idxs)

        query_embeds, good_images, bad_images = unzip(triples)
        query_embeds, good_images, bad_images = np.array(query_embeds), np.array(good_images), np.array(bad_images)

        correct_list = []

        for batch_cnt in range(0, len(triples)//batch_size):
            batch_indices = idxs[batch_cnt*batch_size : (batch_cnt + 1)*batch_size]

            batch_query = query_embeds[batch_indices].reshape(batch_size, 50)
            good_batch = good_images[batch_indices].reshape(batch_size, 512)
            bad_batch = bad_images[batch_indices].reshape(batch_size, 512)

            # print(batch_query.shape)
            # print(good_batch.shape)
            # print(bad_batch.shape)
            # print("____")

            good_image_encode: mg.Tensor = model(good_batch)
            bad_image_encode: mg.Tensor = model(bad_batch)

            # print(good_image_encode.shape)
            # print(bad_image_encode.shape)
            # print("____")

            good_image_encode /= mg.sqrt(mg.sum(good_image_encode**2, axis=1).reshape(batch_size, 1))
            bad_image_encode /= mg.sqrt(mg.sum(bad_image_encode**2, axis=1).reshape(batch_size, 1))
            batch_query /= mg.sqrt(mg.sum(batch_query**2, axis=1).reshape(batch_size, 1))

            # print(good_image_encode.shape)
            # print(bad_image_encode.shape)
            # print(batch_query.shape)

            good_dists = mg.einsum("ij,ij -> i", good_image_encode, batch_query)
            bad_dists = mg.einsum("ij,ij -> i", bad_image_encode, batch_query)

            correct_list.append(good_dists - bad_dists > margin)

            loss: mg.Tensor = margin_ranking_loss(good_dists, bad_dists, 1, margin=margin)

            loss.backward()

            optim.step()

            loss.null_gradients()

            plotter.set_train_batch({"loss": loss.item(), "acc": np.mean(np.array(correct_list))}, batch_size=batch_size)

        plotter.set_test_epoch()
