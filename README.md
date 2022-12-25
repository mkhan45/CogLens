# CogLens
## By Crystal Zhu, Mikail Khan, Christy Jestin

We built an image search engine for the Microsoft COCO dataset. A query is encoded using a bag of words representation with TF-IDF weighting and Word2Vec for word embeddings. The query vector is then placed in the image space via an encoder. This image embedding is then compared to the ResNet embeddings of images in the COCO dataset, and we return the most relevant images.

The encoder is trained on the Microsoft COCO dataset such that matching image caption pairs have high similarity.
