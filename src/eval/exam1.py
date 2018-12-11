import math

import numpy as np


def dis(embeddings1, embeddings2):
    dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
    norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
    similarity = dot / norm
    dist = np.arccos(similarity) / math.pi
    return similarity, dist


print dis(np.array([[0, 1]]), np.array([[1, 0]]))
