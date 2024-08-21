import numpy as np
from sklearn.metrics.pairwise import (cosine_similarity, manhattan_distances,
                                      euclidean_distances)


def bulk_cosine_similarity(u: np.ndarray, bulk: np.ndarray) -> np.ndarray:
    out = np.zeros(bulk.shape[0])
    for i in range(bulk.shape[0]):
        out[i] = cosine_similarity(
            u.reshape(1, -1), bulk[i].reshape(1, -1)
        ).item()
        print(out[i])
    return out


def bulk_binary_manhattan_similarity(
        u: np.ndarray, bulk: np.ndarray) -> np.ndarray:
    out = np.zeros(bulk.shape[0])
    for i in range(bulk.shape[0]):
        distance = manhattan_distances(
            u.reshape(1, -1), bulk[i].reshape(1, -1)
        ).item() / (u.shape[0])
        out[i] = 1 - distance
    return out


def bulk_binary_euclidean_similarity(
        u: np.ndarray, bulk: np.ndarray) -> np.ndarray:
    out = np.zeros(bulk.shape[0])
    for i in range(bulk.shape[0]):
        distance = euclidean_distances(
            u.reshape(1, -1), bulk[i].reshape(1, -1)
        ).item() / (u.shape[0])
        out[i] = 1 - distance
    return out


def bulk_euclidean(u: np.ndarray, bulk: np.ndarray) -> np.ndarray:
    out = np.zeros(bulk.shape[0])
    for i in range(bulk.shape[0]):
        out[i] = euclidean_distances(
            u.reshape(1, -1), bulk[i].reshape(1, -1)
        ).item()
    return out


def bulk_manhattan(u: np.ndarray, bulk: np.ndarray) -> np.ndarray:
    out = np.zeros(bulk.shape[0])
    for i in range(bulk.shape[0]):
        out[i] = manhattan_distances(
            u.reshape(1, -1), bulk[i].reshape(1, -1)
        ).item()
    return out
