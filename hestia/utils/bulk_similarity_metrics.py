import numpy as np
from sklearn.metrics.pairwise import (cosine_similarity, manhattan_distances,
                                      euclidean_distances)


def bulk_np_tanimoto(u: np.ndarray, bulk: np.ndarray) -> np.ndarray:
    a = u.sum()
    out = np.zeros(bulk.shape[0])
    for i in range(bulk.shape[0]):
        b = bulk[i].sum()
        c = ((u + bulk[i]) > 1).sum()
        denominator = (a + b) - c
        if denominator == 0:
            out[i] = 1
        else:
            out[i] = c / denominator
    return out


def bulk_np_dice(u: np.ndarray, bulk: np.ndarray) -> np.ndarray:
    a = u.sum()
    out = np.zeros(bulk.shape[0])
    for i in range(bulk.shape[0]):
        b = bulk[i].sum()
        c = ((u + bulk[i]) > 1).sum()
        denominator = a + b
        if denominator == 0:
            out[i] = 1
        else:
            out[i] = (2 * c) / denominator
    return out


def bulk_np_sokal(u: np.ndarray, bulk: np.ndarray) -> np.ndarray:
    a = u.sum()
    out = np.zeros(bulk.shape[0])
    for i in range(bulk.shape[0]):
        b = bulk[i].sum()
        c = ((u + bulk[i]) > 1).sum()
        denominator = 2 * (a + b) - 3 * c
        if denominator == 0:
            out[i] = 1
        else:
            out[i] = c / denominator
    return out


def bulk_np_rogot_goldberg(u: np.ndarray, bulk: np.ndarray) -> np.ndarray:
    a = u.sum()
    e = bulk.shape[1]
    out = np.zeros(bulk.shape[0])

    for i in range(bulk.shape[0]):
        b = bulk[i].sum()
        c = np.dot(u, bulk[i])
        d = e + c - (a + b)

        denominator_1 = a + b
        denominator_2 = 2 * e - (a + b)
        if a == e or d == e:
            out[i] = 1
        elif denominator_1 == 0 or denominator_2 == 0:
            out[i] = 0
        else:
            out[i] = c / denominator_1 - d / denominator_2
    return out


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
