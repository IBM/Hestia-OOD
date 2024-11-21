import numpy as np
from sklearn.metrics.pairwise import (cosine_similarity, manhattan_distances,
                                      euclidean_distances)


def bulk_np_jaccard(u: np.ndarray, bulk: np.ndarray) -> np.ndarray:
    bits = bulk.shape[1]
    comp = (bulk == u)
    counts = comp.sum(1)
    return counts / bits


def bulk_np_tanimoto(u: np.ndarray, bulk: np.ndarray) -> np.ndarray:
    a = u.sum()
    b = bulk.sum(axis=1)
    c = np.dot(bulk, u)
    denominator = (a + b) - c
    return np.where(denominator == 0, 1, c / denominator)


def bulk_np_dice(u: np.ndarray, bulk: np.ndarray) -> np.ndarray:
    a = u.sum()
    b = bulk.sum(axis=1)
    c = np.dot(bulk, u)
    denominator = a + b
    return np.where(denominator == 0, 1, (2 * c) / denominator)


def bulk_np_sokal(u: np.ndarray, bulk: np.ndarray) -> np.ndarray:
    a = u.sum()
    b = bulk.sum(axis=1)
    c = np.dot(bulk, u)
    denominator = 2 * (a + b) - 3 * c
    return np.where(denominator == 0, 1, c / denominator)


def bulk_np_rogot_goldberg(u: np.ndarray, bulk: np.ndarray) -> np.ndarray:
    a = u.sum()
    e = bulk.shape[1]
    b = bulk.sum(axis=1)
    c = np.dot(bulk, u)
    d = e + c - (a + b)
    denominator_1 = a + b
    denominator_2 = 2 * e - (a + b)
    result = np.where((a == e) | (d == e),  1, np.where(
            (denominator_1 == 0) | (denominator_2 == 0),  0, c / denominator_1 - d / denominator_2
        )
    )
    return result


def bulk_cosine_similarity(u: np.ndarray, bulk: np.ndarray) -> np.ndarray:
    u_norm = u / np.linalg.norm(u)
    bulk_norm = bulk / np.linalg.norm(bulk, axis=1, keepdims=True)
    return np.dot(bulk_norm, u_norm)


def bulk_binary_manhattan_similarity(u: np.ndarray, bulk: np.ndarray) -> np.ndarray:
    distances = np.abs(bulk - u).sum(axis=1)
    return distances / u.shape[0]


def bulk_euclidean(u: np.ndarray, bulk: np.ndarray) -> np.ndarray:
    squared_diff = np.square(bulk - u)
    distances = np.sqrt(np.sum(squared_diff, axis=1))
    return distances


def bulk_manhattan(u: np.ndarray, bulk: np.ndarray) -> np.ndarray:
    distances = np.abs(bulk - u).sum(axis=1)
    return distances


def bulk_canberra(u: np.ndarray, bulk: np.ndarray) -> np.ndarray:
    diff = np.abs(bulk - u)
    sum_abs = np.abs(bulk) + np.abs(u)
    sum_abs = np.where(sum_abs == 0, 1e-10, sum_abs)
    return np.sum(diff / sum_abs, axis=1)
