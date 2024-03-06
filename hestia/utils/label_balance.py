import numpy as np


def _balanced_labels(labels: np.ndarray, value: list, test: list,
                     test_size: float,
                     df_size: int = None) -> bool:
    if labels is None:
        return ((len(test) + len(value)) / df_size) < test_size
    if ((len(test) + len(value)) / df_size) > test_size:
        return False
    unique_labels = np.unique(labels)
    new_labels, new_counts = np.unique(labels[value], return_counts=True)
    test_labels, test_counts = np.unique(labels[test], return_counts=True)
    proportion = test_size * (labels.shape[0] / len(unique_labels))
    for label in unique_labels:
        test_idx = np.where(test_labels == label)
        new_idx = np.where(new_labels == label)
        if test_counts[test_idx] + new_counts[new_idx] > proportion * 1.3:
            return False
    return True
