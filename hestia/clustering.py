import time

import pandas as pd
from scipy.sparse.csgraph import connected_components

from hestia.similarity import sim_df2mtx


def generate_clusters(
    df: pd.DataFrame,
    field_name: str,
    sim_df: pd.DataFrame,
    threshold: float = 0.4,
    verbose: int = 0,
    cluster_algorithm: str = 'greedy_incremental',
) -> pd.DataFrame:
    """Generates clusters from a DataFrame.

    :param df: DataFrame with entities to cluster.
    :type df: pd.DataFrame
    :param field_name: Name of the field with the entity information
    (e.g., `protein_sequence` or `structure_path`), defaults to 'sequence'.
    :type field_name: str
    :param threshold: Similarity value above which entities will be
    considered similar, defaults to 0.4
    :param sim_df: DataFrame with similarities (`metric`) between
    `query` and `target`, it is the product of `calculate_similarity` function
    :type sim_df: pd.DataFrame
    :type threshold: float
    :param verbose: How much information will be displayed.
    Options:
        - 0: Errors,
        - 1: Warnings,
        - 2: All
    Defaults to 0
    :type verbose: int
    :param cluster_algorithm: Clustering algorithm to use.
    Options:
        - `CDHIT` or `greedy_incremental`
        - `greedy_cover_set`
        - `connected_components`
    Defaults to "CDHIT".
    :type cluster_algorithm: str, optional
    :raises NotImplementedError: Clustering algorithm is not supported
    :return: DataFrame with entities and the cluster they belong to.
    :rtype: pd.DataFrame
    """
    start = time.time()
    if cluster_algorithm in ['greedy_incremental', 'CDHIT']:
        cluster_df = _greedy_incremental_clustering(df, field_name, sim_df,
                                                    threshold, verbose)
    elif cluster_algorithm in ['greedy_cover_set']:
        cluster_df = _greedy_cover_set(df, sim_df, threshold, verbose)
    elif cluster_algorithm in ['connected_components']:
        cluster_df = _connected_components_clustering(df, sim_df, threshold,
                                                      verbose)
    else:
        raise NotImplementedError(
            f'Clustering algorithm: {cluster_algorithm} is not supported'
        )
    if verbose > 1:
        print(f'Clustering has taken {time.time() - start:.3f} s to compute.')

    return cluster_df


def _greedy_incremental_clustering(
    df: pd.DataFrame,
    field_name: str,
    sim_df: pd.DataFrame,
    threshold: float,
    verbose: int
) -> pd.DataFrame:
    df['lengths'] = df[field_name].map(len)
    df.sort_values(by='lengths', ascending=False, inplace=True)

    clusters = []
    clustered = set()
    sim_df = sim_df[sim_df['metric'] > threshold]

    for i in df.index:
        in_cluster = set(sim_df.loc[sim_df['query'] == i, 'target'])
        in_cluster.update(set(sim_df.loc[sim_df['target'] == i, 'query']))

        if i in clustered:
            continue

        for j in in_cluster:
            if i == j:
                continue
            clusters.append({
                'cluster': i,
                'member': j
            })
        clustered.update(in_cluster)

    cluster_df = pd.DataFrame(clusters)

    if verbose > 1:
        print('Clustering has generated:',
              f'{len(cluster_df.cluster.unique()):,d} clusters for',
              f'{len(df):,} entities')
    return cluster_df


def _greedy_cover_set(
    df: pd.DataFrame,
    sim_df: pd.DataFrame,
    threshold: float,
    verbose: int
) -> pd.DataFrame:
    def _find_connectivity(df, sim_df):
        neighbours = []
        for i in df.index:
            in_cluster = set(sim_df.loc[sim_df['query'] == i, 'target'])
            in_cluster.update(set(sim_df.loc[sim_df['target'] == i, 'query']))
            neighbours.append(in_cluster)
        return neighbours

    sim_df = sim_df[sim_df['metric'] > threshold]
    neighbours = _find_connectivity(df, sim_df)
    df['conectivity'] = list(map(len, neighbours))
    df.sort_values(by='conectivity', ascending=False, inplace=True)

    clusters = []
    clustered = set()

    for i in df.index:
        in_cluster = neighbours.pop(0)

        if i in clustered:
            continue
        for j in in_cluster:
            if i == j:
                continue
            clusters.append({
                'cluster': i,
                'member': j
            })
        clustered.update(in_cluster)

    cluster_df = pd.DataFrame(clusters)

    if verbose > 1:
        print('Clustering has generated:',
              f'{len(cluster_df.cluster.unique()):,d} clusters for',
              f'{len(df):,} entities')
    return cluster_df


def _connected_components_clustering(
    df: pd.DataFrame,
    sim_df: pd.DataFrame,
    threshold: float,
    verbose: int
) -> pd.DataFrame:
    matrix = sim_df2mtx(sim_df, threshold)
    n, labels = connected_components(matrix, directed=True,
                                     return_labels=True)
    cluster_df = [{'cluster': labels[i],
                   'member': i} for i in range(labels.shape[0])]
    if verbose > 0:
        print('Clustering has generated:',
              f'{n:,d} connected componentes for',
              f'{len(df):,} entities')
    return pd.DataFrame(cluster_df)
