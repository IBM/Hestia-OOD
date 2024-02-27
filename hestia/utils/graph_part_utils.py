"""
Code adapted from GraphPart
Author(s) of Original Code:
    Magnús Halldór Gíslason.
    Felix Teufel
    with the collaboration of
    José Juan Almagro Armenteros and Henrik Nielsen.

Original Project:
https://github.com/graph-part/graph-part/
"""
from typing import List, Tuple

import networkx as nx
import numpy as np
from tqdm import tqdm
import scipy.sparse as spr


def limited_agglomerative_clustering(
    mtx: spr.csc_matrix,
    n_parts: int,
    threshold: float,
    labels: List[int]
) -> np.ndarray:
    mtx = mtx > threshold
    part_size = mtx.shape[0] // n_parts
    rows, columns = mtx.nonzero()
    data = mtx.data
    sorted_data = np.argsort(data)[::-1]
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    label_limits = np.ceil(label_counts / n_parts)
    label_counts = np.zeros((len(unique_labels)))

    graph = nx.Graph()

    for item in range(mtx.shape[0]):
        graph.add_node(item)
        label_counts_item = label_counts.copy()
        label_counts_item[labels[item]] = 1
        attributes = {
            item: {
                'cluster': item,
                'size': 1,
                'label_counts': label_counts_item
            }
        }
        nx.set_node_attributes(graph, attributes)

    for item in tqdm(sorted_data):
        qry, tgt = rows[item], columns[item]
        if (graph.nodes[qry]['cluster'] == graph.nodes[tgt]['cluster'] or
            graph.nodes[qry]['size'] >= part_size or
            graph.nodes[tgt]['size'] >= part_size or
            (graph.nodes[qry]['label_counts'] +
             graph.nodes[tgt]['label_counts'] >= label_limits).any()):
            continue

        attribute = graph.nodes[qry]
        if attribute['cluster'] != graph.nodes[tgt]['cluster']:
            attribute['size'] += graph.nodes[tgt]['size']
            attribute['label_counts'] += graph.nodes[tgt]['label_counts']

        nx.set_node_attributes(graph, {tgt: attribute})
        graph.add_edge(qry, tgt)

        for descendant in nx.descendants(graph, qry):
            nx.set_node_attributes(graph, {descendant: attribute})

    ids, clusters = [], []

    for cc_nr, cc in enumerate(nx.connected_components(graph)):
        for n in cc:
            ids.append(n)
            clusters.append(cc_nr)

    clusters = np.array(clusters, dtype=int)
    return clusters[np.argsort(ids)]


def _neighbour_analysis(
    mtx: spr.csr_array,
    clusters: np.ndarray
) -> np.ndarray:
    E_f = np.zeros(mtx.shape[0], dtype=int)
    for qry in range(mtx.shape[0]):
        clust = clusters[qry]
        neighbours = mtx[qry, :].toarray().reshape(mtx.shape[0])
        neighbours = neighbours
        neighbour_clst = clusters[neighbours]
        clsts, num_nghbrs_in_clst = np.unique(neighbour_clst,
                                              return_counts=True)
        clst_idx = clsts == clust
        intra_clust_nghbrs = num_nghbrs_in_clst[clst_idx]
        inter_clust_nghbrs = num_nghbrs_in_clst.sum() - intra_clust_nghbrs
        if inter_clust_nghbrs > 0:
            E_f[qry] = inter_clust_nghbrs
    return E_f


def _cluster_reassignment(
    mtx: spr.csr_array,
    clusters: np.ndarray,
    removed: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    E_f = np.zeros(mtx.shape[0], dtype=int)
    for qry in range(mtx.shape[0]):
        if not removed[qry]:
            continue
        clust = clusters[qry]
        neighbours = mtx[qry, :].toarray().reshape(mtx.shape[0])
        neighbours = neighbours * removed
        neighbour_clst = clusters[neighbours]
        clsts, num_nghbrs_in_clst = np.unique(neighbour_clst,
                                              return_counts=True)
        clst_idx = clsts == clust
        intra_clust_nghbrs = num_nghbrs_in_clst[clst_idx]
        inter_clust_nghbrs = num_nghbrs_in_clst.sum() - intra_clust_nghbrs

        if inter_clust_nghbrs > 0:
            best_clst = np.argmax(num_nghbrs_in_clst)
            intra_clust_nghbrs = num_nghbrs_in_clst[best_clst]
            inter_clust_nghbrs = (num_nghbrs_in_clst.sum() -
                                  intra_clust_nghbrs)
            E_f[qry] += inter_clust_nghbrs
        else:
            best_clst = clusters[qry]
        clusters[qry] = best_clst
    return clusters, E_f


def _assign_partitions(clusters: np.ndarray, labels: np.ndarray,
                       n_parts: int, verbose: int) -> np.ndarray:
    cluster_inds, cluster_sizes = np.unique(clusters, return_counts=True)
    unique_labs, lab_counts = np.unique(labels, return_counts=True)
    n_labels = len(unique_labs)
    cluster_labs = np.ones((n_parts, n_labels), dtype=int)

    if verbose > 1:
        print(f'Clustering generated {len(cluster_inds):,} clusters...')

    for ind in cluster_inds:
        clst_members = clusters == ind
        clst_labels = labels[clst_members]
        label, count_labels = np.unique(clst_labels, return_counts=True)
        clst_lab_count = cluster_labs.copy()
        clst_lab_count[:, label] += count_labels
        clst_lab_prop = cluster_labs / clst_lab_count
        best_group = np.argmin(np.sum(clst_lab_prop, axis=1))
        cluster_labs[best_group, label] += count_labels
        clusters[clst_members] = best_group
    return clusters
