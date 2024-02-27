from multiprocessing import cpu_count
from typing import Optional

import pandas as pd

from hestia.clustering import generate_clusters
from hestia.similarity import calculate_similarity


def similarity_reduction(
    df: pd.DataFrame,
    similarity_metric: str,
    field_name: str,
    threads: int = cpu_count(),
    clustering_mode: str = "CDHIT",
    denominator: str = "shortest",
    threshold: float = 0.3,
    verbose: int = 2,
    species: str = 'protein',
    distance: str = 'tanimoto',
    representation: str = '3di+aa',
    bits: int = 1024,
    radius: int = 2,
    sim_df: Optional[pd.DataFrame] = None,
    config: dict = None
) -> pd.DataFrame:
    if 'connected_components' == clustering_mode:
        mssg = 'Similarity reduction is not'
        mssg += ' implemented for connected components.'
        raise NotImplementedError(mssg)
    if sim_df is None:
        sim_df = calculate_similarity(
            df, df, species=species,
            similarity_metric=similarity_metric,
            field_name=field_name, threshold=threshold,
            threads=threads, verbose=verbose,
            save_alignment=False, filename=None, distance=distance,
            bits=bits, denominator=denominator, radius=radius,
            representation=representation, config=config
        )
    clusters_df = generate_clusters(df=df, field_name=field_name,
                                    sim_df=sim_df, verbose=verbose,
                                    threshold=threshold,
                                    cluster_algorithm=clustering_mode)
    representatives = clusters_df.cluster.unique()
    df = df[df.index.isin(representatives)]
    return df
