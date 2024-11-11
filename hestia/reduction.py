import pandas as pd

from hestia.clustering import generate_clusters


def similarity_reduction(
    df: pd.DataFrame,
    sim_df: pd.DataFrame,
    field_name: str,
    clustering_mode: str = "CDHIT",
    threshold: float = 0.3,
    verbose: int = 2,
) -> pd.DataFrame:
    if 'connected_components' == clustering_mode:
        mssg = 'Similarity reduction is not'
        mssg += ' implemented for connected components.'
        raise NotImplementedError(mssg)

    clusters_df = generate_clusters(df=df, field_name=field_name,
                                    sim_df=sim_df, verbose=verbose,
                                    threshold=threshold,
                                    cluster_algorithm=clustering_mode)
    representatives = clusters_df.cluster.unique()
    df = df[df.index.isin(representatives)]
    return df
