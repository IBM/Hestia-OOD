from multiprocessing import cpu_count
import os
import shutil
import subprocess
import time
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.sparse as spr
from concurrent.futures import ThreadPoolExecutor

from hestia.utils import BULK_SIM_METRICS


SUPPORTED_FPS = ['ecfp', 'mapc', 'maccs']


def sim_df2mtx(sim_df: pd.DataFrame,
               size_query: Optional[int] = None,
               size_target: Optional[int] = None,
               threshold: Optional[float] = 0.0,
               filter_smaller: Optional[bool] = True,
               boolean_out: Optional[bool] = True) -> spr.csr_matrix:
    """
    Converts a DataFrame of similarity scores into a sparse matrix representation, optionally filtering 
    based on a similarity threshold and producing a boolean or numerical output.

    :param sim_df: DataFrame containing similarity data with `query`, `target`, and `metric` columns.
    :type sim_df: pd.DataFrame
    :param size_query: Total number of unique query indices, defining the first dimension of the matrix. 
                       Defaults to the number of unique queries in `sim_df`.
    :type size_query: int, optional
    :param size_target: Total number of unique target indices, defining the second dimension of the matrix. 
                        Defaults to `size_query`, assuming a square matrix.
    :type size_target: int, optional
    :param threshold: Similarity score threshold for filtering. Defaults to 0.0.
    :type threshold: float, optional
    :param filter_smaller: If True, retains values above the threshold. If False, retains values below it.
    :type filter_smaller: bool, optional
    :param boolean_out: If True, converts output to boolean values, representing presence/absence of similarity. 
                        If False, retains original similarity values.
    :type boolean_out: bool, optional

    :return: Symmetric sparse matrix of filtered similarity scores, either in boolean or numerical format.
    :rtype: spr.csr_matrix
    """
    if size_query is None:
        size_query = len(sim_df['query'].unique())
    if size_target is None:
        size_target = size_query

    dtype = np.bool_ if boolean_out else sim_df.metric.dtype
    if filter_smaller:
        sim_df = sim_df[sim_df.metric > threshold]
    else:
        sim_df = sim_df[sim_df.metric < threshold]

    if dtype == np.float16:
        dtype = np.float32

    queries = sim_df['query'].to_numpy()
    targets = sim_df['target'].to_numpy()
    metrics = sim_df['metric'].to_numpy()
    if boolean_out:
        if filter_smaller:
            metrics[metrics > threshold] = True
        else:
            metrics[metrics < threshold] = True
    mtx = spr.coo_matrix((metrics, (queries, targets)),
                         shape=(size_query, size_target),
                         dtype=dtype)
    return mtx.maximum(mtx.transpose())


def embedding_similarity(
    query_embds: np.ndarray,
    target_embds: Optional[np.ndarray] = None,
    sim_function: Union[str, Callable] = 'cosine',
    threads: int = cpu_count(),
    threshold: float = 0.0,
    save_alignment: bool = False,
    filename: str = None,
    **kwargs
) -> pd.DataFrame:
    """
    Calculates pairwise similarity between embeddings in `query_embds` and `target_embds` using specified
    similarity functions. Supports parallel processing to handle large datasets efficiently.

    :param query_embds: Array of embeddings for the query set. Each row should represent a single embedding.
    :type query_embds: np.ndarray
    :param target_embds: Array of embeddings for the target set. If None, self-comparison of `query_embds` 
                         is performed.
    :type target_embds: np.ndarray, optional
    :param sim_function: Similarity function to use for pairwise comparison. Default is 'cosine'. Can be either 
                         a string specifying a built-in function or a custom callable.
    :type sim_function: Union[str, Callable], optional
    :param threads: Number of CPU threads for parallel processing. Defaults to the system CPU count.
    :type threads: int, optional
    :param threshold: Minimum similarity score required to include a pair in the results. Defaults to 0.0.
    :type threshold: float, optional
    :param save_alignment: If True, saves the alignment results to a compressed CSV file.
    :type save_alignment: bool, optional
    :param filename: Name for the output file if `save_alignment` is True. Defaults to a timestamp if None.
    :type filename: str, optional
    :param **kwargs: Additional keyword arguments for compatibility.

    :raises RuntimeError: If any exception occurs in a thread during similarity calculation.
    :raises KeyError: If the specified `sim_function` is not supported.

    :return: DataFrame with columns `query`, `target`, and `metric`, where each row represents a pairwise 
             similarity score above the specified `threshold`.
    :rtype: pd.DataFrame
    """
    if target_embds is None:
        target_embds = query_embds

    bulk_sim_metric = BULK_SIM_METRICS[sim_function]
    chunk_size = threads * 1_000
    chunks_target = (len(target_embds) // chunk_size) + 1
    queries, targets, metrics = [], [], []
    pbar = tqdm(range(len(query_embds)))

    with ThreadPoolExecutor(max_workers=threads) as executor:
        for chunk in pbar:
            jobs = []
            for chunk_t in range(chunks_target):
                start_t = chunk_t * chunk_size
                if chunk_t == chunks_target - 1:
                    end_t = -1
                else:
                    end_t = (chunk_t + 1) * chunk_size
                if end_t == -1:
                    chunk_fps = target_embds[start_t:]
                else:
                    chunk_fps = target_embds[start_t:end_t]

                query_fp = query_embds[chunk]
                job = executor.submit(bulk_sim_metric, query_fp, chunk_fps)
                jobs.append(job)

            for idx, job in enumerate(jobs):
                if job.exception() is not None:
                    raise RuntimeError(job.exception())
                result = job.result()
                for idx_target, metric in enumerate(result):
                    if metric < threshold:
                        continue
                    queries.append(int(chunk))
                    targets.append(int((idx * chunk_size) + idx_target))
                    metrics.append(metric)

    df = pd.DataFrame({'query': queries, 'target': targets,
                       'metric': metrics})
    if sim_function not in ['cosine-np']:
        df.metric = df.metric.map(lambda x: 1 / (1 + x))

    df = df[df['metric'] > threshold]
    if save_alignment:
        if filename is None:
            filename = time.time()
        df.to_csv(f'{filename}.csv.gz', index=False, compression='gzip')
    return df


def molecular_similarity(
    df_query: pd.DataFrame,
    df_target: pd.DataFrame = None,
    field_name: str = 'smiles',
    sim_function: str = 'tanimoto',
    fingerprint: str = 'ecfp',
    bits: int = 1024,
    radius: int = 2,
    threshold: float = 0.0,
    threads: int = cpu_count(),
    verbose: int = 0,
    save_alignment: bool = False,
    filename: str = None,
    **kwargs
) -> pd.DataFrame:
    """
    Calculates pairwise molecular similarity between query and target molecules using specified fingerprint
    and similarity functions. Uses RDKit for molecular fingerprinting and similarity calculations.

    :param df_query: DataFrame containing SMILES strings of query molecules. Each row should have a column 
                     specified by `field_name` with SMILES strings.
    :type df_query: pd.DataFrame
    :param df_target: DataFrame containing SMILES strings of target molecules. If None, self-comparison of 
                      `df_query` is performed.
    :type df_target: pd.DataFrame, optional
    :param field_name: Column name in `df_query` and `df_target` that contains SMILES strings. Defaults to 'smiles'.
    :type field_name: str, optional
    :param sim_function: Similarity function to use for pairwise comparison. Options include 'tanimoto', 'dice', 
                         'sokal', 'rogot-goldberg', and 'cosine'. Defaults to 'tanimoto'.
    :type sim_function: str, optional
    :param fingerprint: Type of fingerprint to use, options are 'ecfp' (Extended-Connectivity Fingerprint), 
                        'maccs' (MACCS keys), or 'mapc' (requires the mapchiral package). Defaults to 'ecfp'.
    :type fingerprint: str, optional
    :param bits: Size of the fingerprint bit vector, applicable to `ecfp`. Defaults to 1024.
    :type bits: int, optional
    :param radius: Radius for the ECFP fingerprint, applicable to `ecfp`. Defaults to 2.
    :type radius: int, optional
    :param threshold: Minimum similarity score required to include a pair in the results. Defaults to 0.0.
    :type threshold: float, optional
    :param threads: Number of CPU threads for parallel processing. Defaults to the system CPU count.
    :type threads: int, optional
    :param verbose: Verbosity level, where higher values increase output detail. Defaults to 0.
    :type verbose: int, optional
    :param save_alignment: If True, saves the alignment results to a compressed CSV file.
    :type save_alignment: bool, optional
    :param filename: Name for the output file if `save_alignment` is True. Defaults to a timestamp if None.
    :type filename: str, optional
    :param **kwargs: Additional keyword arguments for compatibility.

    :raises ImportError: If RDKit (or mapchiral, if used with 'mapc') is not installed.
    :raises ValueError: If `field_name` is missing from `df_query` or `df_target`.
    :raises NotImplementedError: If `sim_function` is not supported by the function.

    :return: DataFrame with columns `query`, `target`, and `metric`, where each row represents a pairwise 
             similarity score above the specified `threshold`.
    :rtype: pd.DataFrame
    """

    try:
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors
        from rdkit.DataStructs import (
            BulkTanimotoSimilarity, BulkDiceSimilarity,
            BulkSokalSimilarity, BulkRogotGoldbergSimilarity,
            BulkCosineSimilarity
        )
    except ModuleNotFoundError:
        raise ImportError("This function requires RDKit to be installed.")

    BULK_SIM_METRICS.update({
        'tanimoto': BulkTanimotoSimilarity,
        'dice': BulkDiceSimilarity,
        'sokal': BulkSokalSimilarity,
        'rogot-goldberg': BulkRogotGoldbergSimilarity,
        'cosine': BulkCosineSimilarity
    })

    if fingerprint == 'ecfp':
        fpgen = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=bits
        )

        def _get_fp(smile: str):
            mol = Chem.MolFromSmiles(smile)
            if sim_function in ['dice', 'tanimoto', 'sokal', 'rogot-goldberg',
                                'cosine']:
                fp = fpgen.GetFingerprint(mol)
            else:
                fp = fpgen.GetFingerprintAsNumPy(mol).astype(np.int8)
            return fp

    elif fingerprint == 'maccs':

        def _get_fp(smile: str):
            mol = Chem.MolFromSmiles(smile)
            if sim_function in ['dice', 'tanimoto', 'sokal', 'rogot-goldberg',
                                'cosine']:
                fp = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
            else:
                fp = fpgen.GetFingerprintAsNumPy(mol).astype(np.int8)
            return fp

    elif fingerprint == 'mapc':
        try:
            from mapchiral.mapchiral import encode
        except ModuleNotFoundError:
            raise ImportError('This fingerprint requires mapchiral to be installed: `pip install mapchiral`')

        def _get_fp(smile: str):
            mol = Chem.MolFromSmiles(smile, sanitize=True)
            fp = encode(mol, max_radius=radius,
                        n_permutations=bits, mapping=False)
            return fp

        if sim_function != 'jaccard':
            raise ValueError('MAPc can only be used with `jaccard`.')

    if sim_function in BULK_SIM_METRICS:
        bulk_sim_metric = BULK_SIM_METRICS[sim_function]
    else:
        raise NotImplementedError(
            f'Distance metric: {sim_function} not implemented. ' +
            f"Supported metrics: {', '.join(BULK_SIM_METRICS.keys())}"
        )

    def _parallel_fps(mols: List[str], mssg: str) -> list:
        fps = []
        jobs = []
        with ThreadPoolExecutor(max_workers=threads) as executor:
            for mol in mols:
                job = executor.submit(_get_fp, mol)
                jobs.append(job)
            pbar = tqdm(jobs, desc=mssg)
            for job in pbar:
                if job.exception() is not None:
                    raise RuntimeError(job.exception())
                result = job.result()
                fps.append(result)
        pbar.close()
        return fps

    if (field_name not in df_query.columns):
        raise ValueError(f'{field_name} not found in query DataFrame')
    if df_target is not None and field_name not in df_target.columns:
        raise ValueError(f'{field_name} not found in target DataFrame')

    if verbose > 1:
        print(f'Calculating molecular similarities using {fingerprint}-{radius * 2}',
              f'with {bits:,} bits and {sim_function} index...')\

    query_mols = df_query[field_name].tolist()
    chunk_size = threads * 1_000
    query_fps = _parallel_fps(query_mols, 'Query FPs')

    if df_target is None:
        df_target = df_query
        target_fps = query_fps
    else:
        target_fps = _parallel_fps(df_target[field_name], 'Target FPs')

    if fingerprint == 'mapc':
        query_fps = np.stack(query_fps)
        target_fps = np.stack(target_fps)
    if isinstance(query_fps, np.ndarray):
        max_complex = query_fps.shape[0] * target_fps.shape[0]
        query_size = query_fps.shape[0]
    else:
        max_complex = len(query_fps) * len(target_fps)
        query_size = len(query_fps)

    chunks_target = (len(df_target) // chunk_size) + 1
    metrics = np.zeros((max_complex), dtype=np.float16)

    if max_complex < 1e5:
        index_type = np.uint16
    elif max_complex < 1e9:
        index_type = np.uint32
    elif max_complex < 1e19:
        index_type = np.uint64
    else:
        index_type = np.uint128

    queries = np.zeros_like(metrics, dtype=index_type)
    targets = np.zeros_like(metrics, dtype=index_type)
    if verbose > 1:
        print()
        pbar = tqdm(range(query_size), desc='Similarity calculation')
    else:
        pbar = range(query_size)

    with ThreadPoolExecutor(max_workers=threads) as executor:
        for chunk in pbar:
            jobs = []
            for chunk_t in range(chunks_target):
                start_t = chunk_t * chunk_size
                if chunk_t == chunks_target - 1:
                    end_t = -1
                else:
                    end_t = (chunk_t + 1) * chunk_size
                if end_t == -1:
                    chunk_fps = target_fps[start_t:]
                else:
                    chunk_fps = target_fps[start_t:end_t]

                query_fp = query_fps[chunk]

                job = executor.submit(bulk_sim_metric, query_fp, chunk_fps)
                jobs.append(job)

            for idx, job in enumerate(jobs):
                if job.exception() is not None:
                    raise RuntimeError(job.exception())
                result = job.result()
                for idx_target, metric in enumerate(result):
                    target_pointer = int((idx * chunk_size) + idx_target)
                    query_pointer = int(chunk)
                    pointer = (query_pointer * query_size) + target_pointer
                    if metric < threshold:
                        continue
                    queries[pointer] = query_pointer
                    targets[pointer] = target_pointer
                    metrics[pointer] = metric

    mask = metrics > threshold
    queries = queries[mask]
    targets = targets[mask]
    metrics = metrics[mask]

    df = pd.DataFrame({'query': queries, 'target': targets, 'metric': metrics})
    df = df[df['metric'] > threshold]
    if save_alignment:
        if filename is None:
            filename = time.time()
        df.to_csv(f'{filename}.csv.gz', index=False, compression='gzip')
    return df


def protein_structure_similarity(
    df_query: pd.DataFrame,
    df_target: pd.DataFrame = None,
    field_name: str = 'structure',
    prefilter: bool = True,
    denominator: str = 'shortest',
    representation: str = '3di+aa',
    threshold: float = 0.0,
    threads: int = cpu_count(),
    verbose: int = 0,
    save_alignment: bool = False,
    filename: str = None,
    **kwargs
) -> Union[pd.DataFrame, np.ndarray]:   
    """
    Calculates pairwise structural similarity between query and target protein structures using Foldseek.
    Supports alignment based on various representations, including 3D alignment, TM alignment, and 
    combined 3D and amino acid alignments.

    :param df_query: DataFrame containing query protein structures. Each row should have a column 
                     specified by `field_name` with paths to PDB files.
    :type df_query: pd.DataFrame
    :param df_target: DataFrame containing target protein structures, with each row holding paths to 
                      PDB files in `field_name`. If None, self-comparison of `df_query` is performed.
    :type df_target: pd.DataFrame, optional
    :param field_name: Column name in `df_query` and `df_target` with paths to PDB structure files.
                       Defaults to 'structure'.
    :type field_name: str, optional
    :param prefilter: Enables prefiltering to reduce computation. Defaults to True.
    :type prefilter: bool, optional
    :param denominator: Determines similarity normalization, using "shortest" (default), "longest", 
                        or the number of aligned residues (`n_aligned`).
    :type denominator: str, optional
    :param representation: Alignment representation mode, with options '3di', 'TM', or '3di+aa'. 
                           Defaults to '3di+aa'.
    :type representation: str, optional
    :param threshold: Minimum similarity metric required to include an alignment in the results. Defaults to 0.0.
    :type threshold: float, optional
    :param threads: Number of CPU threads for parallel processing. Defaults to system CPU count.
    :type threads: int, optional
    :param verbose: Verbosity level for process logging, where higher values increase output detail.
    :type verbose: int, optional
    :param save_alignment: If True, saves alignment results to a compressed CSV file.
    :type save_alignment: bool, optional
    :param filename: Name for the output file if `save_alignment` is True. Defaults to a timestamp if None.
    :type filename: str, optional
    :param **kwargs: Additional keyword arguments for compatibility.

    :raises ImportError: If Foldseek is not installed or accessible in the system PATH.
    :raises ValueError: If `field_name` is missing from `df_query` or `df_target`.

    :return: DataFrame with columns `query`, `target`, and `metric`, where each row represents an alignment 
             with a similarity metric above `threshold`. Returns the metric value determined by `representation`.
    :rtype: Union[pd.DataFrame, np.ndarray]
    """
    if shutil.which('foldseek') is None:
        mssg = "Foldseek not found. Please install following the instructions"
        mssg += " in: https://github.com/IBM/Hestia-OOD#installation"
        raise ImportError(mssg)
    else:
        foldseek = 'foldseek'

    from hestia.utils import _collect_pdb_files

    if (field_name not in df_query.columns):
        raise ValueError(f'{field_name} not found in query DataFrame')
    if df_target is not None and field_name not in df_target.columns:
        raise ValueError(f'{field_name} not found in target DataFrame')

    ALIGNMENT_DICT = {'3di': '0', 'TM': '1', '3di+aa': '2'}
    DENOMINATOR_DICT = {'n_aligned': '0', 'shortest': '1', 'longest': '2'}

    if df_target is None:
        df_target = df_query
    tmp_dir = f'hestia_tmp_{time.time()}'
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.mkdir(tmp_dir)
    if verbose > 2:
        mmseqs_v = 3
    else:
        mmseqs_v = verbose
    if verbose > 0:
        if prefilter:
            print('Calculating pairwise alignments using Foldseek',
                  'algorithm with prefilter...')
        else:
            print('Calculating pairwise alignments using Foldseek',
                  'algorithm...')
    db_query_dir = os.path.join(tmp_dir, 'query_dir')
    db_query = os.path.join(tmp_dir, 'query_db')
    db_target_dir = os.path.join(tmp_dir, 'target_dir')
    db_target = os.path.join(tmp_dir, 'target_db')
    alignment_db = os.path.join(tmp_dir, 'align_db')
    alignment_csv = os.path.join(tmp_dir, 'align_db.csv')

    if os.path.isdir(db_query_dir):
        shutil.rmtree(db_query_dir)
    if os.path.isdir(db_target_dir):
        shutil.rmtree(db_target_dir)
    os.mkdir(db_query_dir)
    os.mkdir(db_target_dir)

    _collect_pdb_files(df_query[field_name], db_query_dir)
    _collect_pdb_files(df_target[field_name], db_target_dir)

    subprocess.run([foldseek, 'createdb', db_query_dir, db_query,
                    '-v',  str(mmseqs_v)])
    subprocess.run([foldseek, 'createdb', db_target_dir, db_target,
                    '-v',  str(mmseqs_v)])
    denominator = DENOMINATOR_DICT[denominator]
    representation = ALIGNMENT_DICT[representation]
    prefilter = '0' if prefilter else '2'

    subprocess.run([foldseek, 'search', db_query, db_target, alignment_db,
                    'tmp', '-s', '9.5', '-a', '-e', 'inf',
                    '--seq-id-mode', denominator, '--threads',
                    str(threads), '--alignment-type', representation,
                    '--prefilter-mode', prefilter, '-v', str(mmseqs_v)
                    ])
    subprocess.run([foldseek, 'convertalis', db_query, db_target,
                    alignment_db, alignment_csv, '--format-mode', '4',
                    '--threads', str(threads), '--format-output',
                    'query,target,fident,alnlen,qlen,tlen,prob,alntmscore',
                    '-v', str(mmseqs_v)])

    df = pd.read_csv(alignment_csv, sep='\t')
    qry2idx = {os.path.basename(qry).split('.pdb')[0]: idx for idx, qry in
               enumerate(df_query[field_name].unique())}
    tgt2idx = {os.path.basename(tgt).split('.pdb')[0]: idx for idx, tgt in
               enumerate(df_target[field_name].unique())}

    if save_alignment:
        if filename is None:
            filename = time.time()
        df.to_csv(f'{filename}.csv.gz', index=False, compression='gzip')

    if representation.lower() == 'tm':
        df['metric'] = df['alntmscore']
    else:
        df['metric'] = df['prob']

    df['query'] = df['query'].map(lambda x: qry2idx[
        x.split('.pdb')[0].split('_')[0]])
    df['query'] = df['query'].astype(int)
    df['target'] = df['target'].map(lambda x: tgt2idx[
        x.split('.pdb')[0].split('_')[0]])
    df['target'] = df['target'].astype(int)
    df = df[df['metric'] > threshold]
    shutil.rmtree(tmp_dir)
    return df


def sequence_similarity_peptides(
    df_query: pd.DataFrame,
    df_target: Optional[pd.DataFrame] = None,
    field_name: Optional[str] = 'sequence',
    denominator: Optional[str] = 'shortest',
    threads: Optional[int] = cpu_count(),
    threshold: Optional[float] = 0.0,
    verbose: Optional[int] = 0,
    save_alignment: Optional[bool] = False,
    filename: Optional[int] = None
) -> pd.DataFrame:
    """
    Calculates pairwise sequence similarity between query and target peptide sequences using MMSeqs2.
    Sequences are divided into "small," "medium," and "normal" categories based on length, and 
    each category is aligned with a specific method for optimal recall.

    - _small_alignment: For sequences with 8 or fewer residues, checks if one sequence is a subsequence of the other.
    - _medium_alignment: For sequences between 9 and 20 residues, uses a lower threshold with MMSeqs2 to filter alignments.
    - _normal_alignment: For sequences longer than 20 residues, performs full alignments with MMSeqs2.

    :param df_query: DataFrame containing the query peptide sequences. Each row should have a column 
                     specified by `field_name` with peptide sequence strings.
    :type df_query: pd.DataFrame
    :param df_target: DataFrame with target peptide sequences, where each row has a `field_name` column 
                      containing sequence strings. If None, `df_query` will be used for self-comparisons.
    :type df_target: pd.DataFrame, optional
    :param field_name: Column name in `df_query` and `df_target` holding the sequence data to be aligned.
                       Defaults to 'sequence'.
    :type field_name: str, optional
    :param denominator: Determines how similarity is calculated, using either "shortest" (default), 
                        "longest", or the number of aligned residues (`n_aligned`).
    :type denominator: str, optional
    :param threads: Number of threads for parallel processing. Defaults to system CPU count.
    :type threads: int, optional
    :param threshold: Minimum similarity metric for alignment entries to be included in the output. Defaults to 0.0.
    :type threshold: float, optional
    :param verbose: Verbosity level, where 0 is silent and higher levels increase detail in logging.
    :type verbose: int, optional
    :param save_alignment: If True, saves the resulting DataFrame to a compressed CSV file.
    :type save_alignment: bool, optional
    :param filename: Filename for saving the alignment results if `save_alignment` is True. 
                     If None, a timestamp is used as the filename.
    :type filename: str, optional

    :raises RuntimeError: If MMSeqs2 is not installed or is unavailable in the system PATH.
    :raises ValueError: If `field_name` is not found in `df_query` or `df_target`.

    :return: DataFrame with columns `query`, `target`, and `metric`, where each row represents 
             an alignment result with similarity metric above the `threshold`.
    :rtype: pd.DataFrame
    """
    if shutil.which('mmseqs') is None:
        raise RuntimeError(
            "MMSeqs2 not found. Please install following the instructions in:",
            "https://github.com/IBM/Hestia-OOD#installation"
        )
    from hestia.utils.file_format import _write_fasta

    def _small_alignment(df_query, df_target, field_name, denominator) -> pd.DataFrame:
        proto_df = []
        for name, seq in zip(df_query.index.tolist(), df_query[field_name]):
            for name2, seq2 in zip(df_target.index.tolist(), df_target[field_name]):
                if seq in seq2:
                    if denominator == '0':
                        metric = 1.0
                    elif denominator == '1':
                        metric = 1.0,
                    elif denominator == '2':
                        metric = len(seq) / len(seq2)
                    proto_df.append({
                        'query': name,
                        'target': name2,
                        'metric': metric
                    })
        return pd.DataFrame(proto_df)

    def _normal_alignment(df_query, df_target, tmp_dir,
                          field_name, dbtype, denominator,
                          mmseqs_v, threads) -> pd.DataFrame:
        db_query_file = os.path.join(tmp_dir, 'db_query.fasta')
        db_target_file = os.path.join(tmp_dir, 'db_target.fasta')
        _write_fasta(df_query[field_name].tolist(), df_query.index.tolist(),
                     db_query_file)
        _write_fasta(df_target[field_name].tolist(), df_target.index.tolist(),
                     db_target_file)

        subprocess.run(['mmseqs', 'createdb', '--dbtype',
                        dbtype, db_query_file, '-v', '1',
                        f'{tmp_dir}/db_query'])
        subprocess.run(['mmseqs', 'createdb', '--dbtype',
                        dbtype, db_target_file, '-v', '1',
                        f'{tmp_dir}/db_target'])

        subprocess.run(['mmseqs', 'search',  f'{tmp_dir}/db_query',
                        f'{tmp_dir}/db_target', f'{tmp_dir}/align_db',
                        f'{tmp_dir}/tmp', '--alignment-mode', '3',
                        '--seq-id-mode', denominator, '--search-type', '1',
                        '--prefilter-mode', '2', '-s', '7.5',
                        '-v', str(mmseqs_v),
                        '--threads', str(threads)])

        file = os.path.join(tmp_dir, 'alignments.tab')
        subprocess.run(['mmseqs', 'convertalis', f'{tmp_dir}/db_query',
                        f'{tmp_dir}/db_target', f'{tmp_dir}/align_db',
                        '--format-mode', '4', '--threads', str(threads),
                        file, '-v', '1'])

        df = pd.read_csv(file, sep='\t')
        return df

    def _medium_alignment(df_query, df_target, tmp_dir,
                          field_name, dbtype, denominator,
                          mmseqs_v, threads):
        db_query_file = os.path.join(tmp_dir, 'db_query.fasta')
        db_target_file = os.path.join(tmp_dir, 'db_target.fasta')
        _write_fasta(df_query[field_name].tolist(), df_query.index.tolist(),
                     db_query_file)
        _write_fasta(df_target[field_name].tolist(), df_target.index.tolist(),
                     db_target_file)

        subprocess.run(['mmseqs', 'createdb', '--dbtype',
                        dbtype, db_query_file, '-v', '1',
                        f'{tmp_dir}/db_query'])
        subprocess.run(['mmseqs', 'createdb', '--dbtype',
                        dbtype, db_target_file, '-v', '1',
                        f'{tmp_dir}/db_target'])

        subprocess.run(['mmseqs', 'search', f'{tmp_dir}/db_query',
                        f'{tmp_dir}/db_target', f'{tmp_dir}/align_db',
                        f'{tmp_dir}/tmp', '--alignment-mode', '3',
                        '--seq-id-mode', denominator, '--search-type', '1',
                        '--prefilter-mode', '2', '-s', '2',
                        '-v', str(mmseqs_v),
                        '--threads', str(threads), '--mask', '0',
                        '--comp-bias-corr', '0',
                        '-e', '1e7'])

        file = os.path.join(tmp_dir, 'alignments.tab')
        subprocess.run(['mmseqs', 'convertalis', f'{tmp_dir}/db_query',
                        f'{tmp_dir}/db_target', f'{tmp_dir}/align_db',
                        '--format-mode', '4', '--threads', str(threads),
                        file, '-v', '1'])

        df = pd.read_csv(file, sep='\t')
        return df

    DENOMINATOR_DICT = {'n_aligned': '0', 'shortest': '1', 'longest': '2'}
    tmp_dir = f'hestia_tmp_{time.time()}'
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.mkdir(tmp_dir)

    df_query['seq_len'] = df_query[field_name].map(len)
    if df_target is None:
        df_target = df_query
    normal_df_query = df_query[df_query['seq_len'] > 20]
    medium_df_query = df_query[(df_query['seq_len'] > 8) &
                               (df_query['seq_len'] <= 20)]
    small_df_query = df_query[df_query['seq_len'] <= 8]
    dbtype = '1'
    denominator = DENOMINATOR_DICT[denominator]

    if verbose > 2:
        mmseqs_v = 3
    else:
        mmseqs_v = verbose

    if verbose > 0:
        print('Calculating pairwise alignments using MMSeqs2 algorithm',
                'with `peptide` mode...')

    if len(normal_df_query) > 0:
        normal_simdf = _normal_alignment(
            df_query=normal_df_query, df_target=df_target, tmp_dir=tmp_dir,
            field_name=field_name, dbtype=dbtype, denominator=denominator,
            mmseqs_v=mmseqs_v, threads=threads
        )
    else:
        normal_simdf = pd.DataFrame()
    if len(medium_df_query) > 0:
        medium_simdf = _medium_alignment(
            df_query=medium_df_query, df_target=df_target, tmp_dir=tmp_dir,
            field_name=field_name, dbtype=dbtype, denominator=denominator,
            mmseqs_v=mmseqs_v, threads=threads
        )
    else:
        medium_simdf = pd.DataFrame()
    if len(small_df_query) > 0:
        small_simdf = _small_alignment(
            df_query=small_df_query, df_target=df_target, field_name=field_name,
            denominator=denominator
        )
    else:
        small_simdf = pd.DataFrame()

    df = pd.concat([normal_simdf, medium_simdf, small_simdf])
    df['metric'] = df['fident']
    df = df[df['metric'] > threshold]
    df = df[['query', 'target', 'metric']]
    if save_alignment:
        if filename is None:
            filename = time.time()
        df.to_csv(f'{filename}.csv.gz', index=False, compression='gzip')

    shutil.rmtree(tmp_dir)
    return df


def sequence_similarity_mmseqs(
    df_query: pd.DataFrame,
    df_target: pd.DataFrame = None,
    field_name: str = 'sequence',
    prefilter: bool = True,
    denominator: str = 'shortest',
    threads: int = cpu_count(),
    is_nucleotide: bool = False,
    threshold: float = 0.0,
    verbose: int = 0,
    save_alignment: bool = False,
    filename: str = None
) -> pd.DataFrame:
    """
    Calculate pairwise sequence similarity between query and target sequences using MMSeqs2, 
    with optional prefiltering for efficiency. Designed for parallel execution and customizable 
    alignment parameters.

    :param df_query: DataFrame containing the query sequences. Each row should have a column 
                     specified by `field_name` with sequence strings.
    :type df_query: pd.DataFrame
    :param df_target: DataFrame with target sequences, where each row has a `field_name` column 
                      containing sequence strings. If None, `df_query` will be used for self-comparisons.
    :type df_target: pd.DataFrame, optional
    :param field_name: Column name in `df_query` and `df_target` holding the sequence data to be aligned.
                       Defaults to 'sequence'.
    :type field_name: str, optional
    :param prefilter: If True, performs an initial filtering step to reduce the number of comparisons.
    :type prefilter: bool, optional
    :param denominator: Determines how similarity is calculated, using either "shortest" (default), 
                        "longest", or the number of aligned residues (`n_aligned`).
    :type denominator: str, optional
    :param threads: Number of threads for parallel processing. Defaults to system CPU count.
    :type threads: int, optional
    :param is_nucleotide: Set to True if sequences are nucleotide-based. Defaults to False (for protein sequences).
    :type is_nucleotide: bool, optional
    :param threshold: Minimum similarity metric for alignment entries to be included in the output. Defaults to 0.0.
    :type threshold: float, optional
    :param verbose: Verbosity level, where 0 is silent and higher levels increase detail in logging.
    :type verbose: int, optional
    :param save_alignment: If True, saves the resulting DataFrame to a compressed CSV file.
    :type save_alignment: bool, optional
    :param filename: Filename for saving the alignment results if `save_alignment` is True.
                     If None, a timestamp is used as the filename.
    :type filename: str, optional

    :raises RuntimeError: If MMSeqs2 is not installed or is unavailable in the system PATH.
    :raises ValueError: If `field_name` is not found in `df_query` or `df_target`.

    :return: DataFrame with columns `query`, `target`, and `metric`, where each row represents
             an alignment result with similarity metric above the `threshold`.
    :rtype: pd.DataFrame
    """

    if shutil.which('mmseqs') is None:
        raise RuntimeError(
            "MMSeqs2 not found. Please install following the instructions in:",
            "https://github.com/IBM/Hestia-OOD#installation"
        )
    from hestia.utils.file_format import _write_fasta

    if (field_name not in df_query.columns):
        raise ValueError(f'{field_name} not found in query DataFrame')
    if df_target is not None and field_name not in df_target.columns:
        raise ValueError(f'{field_name} not found in target DataFrame')

    DENOMINATOR_DICT = {'n_aligned': '0', 'shortest': '1', 'longest': '2'}

    tmp_dir = f'hestia_tmp_{time.time()}'
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.mkdir(tmp_dir)

    if verbose > 2:
        mmseqs_v = 3
    else:
        mmseqs_v = verbose

    if verbose > 0:
        if prefilter:
            print('Calculating pairwise alignments using MMSeqs2 algorithm',
                  'with prefilter...')
        else:
            print('Calculating pairwise alignments using MMSeqs2 algorithm...')

    if df_target is None:
        df_target = df_query

    db_query_file = os.path.join(tmp_dir, 'db_query.fasta')
    db_target_file = os.path.join(tmp_dir, 'db_target.fasta')
    _write_fasta(df_query[field_name].tolist(), df_query.index.tolist(),
                 db_query_file)
    _write_fasta(df_target[field_name].tolist(), df_target.index.tolist(),
                 db_target_file)

    dbtype = '2' if is_nucleotide else '1'
    subprocess.run(['mmseqs', 'createdb', '--dbtype',
                    dbtype, db_query_file, '-v', '1',
                    f'{tmp_dir}/db_query'])
    subprocess.run(['mmseqs', 'createdb', '--dbtype',
                    dbtype, db_target_file, '-v', '1',
                    f'{tmp_dir}/db_target'])

    if is_nucleotide or prefilter:
        subprocess.run(['mmseqs', 'prefilter', '-s',
                        '6', f'{tmp_dir}/db_query',
                        f'{tmp_dir}/db_target',
                        f'{tmp_dir}/pref', '-v',
                        f'{mmseqs_v}'])
    else:
        program_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'utils', 'mmseqs_fake_prefilter.sh'
        )
        subprocess.run([program_path,
                        f'{tmp_dir}/db_query', f'{tmp_dir}/db_target',
                        f'{tmp_dir}/pref', 'db_query'])

    denominator = DENOMINATOR_DICT[denominator]
    subprocess.run(['mmseqs', 'align',  f'{tmp_dir}/db_query',
                    f'{tmp_dir}/db_target', f'{tmp_dir}/pref',
                    f'{tmp_dir}/align_db', '--alignment-mode', '3',
                    '-e', '1e2', '--seq-id-mode', denominator,
                    '--cov-mode', '5', '-c', '0.7',
                    '-v', f'{mmseqs_v}', '--threads', f'{threads}'])

    file = os.path.join(tmp_dir, 'alignments.tab')
    subprocess.run(['mmseqs', 'convertalis', f'{tmp_dir}/db_query',
                    f'{tmp_dir}/db_target', f'{tmp_dir}/align_db',
                    '--format-mode', '4', '--threads', f'{threads}',
                    file, '-v', '1'])

    df = pd.read_csv(file, sep='\t')
    df['metric'] = df['fident']
    df = df[df['metric'] > threshold]
    df = df[['query', 'target', 'metric']]
    if save_alignment:
        if filename is None:
            filename = time.time()
        df.to_csv(f'{filename}.csv.gz', index=False, compression='gzip')

    shutil.rmtree(tmp_dir)
    return df


def sequence_similarity_needle(
    df_query: pd.DataFrame,
    df_target: pd.DataFrame = None,
    field_name: str = 'sequence',
    denominator: str = 'shortest',
    is_nucleotide: bool = False,
    config: dict = None,
    threshold: float = 0.0,
    threads: int = cpu_count(),
    verbose: int = 0,
    save_alignment: bool = False,
    filename: str = None
) -> pd.DataFrame:
    """
    Calculate pairwise sequence similarity between query and target sequences using the 
    EMBOSS `needleall` tool. This function is designed for efficient parallel processing 
    and supports custom alignment parameters.

    :param df_query: DataFrame containing the query sequences. Each row should have a column 
                     specified by `field_name` that contains sequence strings.
    :type df_query: pd.DataFrame
    :param df_target: DataFrame containing the target sequences, with a column specified by 
                      `field_name` containing sequence strings. If None, `df_query` will 
                      be used as the target DataFrame, performing self-comparisons.
    :type df_target: pd.DataFrame, optional
    :param field_name: Name of the column in `df_query` and `df_target` containing the 
                       sequence data to be compared. Defaults to 'sequence'.
    :type field_name: str, optional
    :param denominator: Determines how similarity is calculated; options are "shortest" 
                        (default), "longest", or "average" sequence length between pairs.
    :type denominator: str, optional
    :param is_nucleotide: Indicates if the sequences are nucleotide sequences. If False, 
                          assumes sequences are protein-based.
    :type is_nucleotide: bool, optional
    :param config: Dictionary of EMBOSS `needleall` alignment parameters, such as 
                   `gapopen`, `gapextend`, etc. If None, default configuration is used.
    :type config: dict, optional
    :param threshold: Minimum similarity metric required for alignment entries to be included 
                      in the output. Defaults to 0.0.
    :type threshold: float, optional
    :param threads: Number of threads to use for parallel processing. Defaults to system 
                    CPU count.
    :type threads: int, optional
    :param verbose: Verbosity level of function output; 0 is silent, higher numbers increase 
                    output detail.
    :type verbose: int, optional
    :param save_alignment: If True, saves the resulting DataFrame to a compressed CSV file.
    :type save_alignment: bool, optional
    :param filename: Filename for saving the alignment results if `save_alignment` is True. 
                     If None, a timestamp will be used as the filename.
    :type filename: str, optional

    :raises ImportError: Raised if the `needleall` tool from EMBOSS is not installed.
    :raises ValueError: Raised if `field_name` is missing from `df_query` or `df_target`.
    :raises RuntimeError: Raised if any alignment job encounters an exception during processing.

    :return: DataFrame with columns `query`, `target`, and `metric`, where each row 
             represents a sequence alignment result, filtered by the specified `threshold`.
    :rtype: pd.DataFrame
    """
    if shutil.which("needleall") is None:
        raise ImportError("EMBOSS needleall not found. Please install by ",
                          "running: `conda install emboss -c bioconda`")
    from hestia.utils.file_format import _write_fasta_chunks
    from concurrent.futures import ThreadPoolExecutor

    if (field_name not in df_query.columns):
        raise ValueError(f'{field_name} not found in query DataFrame')
    if df_target is not None and field_name not in df_target.columns:
        raise ValueError(f'{field_name} not found in target DataFrame')

    if config is None:
        config = {
            "gapopen": 10,
            "gapextend": 0.5,
            "endweight": True,
            "endopen": 10,
            "endextend": 0.5,
            "matrix": "EBLOSUM62"
        }

    tmp_dir = f"hestia_tmp_{time.time()}"
    db_query = os.path.join(tmp_dir, 'db_query')
    db_target = os.path.join(tmp_dir, 'db_target')
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.mkdir(tmp_dir)
    os.mkdir(db_query)
    os.mkdir(db_target)

    if df_target is None:
        df_target = df_query

    n_query = len(df_query)
    df_query['tmp_id'] = [i for i in range(n_query)]
    df_target['tmp_id'] = [j + n_query for j in range(len(df_target))]

    all_seqs = df_query[field_name].tolist() + df_target[field_name].tolist()
    query_id2idx = {s_id: idx for idx, s_id
                    in enumerate(df_query.tmp_id)}
    target_idx2id = {s_id: idx for idx, s_id
                     in enumerate(df_target.tmp_id)}
    all_ids = sorted(query_id2idx.keys()) + sorted(target_idx2id.keys())
    seq_lengths = {str(s_id): len(seq) for s_id, seq in zip(all_ids, all_seqs)}
    del all_seqs, all_ids
    jobs_query = _write_fasta_chunks(df_query[field_name].tolist(),
                                     df_query.tmp_id.tolist(),
                                     threads, db_query)
    jobs_target = _write_fasta_chunks(df_target[field_name].tolist(),
                                      df_target.tmp_id.tolist(),
                                      threads, db_target)
    jobs = []
    proto_df = []

    with ThreadPoolExecutor(max_workers=threads) as executor:
        for i in range(jobs_query):
            for j in range(jobs_target):
                query = os.path.join(db_query, f"{i}.fasta.tmp")
                target = os.path.join(db_target, f"{j}.fasta.tmp")
                job = executor.submit(_compute_needle, query, target,
                                      threshold, denominator, is_nucleotide,
                                      seq_lengths, **config)
                jobs.append(job)

        if verbose > 1:
            pbar = tqdm(jobs)
        else:
            pbar = jobs

        for job in pbar:
            if job.exception() is not None:
                raise RuntimeError(job.exception())

            result = job.result()
            for query, target, metric in result:
                entry = {
                    'query': query_id2idx[int(query)],
                    'target': target_idx2id[int(target)],
                    'metric': metric
                }
                proto_df.append(entry)

    df = pd.DataFrame(proto_df)
    if save_alignment:
        if filename is None:
            filename = time.time()
        df.to_csv(f'{filename}.csv.gz', index=False, compression='gzip')
    shutil.rmtree(tmp_dir)
    df = df[df['metric'] > threshold]
    return df


def _compute_needle(
    query: str,
    target: str,
    threshold: float,
    denominator: str,
    is_nucleotide: bool,
    seq_lengths: dict,
    gapopen: float = 10,
    gapextend: float = 0.5,
    endweight: bool = True,
    endopen: float = 10,
    endextend: float = 0.5,
    matrix: str = 'EBLOSUM62'
):
    if is_nucleotide:
        type_1, type_2, = '-snucleotide1', '-snucleotide2'
    else:
        type_1, type_2 = '-sprotein1', '-sprotein2'

    FIDENT_CALCULATION = {
        'n_aligned': lambda x: float(x.split('(')[1][:-3])/100,
        'shortest': lambda x, q, t: int(x[11:].split('/')[0]) / min(q, t),
        'longest': lambda x, q, t: int(x[11:].split('/')[0]) / max(q, t)
    }[denominator]

    command = ["needleall", "-auto", "-stdout",
               "-aformat", "pair",
               "-gapopen", str(gapopen),
               "-gapextend", str(gapextend),
               "-endopen", str(endopen),
               "-endextend", str(endextend),
               "-datafile", matrix,
               type_1, type_2, query, target]

    if endweight:
        command.append("-endweight")

    result = []

    with subprocess.Popen(
        command, stdout=subprocess.PIPE,
        bufsize=1, universal_newlines=True
    ) as process:
        for idx, line in enumerate(process.stdout):
            if line.startswith('# 1:'):
                query = line[5:].split()[0].split('|')[0]

            elif line.startswith('# 2:'):
                target = line[5:].split()[0].split('|')[0]

            elif line.startswith('# Identity:'):
                fident = FIDENT_CALCULATION(
                    line, seq_lengths[query],
                    seq_lengths[target]
                )
            elif line.startswith('# Gaps:'):
                if (fident < threshold or query == target):
                    continue
                result.append((query, target, fident))
    return result
