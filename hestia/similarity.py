from multiprocessing import cpu_count
import os
import shutil
import subprocess
import time
from typing import List, Union

import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.sparse as spr


def sim_df2mtx(sim_df: pd.DataFrame,
               threshold: float = 0.05) -> spr.bsr_matrix:
    """Generates a similarity matrix from
    a DataFrame with the results from similarity
    calculations in the form of `query`, `target`,
    and `metric`.

    :param sim_df: DataFrame with similarity calculations
    with the columns `query`, `target`, and `metric`.
    :type sim_df: pd.DataFrame
    :param threshold: Similarity threshold below which
    elements are considered dissimilar, defaults to 0.05
    :type threshold: float, optional
    :return: Sparse similarity matrix with shape nxn where
    n are the unique elements in the `query` column.
    :rtype: spr.bsr_matrix
    """
    all_rows = []
    size = len(sim_df['query'].unique())
    row_np = np.zeros((1, size), dtype=np.float16)
    sim_df.sort_values(by='query', ignore_index=True, inplace=True)
    query_grouped = sim_df.groupby(by='query')

    for _, row in query_grouped:
        inds, values = row['target'], row['metric']
        slicing = values > threshold
        inds, values = inds[slicing], values[slicing]
        new_row_np = row_np.copy()
        new_row_np[:, inds] = values
        row_scp = spr.csr_array(new_row_np)
        all_rows.append(row_scp)

    matrix = spr.csr_matrix(spr.vstack(all_rows))
    return matrix.maximum(matrix.transpose())


def calculate_similarity(
    df_query: pd.DataFrame,
    df_target: pd.DataFrame = None,
    species: str = 'protein',
    similarity_metric: str = 'mmseqs+prefilter',
    field_name: str = 'sequence',
    threshold: float = 0.3,
    threads: int = cpu_count(),
    verbose: int = 0,
    save_alignment: bool = False,
    filename: str = None,
    distance: str = 'tanimoto',
    bits: int = 1024,
    radius: int = 2,
    denominator: str = 'shortest',
    representation: str = '3di+aa',
    config: dict = None
) -> pd.DataFrame:
    """Calculate similarity between entities in
    `df_query` and `df_target`. Entities can be
    biological sequences (nucleic acids or proteins),
    protein structures or small molecules (in SMILES format).

    :param df_query: DataFrame with query entities to calculate similarities
    :type df_query: pd.DataFrame
    :param df_target: DataFrame with target entities to calculate
    similarities. If not specified, the `df_query` will be used as `df_target`
    as well, defaults to None
    :type df_target: pd.DataFrame, optional
    :param species: Biochemical species to which the data belongs.
    Options: `protein`, `DNA`, `RNA`, or `small_molecule`; defaults to
    'protein'
    :type species: str, optional
    :param similarity_metric: Similarity function to use.
    Options:
        - `protein`: `mmseqs` (local alignment),
          `mmseqs+prefilter` (fast local alignment), `needle` (global
           alignment), or `foldseek` (structural alignment).
        - `DNA` or `RNA`: `mmseqs` (local alignment),
          `mmseqs+prefilter` (fast local alignment), or `needle`
          (global alignment).
        - `small molecule`: `scaffold` (boolean comparison of Bemis-Murcko
           scaffolds: either identical or not) or
          `fingerprint` (Tanimoto distance between ECFP (extended connectivity
           fingerprints))
    Defaults to `mmseqs+prefilter`.
    :type similarity_metric: str, optional
    :param field_name: Name of the field with the entity information
    (e.g., `protein_sequence` or `structure_path`), defaults to 'sequence'.
    :type field_name: str, optional
    :param threshold: Similarity value above which entities will be
    considered similar, defaults to 0.3
    :type threshold: float, optional
    :param threads: Number of threads available for parallalelization,
    defaults to cpu_count()
    :type threads: int, optional
    :param verbose: How much information will be displayed.
    Options:
        - 0: Errors,
        - 1: Warnings,
        - 2: All
    Defaults to 0
    :type verbose: int, optional
    :param save_alignment: Save file with similarity calculations,
    defaults to False
    :type save_alignment: bool, optional
    :param filename: Filename where to save the similarity calculations
    requires `save_alignment` set to `True`, defaults to None
    :type filename: str, optional
    :param distance: Distance metrics for small molecule comparison.
    Currently, it is restricted to Tanimoto distance will
    be extended in future patches; if interested in a specific
    metric please let us know.
    Options:
        - `tanimoto`: Calculates the Tanimoto distance
    Defaults to 'tanimoto'.
    :type distance: str, optional
    :param bits: Number of bits for ECFP, defaults to 1024
    :type bits: int, optional
    :param radius: Radius for ECFP calculation, defaults to 2
    :type radius: int, optional
    :param denominator: Denominator for sequence alignments, refers
    to which lenght to be used as denominator for calculating
    the sequence identity.
    Options:
        - `shortest`: The shortest sequence of the pair
        - `longest`: The longest sequence of the pair 
                    (recomended only for peptides)
        - `n_aligned`: Full alignment length 
                      (recomended with global alignment)
    Defaults to 'shortest'
    :type denominator: str, optional
    :param representation: Representation for protein structures
    as interpreted by `Foldseek`.
    Options:
        - `3di`: 3D interactions vocabulary.
        - `3di+aa`: 3D interactions vocabulary and amino
                    acid sequence.
        - `TM`: global structural alignment (slow)
    Defaults to '3di+aa'
    :type representation: str, optional
    :param config: Dictionary with options for EMBOSS needle module
    Default values:
        - "gapopen": 10,
        - "gapextend": 0.5,
        - "endweight": True,
        - "endopen": 10,
        - "endextend": 0.5,
        - "matrix": "EBLOSUM62"
    :type config: dict, optional
    :raises NotImplementedError: Biochemical species is not supported
                                 see `species`.
    :raises NotImplementedError: Similarity metric is not supported
                                 see `similarity_algorithm`
    :return: DataFrame with similarities (`metric`) between
    `query` and `target`.
    `query` and `target` are named as the indexes obtained from the 
    `pd.unique` function on the corresponding input DataFrames.
    :rtype: pd.DataFrame
    """
    mssg = f'Alignment method: {similarity_metric} '
    mssg += f'not implemented for species: {species}'
    mssg2 = f'Species: {species} not supported'

    if species == 'protein':
        if 'mmseqs' in similarity_metric:
            sim_df = _mmseqs2_alignment(
                df_query=df_query,
                df_target=df_target,
                field_name=field_name,
                threshold=threshold,
                threads=threads,
                prefilter='prefilter' in similarity_metric,
                denominator=denominator,
                is_nucleotide=False,
                verbose=verbose,
                save_alignment=save_alignment,
                filename=filename
            )
        elif similarity_metric == 'needle':
            sim_df = _needle_alignment(
                df_query=df_query,
                df_target=df_target,
                field_name=field_name,
                threshold=threshold,
                threads=threads,
                is_nucleotide=False,
                verbose=verbose,
                config=config,
                save_alignment=save_alignment,
                filename=filename
            )
        elif similarity_metric == 'foldseek':
            sim_df = _foldseek_alignment(
                df_query=df_query,
                df_target=df_target,
                field_name=field_name,
                threshold=threshold,
                prefilter=False,
                denominator=denominator,
                representation=representation,
                threads=threads,
                verbose=verbose,
                save_alignment=save_alignment,
                filename=filename
            )
        else:
            mssg = f'Alignment method: {similarity_metric} '
            mssg += f'not implemented for species: {species}'
            raise NotImplementedError(mssg)
    elif species.upper() == 'DNA' or species.upper() == 'RNA':
        if 'mmseqs' in similarity_metric:
            sim_df = _mmseqs2_alignment(
                df_query=df_query,
                df_target=df_target,
                field_name=field_name,
                threshold=threshold,
                threads=threads,
                prefilter='prefilter' in similarity_metric,
                denominator=denominator,
                is_nucleotide=True,
                verbose=verbose,
                save_alignment=save_alignment,
                filename=filename
            )
        elif similarity_metric == 'needle':
            sim_df = _needle_alignment(
                df_query=df_query,
                df_target=df_target,
                field_name=field_name,
                threshold=threshold,
                threads=threads,
                is_nucleotide=True,
                verbose=verbose,
                config=config,
                save_alignment=save_alignment,
                filename=filename
            )
        else:
            mssg = f'Alignment method: {similarity_metric} '
            mssg += f'not implemented for species: {species}'
            raise NotImplementedError(mssg)
    elif species == 'small_molecule' or species.lower() == 'smiles':
        if similarity_metric == 'scaffold':
            sim_df = _scaffold_alignment(
                df_query=df_query,
                df_target=df_target,
                field_name=field_name,
                threads=threads,
                verbose=verbose,
                save_alignment=save_alignment,
                filename=filename
            )
        elif similarity_metric == 'fingerprint':
            sim_df = _fingerprint_alignment(
                df_query=df_query,
                df_target=df_target,
                threshold=threshold,
                field_name=field_name,
                distance=distance,
                threads=threads,
                verbose=verbose,
                bits=bits,
                radius=radius,
                save_alignment=save_alignment,
                filename=filename
            )
    else:
        raise NotImplementedError(mssg2)
    return sim_df


def _scaffold_alignment(
    df_query: pd.DataFrame,
    df_target: pd.DataFrame = None,
    field_name: str = 'smiles',
    threads: int = cpu_count(),
    verbose: int = 0,
    save_alignment: bool = False,
    filename: str = None
) -> Union[pd.DataFrame, np.ndarray]:
    """_summary_

    :param df_query: _description_
    :type df_query: pd.DataFrame
    :param df_target: _description_, defaults to None
    :type df_target: pd.DataFrame, optional
    :param field_name: _description_, defaults to 'smiles'
    :type field_name: str, optional
    :param threads: _description_, defaults to cpu_count()
    :type threads: int, optional
    :param verbose: _description_, defaults to 0
    :type verbose: int, optional
    :param save_alignment: _description_, defaults to False
    :type save_alignment: bool, optional
    :param filename: _description_, defaults to None
    :type filename: str, optional
    :raises ImportError: _description_
    :raises ValueError: _description_
    :raises ValueError: _description_
    :raises RuntimeError: _description_
    :return: _description_
    :rtype: Union[pd.DataFrame, np.ndarray]
    """
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
    except ModuleNotFoundError:
        raise ImportError("This function requires RDKit to be installed.")
    from concurrent.futures import ThreadPoolExecutor

    if (field_name not in df_query.columns):
        raise ValueError(f'{field_name} not found in query DataFrame')
    if df_target is not None and field_name not in df_target.columns:
        raise ValueError(f'{field_name} not found in target DataFrame')

    def _compute_distance(query_scaffold: str, target_scaffolds: List[str]):
        distances = []
        for target in target_scaffolds:
            if target == query_scaffold:
                distances.append(1)
            else:
                distances.append(0)
        return distances

    if df_target is None:
        df_target = df_query

    mols_query = [Chem.MolFromSmiles(smiles)
                  for smiles in df_query[field_name]]
    mols_target = [Chem.MolFromSmiles(smiles)
                   for smiles in df_target[field_name]]
    scaffolds_query = [MurckoScaffoldSmiles(mol=mol, includeChirality=True)
                       for mol in mols_query]
    scaffolds_target = [MurckoScaffoldSmiles(mol=mol, includeChirality=True)
                        for mol in mols_target]

    jobs = []

    with ThreadPoolExecutor(max_workers=threads) as executor:
        for query_scafold in scaffolds_query:
            job = executor.submit(_compute_distance, query_scafold,
                                  scaffolds_target)
            jobs.append(job)

        if verbose > 1:
            pbar = tqdm(jobs)
        else:
            pbar = jobs

        proto_df = []
        for idx, job in enumerate(pbar):
            if job.exception() is not None:
                raise RuntimeError(job.exception())
            result = job.result()
            entry = [{'query': idx, 'target': idx_target, 'metric': metric}
                     for idx_target, metric in enumerate(result)]
            proto_df.extend(entry)

    df = pd.DataFrame(proto_df)
    if save_alignment:
        if filename is None:
            filename = time.time()
        df.to_csv(f'{filename}.csv.gz', index=False, compression='gzip')
    return df


def _fingerprint_alignment(
    df_query: pd.DataFrame,
    df_target: pd.DataFrame = None,
    threshold: float = 0.0,
    field_name: str = 'smiles',
    distance: str = 'tanimoto',
    threads: int = cpu_count(),
    verbose: int = 0,
    bits: int = 1024,
    radius: int = 2,
    save_alignment: bool = False,
    filename: str = None,
    **kwargs
) -> Union[pd.DataFrame, np.ndarray]:
    # Threshold for similarity evaluation: 0.85, based on:
    # Patterson DE, Cramer RD, Ferguson AM, Clark RD, Weinberger LE:
    # Neighborhood behavior: A useful concept for validation of ''molecular
    # diversity'' descriptors. J Med Chem 1996, 39:3049-3059.
    """_summary_

    :param df_query: _description_
    :type df_query: pd.DataFrame
    :param df_target: _description_, defaults to None
    :type df_target: pd.DataFrame, optional
    :param threshold: _description_, defaults to 0.0
    :type threshold: float, optional
    :param field_name: _description_, defaults to 'smiles'
    :type field_name: str, optional
    :param alignment: _description_, defaults to 'tanimoto'
    :type alignment: str, optional
    :param threads: _description_, defaults to cpu_count()
    :type threads: int, optional
    :param verbose: _description_, defaults to 0
    :type verbose: int, optional
    :param bits: _description_, defaults to 1024
    :type bits: int, optional
    :param radius: _description_, defaults to 2
    :type radius: int, optional
    :param save_alignment: _description_, defaults to False
    :type save_alignment: bool, optional
    :param filename: _description_, defaults to None
    :type filename: str, optional
    :raises ImportError: _description_
    :raises ValueError: _description_
    :raises ValueError: _description_
    :raises RuntimeError: _description_
    :return: _description_
    :rtype: Union[pd.DataFrame, np.ndarray]
    """
    try:
        from rdkit import Chem
        from rdkit import DataStructs
        from rdkit.Chem import AllChem
    except ModuleNotFoundError:
        raise ImportError("This function requires RDKit to be installed.")
    from concurrent.futures import ThreadPoolExecutor

    def _compute_tanimoto(query_fp: list, target_fps: list):
        scores = DataStructs.BulkTanimotoSimilarity(query_fp, target_fps)
        return scores

    if (field_name not in df_query.columns):
        raise ValueError(f'{field_name} not found in query DataFrame')
    if df_target is not None and field_name not in df_target.columns:
        raise ValueError(f'{field_name} not found in target DataFrame')

    if verbose > 0:
        print(f'Calculating molecular similarities using ECFP-{radius * 2}',
              f'with {bits:,} bits and Tanimoto distance...')

    if df_target is None:
        df_target = df_query

    chunk_size = threads
    chunks_query = (len(df_query) // chunk_size) + 1
    chunks_target =(len(df_query) // chunk_size) + 1
    proto_df = []
    pbar = tqdm(range(chunks_query))
    with ThreadPoolExecutor(max_workers=threads) as executor:
        for chunk in pbar:
            start = chunk * chunk_size
            if chunk == chunks_query - 1:
                end = -1
            else:
                end = (chunk + 1) * chunk_size
            mols_query = [Chem.MolFromSmiles(smiles)
                        for smiles in df_query[field_name][start:end]]
            fps_query = [AllChem.GetMorganFingerprintAsBitVect(x, radius, bits)
                        for x in mols_query]
            for chunk_t in range(chunks_target):
                pbar.set_description(f'Covered: {chunk_t}/{chunks_target}')
                start_t = chunk_t * chunk_size
                if chunk_t == chunks_target - 1:
                    end_t = -1
                else:
                    end_t = (chunk_t + 1) * chunk_size

                mols_target = [Chem.MolFromSmiles(smiles)
                            for smiles in df_target[field_name][start_t:end_t]]
                fps_target = [AllChem.GetMorganFingerprintAsBitVect(x, radius, bits)
                            for x in mols_target]
                jobs = []
                for query_fp in fps_query:
                    job = executor.submit(_compute_tanimoto, query_fp, fps_target)
                    jobs.append(job)

                for idx, job in enumerate(jobs):
                    if job.exception() is not None:
                        raise RuntimeError(job.exception())
                    result = job.result()
                    entry = [{'query': start + idx, 'target': start_t + idx_target, 'metric': metric}
                            for idx_target, metric in enumerate(result)]
                    proto_df.extend(entry)

    df = pd.DataFrame(proto_df)
    if save_alignment:
        if filename is None:
            filename = time.time()
        df.to_csv(f'{filename}.csv.gz', index=False, compression='gzip')

    # df = df[df.metric >= threshold]
    return df


def _foldseek_alignment(
    df_query: pd.DataFrame,
    df_target: pd.DataFrame = None,
    field_name: str = 'structure',
    threshold: float = 0.0,
    prefilter: bool = True,
    denominator: str = 'shortest',
    representation: str = '3di+aa',
    threads: int = cpu_count(),
    verbose: int = 0,
    save_alignment: bool = False,
    filename: str = None,
    **kwargs
) -> Union[pd.DataFrame, np.ndarray]:
    """_summary_

    :param df_query: _description_
    :type df_query: pd.DataFrame
    :param df_target: _description_, defaults to None
    :type df_target: pd.DataFrame, optional
    :param field_name: _description_, defaults to 'structure'
    :type field_name: str, optional
    :param threshold: _description_, defaults to 0.0
    :type threshold: float, optional
    :param prefilter: _description_, defaults to True
    :type prefilter: bool, optional
    :param denominator: _description_, defaults to 'shortest'
    :type denominator: str, optional
    :param representation: _description_, defaults to '3di+aa'
    :type representation: str, optional
    :param threads: _description_, defaults to cpu_count()
    :type threads: int, optional
    :param verbose: _description_, defaults to 0
    :type verbose: int, optional
    :param save_alignment: _description_, defaults to False
    :type save_alignment: bool, optional
    :param filename: _description_, defaults to None
    :type filename: str, optional
    :raises ImportError: _description_
    :raises ValueError: _description_
    :raises ValueError: _description_
    :return: _description_
    :rtype: Union[pd.DataFrame, np.ndarray]
    """
    foldseek = os.path.join(os.path.dirname(__file__), '..',
                            'bin', 'foldseek')
    if os.path.exists(foldseek):
        pass
    elif shutil.which('foldseek') is None:
        mssg = "Foldseek not found. Please install following the instructions"
        mssg += " in: https://github.com/IBM/Hestia#installation"
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
                    'query,target,fident,alnlen,qlen,tlen,prob',
                    '-v', str(mmseqs_v)])

    df = pd.read_csv(alignment_csv, sep='\t')
    qry2idx = {os.path.basename(qry).split('.pdb')[0]: idx for idx, qry in
               enumerate(df_query[field_name].unique())}
    tgt2idx = {os.path.basename(tgt).split('.pdb')[0]: idx for idx, tgt in
               enumerate(df_target[field_name].unique())}
    to_add = set()
    for i in df_query.index:
        if i not in qry2idx.values():
            to_add.add(i)
    for i in df_target.index:
        if i not in tgt2idx.values():
            to_add.add(i)

    new_df = pd.DataFrame([{'query': i, 'target': i, 'metric': 0.0}
                          for i in to_add])
    df = pd.concat([df, new_df])

    if save_alignment:
        if filename is None:
            filename = time.time()
        df.to_csv(f'{filename}.csv.gz', index=False, compression='gzip')

    df['metric'] = 1 - df['prob']
    df['query'] = df['query'].map(lambda x: qry2idx[x.split('.pdb')[0]])
    df['query'] = df['query'].astype(int)
    df['target'] = df['target'].map(lambda x: qry2idx[x.split('.pdb')[0]])
    df['target'] = df['target'].astype(int)

    shutil.rmtree(tmp_dir)
    return df


def _mmseqs2_alignment(
    df_query: pd.DataFrame,
    df_target: pd.DataFrame = None,
    field_name: str = 'sequence',
    threshold: float = 0.0,
    prefilter: bool = True,
    denominator: str = 'shortest',
    threads: int = cpu_count(),
    is_nucleotide: bool = False,
    verbose: int = 0,
    save_alignment: bool = False,
    filename: str = None
) -> Union[pd.DataFrame, np.ndarray]:
    """_summary_

    :param df_query: _description_
    :type df_query: pd.DataFrame
    :param df_target: _description_, defaults to None
    :type df_target: pd.DataFrame, optional
    :param field_name: _description_, defaults to 'sequence'
    :type field_name: str, optional
    :param threshold: _description_, defaults to 0.0
    :type threshold: float, optional
    :param prefilter: _description_, defaults to True
    :type prefilter: bool, optional
    :param denominator: _description_, defaults to 'shortest'
    :type denominator: str, optional
    :param threads: _description_, defaults to cpu_count()
    :type threads: int, optional
    :param is_nucleotide: _description_, defaults to False
    :type is_nucleotide: bool, optional
    :param verbose: _description_, defaults to 0
    :type verbose: int, optional
    :param save_alignment: _description_, defaults to False
    :type save_alignment: bool, optional
    :param filename: _description_, defaults to None
    :type filename: str, optional
    :raises RuntimeError: _description_
    :raises ValueError: _description_
    :raises ValueError: _description_
    :return: _description_
    :rtype: Union[pd.DataFrame, np.ndarray]
    """
    if shutil.which('mmseqs') is None:
        raise RuntimeError(
            "MMSeqs2 not found. Please install following the instructions in:",
            "https://github.com/IBM/AutoPeptideML#installation"
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
                        '7.5', f'{tmp_dir}/db_query',
                        f'{tmp_dir}/db_target',
                        f'{tmp_dir}/pref', '-v',
                        f'{mmseqs_v}'])
    else:
        program_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'utils', 'mmseqs_fake_prefilter.sh'
        )
        # This whole thing has to be checked compare to original
        subprocess.run([program_path,
                        f'{tmp_dir}/db_query', f'{tmp_dir}/db_target',
                        f'{tmp_dir}/pref', 'db_query'])

    denominator = DENOMINATOR_DICT[denominator]
    subprocess.run(['mmseqs', 'align',  f'{tmp_dir}/db_query',
                    f'{tmp_dir}/db_target', f'{tmp_dir}/pref',
                    f'{tmp_dir}/align_db', '--alignment-mode', '3',
                    '-e', 'inf', '--seq-id-mode', denominator,
                    '-v', f'{mmseqs_v}', '--threads', f'{threads}'])

    file = os.path.join(tmp_dir, 'alignments.tab')
    subprocess.run(['mmseqs', 'convertalis', f'{tmp_dir}/db_query',
                    f'{tmp_dir}/db_target', f'{tmp_dir}/align_db',
                    '--format-mode', '4', '--threads', f'{threads}',
                    file, '-v', '1'])

    df = pd.read_csv(file, sep='\t')
    query_indxs = df['query'].unique().tolist()
    target_indxs = df['target'].unique().tolist()
    to_add = set()
    for i in df_query.index:
        if i not in query_indxs:
            to_add.add(i)
    for i in df_target.index:
        if i not in target_indxs:
            to_add.add(i)
    df['metric'] = df['fident']
    new_df = pd.DataFrame([{'query': i, 'target': i, 'metric': 0.0}
                          for i in to_add])
    df = pd.concat([df, new_df])

    if save_alignment:
        if filename is None:
            filename = time.time()
        df.to_csv(f'{filename}.csv.gz', index=False, compression='gzip')

    shutil.rmtree(tmp_dir)
    return df


def _needle_alignment(
    df_query: pd.DataFrame,
    df_target: pd.DataFrame = None,
    field_name: str = 'sequence',
    threshold: float = 0.0,
    denominator: str = 'shortest',
    threads: int = cpu_count(),
    is_nucleotide: bool = False,
    verbose: int = 0,
    config: dict = None,
    save_alignment: bool = False,
    filename: str = None
):
    """_summary_

    :param df_query: _description_
    :type df_query: pd.DataFrame
    :param df_target: _description_, defaults to None
    :type df_target: pd.DataFrame, optional
    :param field_name: _description_, defaults to 'sequence'
    :type field_name: str, optional
    :param threshold: _description_, defaults to 0.0
    :type threshold: float, optional
    :param denominator: _description_, defaults to 'shortest'
    :type denominator: str, optional
    :param threads: _description_, defaults to cpu_count()
    :type threads: int, optional
    :param is_nucleotide: _description_, defaults to False
    :type is_nucleotide: bool, optional
    :param verbose: _description_, defaults to 0
    :type verbose: int, optional
    :param config: _description_, defaults to None
    :type config: dict, optional
    :param save_alignment: _description_, defaults to False
    :type save_alignment: bool, optional
    :param filename: _description_, defaults to None
    :type filename: str, optional
    :raises ImportError: _description_
    :raises ValueError: _description_
    :raises ValueError: _description_
    :raises RuntimeError: _description_
    :return: _description_
    :rtype: _type_
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
    # df = df[df['metric'] > threshold]
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
