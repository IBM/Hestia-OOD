import pickle
import json
from multiprocessing import cpu_count
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from hestia.similarity import (sequence_similarity_mmseqs,
                               sequence_similarity_needle,
                               sequence_similarity_peptides,
                               molecular_similarity,
                               embedding_similarity,
                               protein_structure_similarity)
from hestia.partition import random_partition, ccpart, graph_part


class SimArguments:
    """Dataclass with the inputs for similarity calculation.
    """
    def __init__(
        self,
        data_type: str = 'protein',
        field_name: str = 'sequence',
        min_threshold: float = 0.0,
        threads: int = cpu_count(),
        verbose: int = 0,
        save_alignment: bool = False,
        filename: Optional[str] = 'alignment',
        sim_function: Optional[str] = None,
        bits: Optional[int] = None,
        radius: Optional[int] = None,
        fingerprint: Optional[str] = None,
        denominator: Optional[str] = None,
        representation: Optional[str] = None,
        prefilter: Optional[bool] = None,
        alignment_algorithm: Optional[str] = None,
        query_embds: Optional[np.ndarray] = None,
        target_embds: Optional[np.ndarray] = None,
        target_df: Optional[pd.DataFrame] = None,
        needle_config: Optional[dict] = None
    ):
        self.data_type = data_type
        self.field_name = field_name
        self.min_threshold = min_threshold
        self.threads = threads
        self.verbose = verbose
        self.save_alignment = save_alignment
        self.filename = filename
        self.target_df = target_df

        if 'molecule' in self.data_type or 'smiles' in self.data_type:
            self.bits = (1_024 if bits is None else bits)
            self.radius = (2 if radius is None else radius)
            self.fingerprint = ('ecfp' if fingerprint is None else fingerprint)
            self.sim_function = ('tanimoto' if sim_function is None else sim_function)
        elif 'sequence' in self.data_type:
            self.denominator = (denominator if denominator is not None
                                else 'n_aligned')
            self.prefilter = (False if prefilter is None else prefilter)
            self.alignment_algorithm = ('mmseqs' if alignment_algorithm is None
                                        else alignment_algorithm)
            if alignment_algorithm == 'needle':
                self.needle_config = {
                    "gapopen": 10,
                    "gapextend": 0.5,
                    "endweight": True,
                    "endopen": 10,
                    "endextend": 0.5,
                    "matrix": "EBLOSUM62"
                } if needle_config is None else needle_config
            if ('dna' in self.data_type.lower() or
                'RNA' in self.data_type.lower() or
               'nucl' in self.data_type.lower()):
                self.is_nucleotide = True
            else:
                self.is_nucleotide = False
        elif self.data_type == 'protein_structure':
            self.denominator = (denominator if denominator is not None
                                else 'n_aligned')
            self.representation = (representation if representation is not None
                                   else '3di+aa')
            self.prefilter = (False if prefilter is None else prefilter)
        elif self.data_type == 'embedding':
            if query_embds is None:
                raise ValueError('Query embds need to be provided for embedding similarity.')
            self.sim_function = (sim_function if sim_function is not None
                                 else 'cosine-np')
            self.query_embds = query_embds
            self.target_embds = target_embds
        else:
            raise NotImplementedError(f"Data type: {data_type} not implemented.")


class HestiaGenerator:
    """Class for generating multiple Dataset
    partitions for generalisation evaluation.
    """
    def __init__(self, data: pd.DataFrame, verbose: Optional[bool] = True):
        """Initialise class

        :param data: DataFrame with the original data from which
            datasets will be generated.
        :type data: pd.DataFrame
        """
        self.data = data
        self.sim_df = None
        self.partitions = None
        self.sim_args = None
        self.verbose = verbose
        if self.verbose:
            print('Initialising Hestia Dataset Generator')
            print(f'Number of items in data: {len(self.data):,}')

    def get_partition(self, partition: Union[str, float]) -> dict:
        return self.partitions[partition]

    def get_partitions(self, filter: Union[bool, int, float] = False) -> dict:
        out_partitions = {}

        if isinstance(filter, bool) and filter:
            thresh = len(self.data) * 0.185
        elif isinstance(filter, int):
            thresh = filter
        elif isinstance(filter, float):
            thresh = len(self.data) * filter
        else:
            thresh = 0

        for key, part in self.partitions.items():
            if len(part['test']) < thresh:
                continue
            out_partitions[key] = part

        return out_partitions.items()

    def from_precalculated(self, data_path: str):
        """Load partition indexes if they have already being calculated.

        :param data_path: Path to saved partition indexes.
        :type data_path: str
        """
        # with gzip.open(data_path, 'r') as fin:
        #     input = json.loads(fin.read().decode('utf-8'))
        input = pickle.load(open(data_path, 'rb'))
        if 'partitions' in input:
            self.partitions = input['partitions']
        else:
            self.partitions = input

        if 'metadata' not in input:
            print('Warning: there is no metadata available.')
        else:
            self.metadata = input['metadata']
            if ('cluster_composition' not in self.metadata and
               'clusters' not in self.metadata):
                print('Warning: there is no clusters metadata available.')

            if 'partition_algorithm' not in self.metadata:
                print('Warning: there is no metadata available regarding ' +
                      'the partitioning algorithm.')

            if 'similarity_metric' not in self.metadata:
                print('Warning: there is no metadata available regarding ' +
                      'the similarity metric.')

        new_dict = {}
        for key, value in self.partitions.items():
            if key != 'random':
                new_dict[float(key)] = value
            else:
                new_dict[key] = value
        self.partitions = new_dict

    def save_precalculated(self, output_path: str,
                           include_metada: Optional[bool] = True):
        """Save partition indexes to disk for quickier re-running.

        :param output_path: Path where partition indexes should be saved.
        :type output_path: str
        """
        clusters = {th: c['clusters'] for th, c in self.partitions.items()
                    if th != 'random'}
        for th, c in self.partitions.items():
            if th != 'random':
                del c['clusters']

        if include_metada:
            self.metadata['cluster_composition'] = clusters
        else:
            self.metadata['cluster_composition'] = {
                cluster.item: n_elements.item for cluster, n_elements in
                zip(np.unique(clusters, return_counts=True))
            }
        output = {
            'partitions': self.partitions,
            'metadata': self.metadata
        }
        pickle.dump(output, open(output_path, 'wb'))

    def calculate_similarity(self, sim_args: SimArguments) -> pd.DataFrame:
        """Calculate pairwise similarity between all the elements in the dataset.

        :param sim_args: See similarity arguments entry.
        :type similarity_args: SimArguments
        """
        if self.verbose:
            print('Calculating similarity...')
        if self.sim_args is None:
            self.sim_args = sim_args
        if 'sequence' in sim_args.data_type:
            if sim_args.alignment_algorithm == 'mmseqs':
                sim_df = sequence_similarity_mmseqs(
                    df_query=self.data, df_target=sim_args.target_df,
                    field_name=sim_args.field_name,
                    prefilter=sim_args.prefilter,
                    denominator=sim_args.denominator,
                    is_nucleotide=sim_args.is_nucleotide,
                    threshold=sim_args.min_threshold,
                    threads=sim_args.threads,
                    save_alignment=sim_args.save_alignment,
                    filename=sim_args.filename,
                    verbose=sim_args.verbose)
            elif sim_args.alignment_algorithm == 'needle':
                sim_df = sequence_similarity_needle(
                    df_query=self.data, df_target=sim_args.target_df,
                    field_name=sim_args.field_name,
                    denominator=sim_args.denominator,
                    is_nucleotide=sim_args.is_nucleotide,
                    config=sim_args.needle_config,
                    threshold=sim_args.min_threshold,
                    threads=sim_args.threads,
                    save_alignment=sim_args.save_alignment,
                    filename=sim_args.filename,
                    verbose=sim_args.verbose
                )
            elif sim_args.alignment_algorithm == 'peptides':
                sim_df = sequence_similarity_peptides(
                    df_query=self.data, df_target=sim_args.target_df,
                    field_name=sim_args.field_name,
                    denominator=sim_args.denominator,
                    threshold=sim_args.min_threshold,
                    threads=sim_args.threads,
                    save_alignment=sim_args.save_alignment,
                    filename=sim_args.filename,
                    verbose=sim_args.verbose
                )
        elif 'protein_structure' in sim_args.data_type:
            sim_df = protein_structure_similarity(
                df_query=self.data, df_target=sim_args.target_df,
                field_name=sim_args.field_name,
                prefilter=sim_args.prefilter, denominator=sim_args.denominator,
                representation=sim_args.representation,
                threshold=sim_args.min_threshold, threads=sim_args.threads,
                verbose=sim_args.verbose,
                save_alignment=sim_args.save_alignment,
                filename=sim_args.filename
            )
        elif 'molecu' in sim_args.data_type:
            sim_df = molecular_similarity(
                df_query=self.data,
                df_target=sim_args.target_df,
                field_name=sim_args.field_name,
                sim_function=sim_args.sim_function,
                fingerprint=sim_args.fingerprint,
                bits=sim_args.bits, radius=sim_args.radius,
                threshold=sim_args.min_threshold, threads=sim_args.threads,
                verbose=sim_args.verbose,
                save_alignment=sim_args.save_alignment,
                filename=sim_args.filename
            )
        elif 'embedding' in sim_args.data_type:
            sim_df = embedding_similarity(
                query_embds=sim_args.query_embds,
                df_target=sim_args.target_df,
                sim_function=sim_args.sim_function,
                threads=sim_args.threads, threshold=sim_args.min_threshold,
                save_alignment=sim_args.save_alignment,
                filename=sim_args.filename
            )
        if self.verbose:
            print('Similarity successfully calculated!')
        self.sim_df = sim_df
        return sim_df

    def load_similarity(self, output_path: str):
        """Load similarity calculation from file.

        :param output_path: File with similarity calculations.
        :type output_path: str
        """
        if self.verbose:
            print('Loading precalculated similarity...')
        self.sim_df = pd.read_csv(output_path, compression='gzip')
        if self.verbose:
            print('Precalculated similarity loaded successfully!')

    def calculate_partitions(
        self,
        sim_args: Optional[SimArguments] = None,
        sim_df: Optional[pd.DataFrame] = None,
        label_name: Optional[str] = None,
        min_threshold: Optional[float] = 0.,
        threshold_step: Optional[float] = 0.05,
        test_size: Optional[float] = 0.2,
        valid_size: Optional[float] = 0.1,
        partition_algorithm: Optional[str] = 'ccpart',
        random_state: Optional[int] = 42,
        n_partitions: Optional[int] = None
    ):
        """
        Calculates multiple partitions of a dataset for training, validation, and testing based on sequence similarity.
        Supports two partitioning algorithms: `ccpart` and `graph_part`. Additionally, it computes partitions for 
        different similarity thresholds and random partitions.

        :param sim_args: Object containing the similarity parameters for partitioning. This includes options for 
                        calculating sequence similarity, such as the alignment method and similarity threshold.
                        Defaults to None.
        :type sim_args: Optional[SimArguments], optional
        :param sim_df: Precomputed similarity DataFrame. If None, the similarity will be calculated using `sim_args`.
        :type sim_df: Optional[pd.DataFrame], optional
        :param label_name: The name of the label column for the dataset. Defaults to None.
        :type label_name: Optional[str], optional
        :param min_threshold: The minimum similarity threshold to start partitioning. Defaults to 0.0.
        :type min_threshold: Optional[float], optional
        :param threshold_step: The step size for varying the similarity threshold during partitioning. Defaults to 0.05.
        :type threshold_step: Optional[float], optional
        :param test_size: The proportion of the dataset to allocate to the test set. Defaults to 0.2.
        :type test_size: Optional[float], optional
        :param valid_size: The proportion of the training set to allocate to the validation set. Defaults to 0.1.
        :type valid_size: Optional[float], optional
        :param partition_algorithm: The partitioning algorithm to use. Options are:
                                - `'ccpart'`: Community detection partitioning algorithm.
                                - `'graph_part'`: Graph-based partitioning.
                                Defaults to `'ccpart'`.
        :type partition_algorithm: Optional[str], optional
        :param random_state: The random seed for reproducibility. Defaults to 42.
        :type random_state: Optional[int], optional
        :param n_partitions: The number of partitions to create when using `graph_part`. Defaults to None.
        :type n_partitions: Optional[int], optional
        :raises ValueError: If an unsupported partition algorithm is specified.
        :return: A dictionary containing the partitions for each threshold. The dictionary has keys:
                - `train`: DataFrame for the training set.
                - `valid`: DataFrame for the validation set.
                - `test`: DataFrame for the test set.
                - `clusters`: The clusters formed by the partitioning algorithm.
                - For random partitions, the key `'random'` will contain the train, valid, and test sets.
        :rtype: dict

        :example:
        # Example of partitioning with a similarity threshold of 0.3 and a test size of 0.2
        partitions = calculate_partitions(
            sim_args=similarity_args,
            label_name='Y',
            min_threshold=0.2,
            threshold_step=0.05,
            test_size=0.2,
            partition_algorithm='ccpart',
            random_state=42
        )

        # Accessing the partitions for a specific threshold
        train_set = partitions[0.3]['train']
        valid_set = partitions[0.3]['valid']
        test_set = partitions[0.3]['test']
        """
        self.partitions = {}
        if sim_df is None and self.sim_df is None:
            sim_args.min_threshold = min_threshold
            sim_df = self.calculate_similarity(sim_args)
        elif sim_df is None and self.sim_df is not None:
            sim_df = self.sim_df
        if self.verbose:
            print('Calculating partitions...')

        if partition_algorithm not in ['ccpart', 'graph_part']:
            raise ValueError(
                f'Partition algorithm: {partition_algorithm} is not ' +
                'supported. Try using: `ccpart` or `graph_part`.'
            )
        min_threshold = int(min_threshold * 100)
        threshold_step = int(threshold_step * 100)

        for th in tqdm(range(min_threshold, 100, threshold_step)):
            if partition_algorithm == 'ccpart':
                train, test, clusters = ccpart(
                    self.data,
                    label_name=label_name,
                    test_size=test_size,
                    threshold=th / 100,
                    sim_df=sim_df, verbose=2
                )
                th_parts = (train, test)
            elif partition_algorithm == 'graph_part':
                try:
                    th_parts, clusters = graph_part(
                        self.data,
                        label_name=label_name,
                        test_size=test_size if n_partitions is None else 0.0,
                        threshold=th / 100,
                        sim_df=sim_df, verbose=2,
                        n_parts=n_partitions
                    )

                except RuntimeError as e:
                    print(e)
                    continue

            if n_partitions is None:
                if valid_size > 0.:
                    train_th_parts = random_partition(
                        self.data.iloc[th_parts[0]].reset_index(drop=True),
                        test_size=valid_size, random_state=random_state
                    )
                    self.partitions[th / 100] = {
                        'train': train_th_parts[0],
                        'valid': train_th_parts[1],
                        'test': th_parts[1],
                        'clusters': clusters
                    }
                else:
                    self.partitions[th / 100] = {
                        'train': th_parts[0],
                        'test': th_parts[1],
                        'clusters': clusters
                    }
            else:
                th_parts = [[i[0] for i in part] for part in th_parts]
                self.partitions[th / 100] = {
                    i: th_parts[i] for i in range(n_partitions)
                }

        random = random_partition(self.data, test_size=test_size,
                                  random_state=random_state)
        if valid_size > 0.:
            train_random = random_partition(
                self.data.iloc[random[0]].reset_index(drop=True),
                test_size=valid_size, random_state=random_state
            )
            self.partitions['random'] = {
                'train': train_random[0],
                'valid': train_random[1],
                'test': random[1]
            }
        else:
            self.partitions['random'] = {
                'train': random[0],
                'test': random[1]
            }
        sim_metadata = vars(sim_args)
        if sim_args.data_type == 'embedding':
            del sim_metadata['query_embds']
            if 'target_embds' in sim_metadata:
                del sim_metadata['target_embds']

        self.metadata = {
            'partition_algorithm': {
                'algorithm': partition_algorithm,
                'min_threshold': min_threshold,
                'threshold_step': threshold_step,
                'test_size': test_size,
                'valid_size': valid_size,
                'random_state': random_state,
                'n_partitions': n_partitions
            },
            'similarity_metric': sim_metadata
        }
        if self.verbose:
            print('Partitions successfully calculated!')

    def generate_datasets(self, dataset_type: str, threshold: float) -> dict:
        ds = {}

        if dataset_type == 'huggingface' or dataset_type == 'hf':
            try:
                import datasets
                import pyarrow as pa
            except ImportError:
                raise ImportError(
                    f"This dataset_type: {dataset_type} requires `datasets` " +
                    "to be installed. Install using: `pip install datasets`"
                    )
            for key, value in self.partitions[threshold].items():
                try:
                    ds[key] = datasets.Dataset.from_pandas(
                        self.data.iloc[value].reset_index()
                    )
                except pa.ArrowInvalid:
                    ds[key] = datasets.Dataset.from_dict({
                        column: [row[column] for idx, row in
                                 self.data.iloc[value].iterrows()]
                        for column in self.data.columns
                    })
            return ds
        elif dataset_type == 'pytorch' or dataset_type == 'torch':
            try:
                from hestia.utils.dataset_utils import Dataset_from_pandas
            except ModuleNotFoundError:
                raise ImportError(
                    f"This dataset_type: {dataset_type} requires `torch` " +
                    "to be installed. Install using: `pip install torch`"
                    )
            for key, value in self.partitions[threshold].items():
                ds[key] = Dataset_from_pandas(
                    self.data.iloc[value].reset_index()
                )
            return ds

    def calculate_augood(
        self, results: Dict[float, float],
        target_df: pd.DataFrame, target_field_name: Optional[str],
        target_embds: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float]:
        """Calculate the 'area under the GOOD curve' (AU-GOOD) metric.

        This function calculates an AU-GOOD score by computing a weighted metric from similarity values
        obtained by comparing target deployment distribution to the training distribution. It returns both
        the weighted GOOD curve values and the AU-GOOD score.

        :param results: A dictionary where keys are bins or thresholds (float) and values are metrics or counts
                        associated with each bin.
        :type results: Dict[float, float]
        :param target_df: A DataFrame containing the target data for similarity comparison. The column
                        specified by `target_field_name` will be used to populate the similarity
                        arguments for comparison.
        :type target_df: pd.DataFrame
        :param target_field_name: Name of the field in `target_df` that contains target values for comparison.
        :type target_field_name: Optional[str]
        :param target_embds: A NumPy array containing the target embeddings for similarity calculation.
        :type target_embds: Optional[np.ndarray]
        :return: A tuple containing:
                - `good_curve` (np.ndarray): Array of weighted values representing the GOOD curve.
                - `au_good` (float): The calculated area under the GOOD curve.
        :rtype: Tuple[np.ndarray, float]
        """
        t_sim_args = self.sim_args
        target_df[t_sim_args.field_name] = target_df[target_field_name]
        t_sim_args.target_df = target_df
        t_sim_df = self.calculate_similarity(t_sim_args)
        bins, values, weights = [], [], []

        for key, value in results.items():
            if key == 'random':
                continue
            bins.append(float(key))
            values.append(float(value))

        bins, values = np.array(bins), np.array(values)
        rows = []
        for index in range(len(t_sim_df)):
            i = t_sim_df[t_sim_df['query'] == index].copy()
            rows.append(i['metric'].max())

        rows = np.array(rows)
        for idx, bin in enumerate(bins):
            if idx == 0:
                counts = (rows <= bin).sum()
            else:
                counts = ((rows > bins[idx - 1]) * (rows <= bin)).sum()
            weights.append(counts)

        weights = np.array(weights)
        weights = weights / weights.sum()
        good_curve = weights * values
        au_good = np.dot(weights, values)
        return good_curve, au_good

    @staticmethod
    def compare_models(
        model_results: Dict[str, Union[List[float], np.ndarray]],
        statistical_test: str = 'wilcoxon'
    ) -> np.ndarray:
        """Compare the generalisation capabilities of n models against
        each other, providing p-values for every possible pair of models
        measuring how likely is model A to be better performing than
        model B.

        :param model_results: Dictionary with model name as key and a list
        with the ordered performance values of the model at different
        thresholds.
        :type model_results: Dict[str, Union[List[float], np.ndarray]]
        :param statistical_test: Statistical test to compute the
        model differences. Currently supported:
        - `wilcoxon`: Wilcoxon ranked-sum test
        Defaults to 'wilcoxon'
        :type statistical_test: str, optional
        :rtype: np.ndarray
        """
        from scipy.stats import wilcoxon
        stat_tests = {
            'wilcoxon': wilcoxon
        }
        if statistical_test in stat_tests.keys():
            test = stat_tests[statistical_test]
        else:
            raise NotImplementedError(
                f'Statistical test: {statistical_test} is not' +
                ' currently supported. Please use one of the ' +
                f" following methods: {', '.join(stat_tests.keys())}"
            )

        matrix = np.ones((len(model_results.keys()), len(model_results.keys())))
        for idx, (key, value) in enumerate(model_results.items()):
            for idx2, (key2, value2) in enumerate(model_results.items()):
                if key == key2:
                    continue
                matrix[idx, idx2] = test(value, value2, alternative='greater')[1]
        return matrix
