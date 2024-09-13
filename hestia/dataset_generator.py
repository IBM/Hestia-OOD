import gzip
import json
from multiprocessing import cpu_count
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import auc
from tqdm import tqdm

from hestia.similarity import calculate_similarity
from hestia.partition import random_partition, ccpart, graph_part


class SimilarityArguments:
    """Dataclass with the inputs for similarity calculation.
    """
    def __init__(
        self,
        data_type: str = 'protein',
        similarity_metric: str = 'mmseqs+prefilter',
        field_name: str = 'sequence',
        min_threshold: float = 0.25,
        threads: int = cpu_count(),
        verbose: int = 0,
        save_alignment: bool = False,
        filename: str = 'alignment',
        distance: Optional[str] = None,
        bits: Optional[int] = None,
        radius: Optional[int] = None,
        denominator: Optional[str] = None,
        representation: Optional[str] = None,
        needle_config: Optional[dict] = None
    ):
        """Arguments for similarity calculation.

        :param df_query: DataFrame with query entities to calculate similarities
        :type df_query: pd.DataFrame
        :param df_target: DataFrame with target entities to calculate
        similarities. If not specified, the `df_query` will be used as `df_target`
        as well, defaults to None
        :type df_target: pd.DataFrame, optional
        :param data_type: Biochemical data_type to which the data belongs.
        Options: `protein`, `protein_structure`, `DNA`, `RNA`,
        `small_molecule` or `embedding`; defaults to 'protein'
        :type data_type: str, optional
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
        :param distance: Distance metrics for small molecule or comparison.
        Currently, it is restricted to Tanimoto distance will
        be extended in future patches; if interested in a specific
        metric please let us know.
        Options:
            - `tanimoto`: Calculates the Tanimoto similarity
            - `cosine`: Calculates the Cosine similarity
        Defaults to 'tanimoto' if `data_type` is `small_molecule` and
        to `cosine` if `data_type` is `embedding`.
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
        :param needle_config: Dictionary with options for EMBOSS needle module
        Default values:
            - "gapopen": 10,
            - "gapextend": 0.5,
            - "endweight": True,
            - "endopen": 10,
            - "endextend": 0.5,
            - "matrix": "EBLOSUM62"
        :type needle_config: dict, optional
        """
        self.data_type = data_type
        self.similarity_metric = similarity_metric
        self.field_name = field_name
        self.min_threshold = min_threshold
        self.threads = threads
        self.verbose = verbose
        self.save_alignment = save_alignment
        self.filename = filename
        self.distance = distance
        self.bits = bits
        self.radius = radius
        self.denominator = denominator
        self.representation = representation
        self.needle_config = needle_config

        if self.data_type == 'small_molecule':
            self.distance = 'tanimoto'
            self.bits = 1_022
            self.radius = radius
        elif self.data_type == 'protein':
            self.denominator = 'shortest'
        elif self.data_type == 'protein_structure':
            self.denominator = 'shortest'
            self.representation = '3di+aa'
        elif (self.data_type in ['DNA', 'RNA', 'protein'] and
              self.similarity_metric == 'needle'):
            self.denominator = 'shortest'
            self.needle_config = {
                "gapopen": 10,
                "gapextend": 0.5,
                "endweight": True,
                "endopen": 10,
                "endextend": 0.5,
                "matrix": "EBLOSUM62"
            }
        elif self.data_type == 'embedding':
            self.distance = 'cosine'


class HestiaDatasetGenerator:
    """Class for generating multiple Dataset
    partitions for generalisation evaluation.
    """
    def __init__(self, data: pd.DataFrame):
        """Initialise class

        :param data: DataFrame with the original data from which
            datasets will be generated.
        :type data: pd.DataFrame
        """
        self.data = data
        self.sim_df = None
        self.partitions = None
        print('Initialising Hestia Dataset Generator')
        print(f'Number of items in data: {len(self.data):,}')

    def get_partition(self, partition: Union[str, float]) -> dict:
        return self.partitions[partition]

    def get_partitions(self) -> dict:
        return self.partitions.items()

    def from_precalculated(self, data_path: str):
        """Load partition indexes if they have already being calculated.

        :param data_path: Path to saved partition indexes.
        :type data_path: str
        """
        with gzip.open(data_path, 'r') as fin:
            input = json.loads(fin.read().decode('utf-8'))

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
        with gzip.open(output_path, 'w') as fout:
            fout.write(json.dumps(output).encode('utf-8'))

    def calculate_similarity(self, similarity_args: SimilarityArguments):
        """Calculate pairwise similarity between all the elements in the dataset.

        :param similarity_args: See similarity arguments entry.
        :type similarity_args: SimilarityArguments
        """
        print('Calculating similarity...')
        self.sim_df = calculate_similarity(
            self.data, self.data, data_type=similarity_args.data_type,
            similarity_metric=similarity_args.similarity_metric,
            field_name=similarity_args.field_name,
            threshold=similarity_args.min_threshold,
            threads=similarity_args.threads,
            verbose=similarity_args.verbose,
            save_alignment=similarity_args.save_alignment,
            filename=similarity_args.filename,
            distance=similarity_args.distance,
            bits=similarity_args.bits,
            radius=similarity_args.radius,
            denominator=similarity_args.denominator,
            representation=similarity_args.representation,
            config=similarity_args.needle_config
        )
        print('Similarity successfully calculated!')

    def load_similarity(self, output_path: str):
        """Load similarity calculation from file.

        :param output_path: File with similarity calculations.
        :type output_path: str
        """
        print('Loading precalculated similarity...')
        self.sim_df = pd.read_csv(output_path, compression='gzip')
        print('Precalculated similarity loaded successfully!')

    def calculate_partitions(
        self,
        label_name: str = None,
        min_threshold: float = 0.3,
        threshold_step: float = 0.05,
        test_size: float = 0.2,
        valid_size: float = 0.1,
        partition_algorithm: str = 'ccpart',
        random_state: int = 42,
        similarity_args: SimilarityArguments = SimilarityArguments(),
        n_partitions: Optional[int] = None
    ):
        """Calculate partitions

        :param label_name:  Name of the field with the label information
        (only use if labels are categorical) (e.g., `class` or `bioactivity`),
        defaults to None
        :type label_name: str, optional
        :param min_threshold: Minimum threshold for the partitions, defaults to 0.3
        :type min_threshold: float, optional
        :param threshold_step: Step between each partition similarity threshold, defaults to 0.05
        :type threshold_step: float, optional
        :param test_size: Proportion of entities to be allocated to
        test subset, defaults to 0.2
        :type test_size: float, optional
        :param valid_size: Proportion of entities to be allocated
        to validation subset, defaults to 0.1
        :type valid_size: float, optional
        :param partition_algorithm: Algorithm for generating the partitions.
        Options:
            - `ccpart`
            - `graphpart`
        Defaults to 'ccpart'
        :type partition_algorithm: str, optional
        :param random_state: Seed for pseudo-random number
        generator algorithm, defaults to 42
        :type random_state: int, optional
        :param similarity_args: See similarity arguments entry, defaults to SimilarityArguments()
        :type similarity_args: SimilarityArguments, optional
        :param n_partitions: Number of partitions to generate, only works with graphpart partitioning algorithm
        :type n_partitions: int, optional
        :raises ValueError: Partitioning algorithm not supported.
        """
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
            'similarity_metric': {
                'data_type': similarity_args.data_type,
                'similarity_metric': similarity_args.similarity_metric,
                'min_threshold': similarity_args.min_threshold,
                'distance': similarity_args.distance,
                'bits': similarity_args.bits,
                'radius': similarity_args.radius,
                'denominator': similarity_args.denominator,
                'representation': similarity_args.representation,
                'needle_config': similarity_args.needle_config
            }
        }
        self.partitions = {}
        if self.sim_df is None:
            self.calculate_similarity(similarity_args)
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
                    label_name=label_name, test_size=test_size,
                    threshold=th / 100,
                    sim_df=self.sim_df, verbose=2
                )
                th_parts = (train, test)
            elif partition_algorithm == 'graph_part':
                try:
                    th_parts, clusters = graph_part(
                        self.data,
                        label_name=label_name,
                        test_size=test_size if n_partitions is None else 0.0,
                        threshold=th / 100,
                        sim_df=self.sim_df, verbose=2,
                        n_parts=n_partitions
                    )
                except RuntimeError:
                    continue

            if n_partitions is None:
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
                th_parts = [[i[0] for i in part] for part in th_parts]
                self.partitions[th / 100] = {
                    i: th_parts[i] for i in range(n_partitions)
                }

        random = random_partition(self.data, test_size=test_size,
                                  random_state=random_state)
        train_random = random_partition(
            self.data.iloc[random[0]].reset_index(drop=True),
            test_size=valid_size, random_state=random_state
        )
        self.partitions['random'] = {
            'train': train_random[0],
            'valid': train_random[1],
            'test': random[1]
        }
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

    @staticmethod
    def calculate_augood(results: dict, metric: str) -> float:
        """Calculate Area between the similarity-performance
        curve (out-of-distribution) and the in-distribution performance.

        :param results: Dictionary with key the partition (either threshold
        value or `random`) and value another dictionary with key the metric name
        and value the metric value.
        :type results: dict
        :param metric: Name of the metric for which the AUSPC is going to be
        calculated
        :type metric: str
        :return: AUSPC value
        :rtype: float
        """
        x, y = [], []
        for key, value in results.items():
            if key == 'random':
                continue
            x.append(float(key))
            y.append(float(value[metric]))
        idxs = np.argsort(x)
        x, y = np.array(x), np.array(y)
        min_x, max_x = np.min(x), np.max(x)
        return auc(x[idxs], y[idxs]) / (max_x - min_x)

    @staticmethod
    def plot_good(results: dict, metric: str):
        """Plot the Area between the similarity-performance
        curve (out-of-distribution) and the in-distribution performance.

        :param results: Dictionary with key the partition (either threshold
        value or `random`) and value another dictionary with key the metric name
        and value the metric value.
        :type results: dict
        :param metric: Name of the metric for which the AUSPC is going to be
        plotted
        :type metric: str
        """
        import matplotlib.pyplot as plt
        x, y = [], []
        for key, value in results.items():
            if key == 'random':
                continue
            x.append(float(key))
            y.append(float(value[metric]))
        idxs = np.argsort(x)
        x, y = np.array(x), np.array(y)
        plt.plot(x[idxs], y[idxs])
        # plt.plot(x[idxs], [results['random'][metric] for _ in range(len(x))], 'r')
        plt.ylabel(f'Performance: {metric}')
        plt.xlabel(f'Threshold similarity')
        # plt.legend(['SP', 'Random'])
        # plt.ylim(0, 1.1)
        # plt.show()

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
