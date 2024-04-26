import gzip
import json
from multiprocessing import cpu_count

import pandas as pd
from sklearn.metrics import auc
from tqdm import tqdm

from hestia.similarity import calculate_similarity
from hestia.partition import random_partition, ccpart, graph_part


class SimilarityArguments:
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
        distance: str = 'tanimoto',
        bits: str = 1024,
        radius: int = 2,
        denominator: str = 'shortest',
        representation: str = '3di+aa',
        config: dict = {
            "gapopen": 10,
            "gapextend": 0.5,
            "endweight": True,
            "endopen": 10,
            "endextend": 0.5,
            "matrix": "EBLOSUM62"
        }
    ):
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
        self.config = config


class HestiaDatasetGenerator:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.sim_df = None
        self.partitions = None
        print('Initialising Hestia Dataset Generator')
        print(f'Number of items in data: {len(self.data)}')

    def from_precalculated(self, data_path: str):
        with gzip.open(data_path, 'r') as fin:
            self.partitions = json.loads(fin.read().decode('utf-8'))
        new_dict = {}
        for key, value in self.partitions.items():
            if key != 'random':
                new_dict[float(key)] = value
            else:
                new_dict[key] = value
        self.partitions = new_dict

    def save_precalculated(self, output_path: str):
        with gzip.open(output_path, 'w') as fout:
            fout.write(json.dumps(self.partitions).encode('utf-8'))

    def calculate_similarity(self, similarity_args: SimilarityArguments):
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
            config=similarity_args.config
        )
        print('Similarity successfully calculated!')

    def load_similarity(self, output_path: str):
        print('Loading precalculated similarity...')
        self.sim_df = pd.read_csv(output_path, compression='gzip')
        print('Precalculated similarity loaded successfully!')

    def calculate_partitions(
        self,
        label_name: str = None,
        min_threshold: float = 0.3,
        threshold_step: float = 0.1,
        test_size: float = 0.2,
        valid_size: float = 0.1,
        partition_algorithm: str = 'ccpart',
        random_state: int = 42,
        similarity_args: SimilarityArguments = SimilarityArguments()
    ):
        print('Calculating partitions...')
        self.partitions = {}
        if self.sim_df is None:
            self.calculate_similarity(similarity_args)
        if partition_algorithm == 'ccpart':
            partition_algorithm = ccpart
        elif partition_algorithm == 'graph_part':
            partition_algorithm = graph_part
        else:
            raise ValueError(
                f'Partition algorithm: {partition_algorithm} is not ' +
                'supported. Try using: `ccpart` or `graph_part`.'
            )
        min_threshold = int(min_threshold * 100)
        threshold_step = int(threshold_step * 100)

        for th in tqdm(range(min_threshold, 100, threshold_step)):
            th_parts = partition_algorithm(
                self.data,
                label_name=label_name, test_size=test_size,
                valid_size=valid_size, threshold=th / 100,
                sim_df=self.sim_df
            )
            train_th_parts = random_partition(
                self.data.iloc[th_parts[0]].reset_index(drop=True),
                test_size=valid_size, random_state=random_state
            )
            self.partitions[th / 100] = {
                'train': train_th_parts[0],
                'valid': train_th_parts[1],
                'test': th_parts[1]
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
            except ImportError:
                raise ImportError(
                    f"This dataset_type: {dataset_type} requires `datasets` " +
                    "to be installed. Install using: `pip install datasets`"
                    )
            for key, value in self.partitions[threshold].items():
                ds[key] = datasets.Dataset.from_pandas(
                    self.data.iloc[value].reset_index()
                )
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

    def calculate_auspc(self, results: dict, metric: str):
        x, y = [], []
        for key, value in results.items():
            if key == 'random':
                continue
            x.append(key)
            y.append(results['random'][metric] - value[metric])
        return auc(x, y)


if __name__ == '__main__':
    df = pd.read_csv('dili.tab', sep='\t')
    generator = HestiaDatasetGenerator(df)
    args = SimilarityArguments(
        data_type='small_molecule', field_name='Drug',
        similarity_metric='fingerprint', verbose=3,
        save_alignment=True
    )
    generator.calculate_similarity(args)
    generator.load_similarity(args.filename + '.csv.gz')
    generator.calculate_partitions('Y', min_threshold=0.3,
                                   threshold_step=0.05,
                                   test_size=0.2, valid_size=0.1)
    generator.save_precalculated('precalculated_partitions.gz')
    generator.from_precalculated('precalculated_partitions.gz')
    ds = generator.generate_datasets('torch', 0.35)
