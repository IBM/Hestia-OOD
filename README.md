<div align="center">
  <h1>Hestia</h1>

  <p>Computational tool for generating generalisation-evaluating evaluation sets.</p>
  
  <a href="https://ibm.github.io/Hestia-OOD/"><img alt="Tutorials" src="https://img.shields.io/badge/docs-tutorials-green" /></a>
  <a href="https://github.com/IBM/Hestia-OOD/blob/main/LICENSE"><img alt="GitHub" src="https://img.shields.io/github/license/IBM/Hestia-OOD" /></a>
  <a href="https://pypi.org/project/hestia-ood/"><img src="https://img.shields.io/pypi/v/hestia-ood" /></a>
  <a href="https://pypi.org/project/hestia-ood/"><img src="https://img.shields.io/pypi/dm/hestia-ood" /></a>

</div>

- **Documentation:**  <a href="https://ibm.github.io/Hestia-OOD/" target="_blank">https://ibm.github.io/Hestia-OOD</a>
- **Source Code:** <a href="https://github.com/IBM/Hestia-OOD" target="_blank">https://github.com/IBM/Hestia-OOD</a>
- **Webserver:** <a href="http://peptide.ucd.ie/Hestia" target="_blank">http://peptide.ucd.ie/Hestia</a>

## Contents

<details open markdown="1"><summary><b>Table of Contents</b></summary>

- [Intallation Guide](#installation)
- [Documentation](#documentation)
- [Examples](#examples)
- [License](#license)
 </details>


 ## Installation <a name="installation"></a>

Installing in a conda environment is recommended. For creating the environment, please run:

```bash
conda create -n autopeptideml python
conda activate autopeptideml
```

### 1. Python Package

#### 1.1.From PyPI


```bash
pip install hestia-ood
```

#### 1.2. Directly from source

```bash
pip install git+https://github.com/IBM/Hestia-OOD
```

### 3. Optional dependencies

#### 3.1. Molecular similarity

RDKit is a dependency necessary for calculating molecular similarities:

```bash
pip install rdkit
```

#### 3.2. Sequence alignment

For using MMSeqs as alignment algorithm is necessary install it in the environment:

```bash 
conda install -c bioconda mmseqs2
```

For using Needleman-Wunch:

```bash
conda install -c bioconda emboss
```

If installation not in conda environment, please check installation instructions for your particular device:

- Linux:
  ```bash
  wget https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz
  tar xvfz mmseqs-linux-avx2.tar.gz
  export PATH=$(pwd)/mmseqs/bin/:$PATH
  ```

  ```bash
  sudo apt install emboss
  ```

  ```bash
  sudo apt install emboss
  ```

- Windows: Download binaries from [EMBOSS](https://emboss.sourceforge.net/download/) and [MMSeqs2-latest](https://mmseqs.com/latest/mmseqs-win64.zip)

- Mac:
  ```bash
  sudo port install emboss
  brew install mmseqs2
  ```

## Documentation <a name="documentation"></a>

### 1. DatasetGenerator

The HestiaDatasetGenerator allows for the easy generation of training/validation/evaluation partitions with different similarity thresholds. Enabling the estimation of model generalisation capabilities. It also allows for the calculation of the ABOID (Area between the similarity-performance curve (Out-of-distribution) and the In-distribution performance).

```python
from hestia.dataset_generator import HestiaDatasetGenerator, SimilarityArguments

# Initialise the generator for a DataFrame
generator = HestiaDatasetGenerator(df)

# Define the similarity arguments (for more info see the documentation page https://ibm.github.io/Hestia-OOD/datasetgenerator)
args = SimilarityArguments(
    data_type='protein', field_name='sequence',
    similarity_metric='mmseqs2+prefilter', verbose=3,
    save_alignment=True
)

# Calculate the similarity
generator.calculate_similarity(args)

# Load pre-calculated similarities
generator.load_similarity(args.filename + '.csv.gz')

# Calculate partitions
generator.calculate_partitions(min_threshold=0.3,
                               threshold_step=0.05,
                               test_size=0.2, valid_size=0.1)

# Save partitions
generator.save_precalculated('precalculated_partitions.gz')

# Load pre-calculated partitions
generator.from_precalculated('precalculated_partitions.gz')

# Training code
# ...

# Calculate ABOID

generator.calculate_aboid(results, 'test_mcc')

# Plot ABOID
generator.plot_aboid(results, 'test_mcc')
```

### 2. Similarity calculation

Calculating pairwise similarity between the entities within a DataFrame `df_query` or between two DataFrames `df_query` and `df_target` can be achieved through the `calculate_similarity` function:

```python
from hestia.similarity import calculate_similarity
import pandas as pd

df_query = pd.read_csv('example.csv')

# The CSV file needs to have a column describing the entities, i.e., their sequence, their SMILES, or a path to their PDB structure.
# This column corresponds to `field_name` in the function.

sim_df = calculate_similarity(df_query, species='protein', similarity_metric='mmseqs+prefilter',
                              field_name='sequence')
```

More details about similarity calculation can be found in the [Similarity calculation documentation](https://ibm.github.io/Hestia-OOD/similarity/).

### 3. Clustering

Clustering the entities within a DataFrame `df` can be achieved through the `generate_clusters` function:

```python
from hestia.similarity import calculate_similarity
from hestia.clustering import generate_clusters
import pandas as pd

df = pd.read_csv('example.csv')
sim_df = calculate_similarity(df, species='protein', similarity_metric='mmseqs+prefilter',
                              field_name='sequence')
clusters_df = generate_clusters(df, field_name='sequence', sim_df=sim_df,
                                cluster_algorithm='CDHIT')
```

There are three clustering algorithms currently supported: `CDHIT`, `greedy_cover_set`, or `connected_components`. More details about clustering can be found in the [Clustering documentation](https://ibm.github.io/Hestia-OOD/clustering/).


### 4. Partitioning

Partitioning the entities within a DataFrame `df` into a training and an evaluation subsets can be achieved through 4 different functions: `ccpart`, `graph_part`, `reduction_partition`, and `random_partition`. An example of how `cc_part` would be used is:

```python
from hestia.partition import ccpart
import pandas as pd

df = pd.read_csv('example.csv')
train, test = cc_part(df, species='protein', similarity_metric='mmseqs+prefilter',
                      field_name='sequence', threshold=0.3, test_size=0.2)

train_df = df.iloc[train, :]
test_df = df.iloc[test, :]
```

License <a name="license"></a>
-------
Hestia is an open-source software licensed under the MIT Clause License. Check the details in the [LICENSE](https://github.com/IBM/Hestia/blob/master/LICENSE) file.

