# NeuroCLF-GNN-Benchmark

Official repository for the paper:

A Benchmark Analysis of Graph and Non-Graph Methods for *Caenorhabditis elegans* Neuron Classification

Jingqi Lu, Keqi Han, Yun Wang, Lu Mi, Carl Yang

This paper is currently under review. A link to a preprint or the final published version will be added upon availability.

## Environment Setup
First, run the following to set up the environment with necessary dependencies:

```
conda env create -f environment.yml
```

This project uses some code from [pumpprobe](https://github.com/leiferlab/pumpprobe) for data reading, which relies on the Python modules [wormdatamodel](https://github.com/leiferlab/wormdatamodel) and [wormbrain](https://github.com/leiferlab/wormbrain). Please refer to their respective READMEs for installation instructions.


## dataset

Our framework is evaluated on the datasets from [Randi et al. 2022, Nature](https://www.nature.com/articles/s41586-023-06683-4) (Neural signal propagation atlas of Caenorhabditis elegans), download dataset from this [link](https://osf.io/e2syt/files/34m5v).

For the initial setup of the dataset, please refer to this [OSF wiki](https://osf.io/e2syt/wiki?wiki=qu643). 

Additionally, you need to modify line 22 in /src/config/data_paths.py to your funatlas_list.txt path
```python
self.dataset_list_file = self._get_path(
    env_var='FUNATLAS_DATASET_LIST',
    default=Path('/your/path/funatlas_list.txt'),#change to your path
)
```

## Run Experiments

```
python main.py 
```
## Acknowledgment

This project (neuroclf-gnn-benchmark) is licensed under the GNU General Public License v3.0.

This project contains parts of the code from [pumpprobe](https://github.com/leiferlab/pumpprobe), which is also used under the GPL-3.0 license.