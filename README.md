# BitCoin Fraud Detection 

## Important Disclaimer

This code was written by IBM Research, Singapore, for research purposes; it is not meant for any commercial purposes at the moment.
However, do note that code is provided purely for our research collaboration, and it should be treated with *confidentiality*. This code:
- should be treated long the same lines as how code is treated during a paper review process.
- should not be copied, or re-purposed, for any other use. 
- should not be made available to general public, (e.g., please do not host code in a public github.com repository, please do not upload the data for public download).

### Maintainers

Any questions contact:
- Laura Wynter (Manager): lywnter@sg.ibm.com
- Fabian Lim: flim@sg.ibm.com
- Aaron Chew: aaron.chew1@ibm.com

## Requirements

Software | Version
--|--
`python` | 3.7, 3.8
`pytorch` | 1.10.0
`cuda` | 1.11
`torch-scatter` | 2.0.9
`torch-sparse` | 0.6.13
`torch-geometric` | 2.0.4

This depends on `torch-geometric`, install as follows to use `CUDA`:

```shell
# install basic requirements
pip install -r requirements.txt

# install torch
pip install torch==1.10.0+cu111  -f https://download.pytorch.org/whl/torch_stable.html

rm /tmp/cuda-installer.log # sometimes cause problems
conda install cudatoolkit=11.1 cudatoolkit-dev=11.1 -c conda-forge

# install scatter, sparse and geometric
pip install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.10.0+cu111.html
pip install torch-sparse==0.6.13 -f https://data.pyg.org/whl/torch-1.10.0+cu111.html
pip install torch-geometric==2.0.4
```

## Data

Data is distributed via [Box](https://ibm.ent.box.com/folder/183567439277). If access is needed contact us at the [coordinates](#maintainers) above. We provide:
- **Tables**: obtained from dumping bitcoin data over a period of `2020-03` to `2020-08`. These include around 47 million transactions.
- **Dataset**: prepared from **Tables** to obtain around 477 graphs examples. 
    * Each graph has about 100K transactions (around 8 hours).
    * labels obtained ing BGS fradulent addresses.

Badawi, Emad, Guy-Vincent Jourdan, Gregor Bochmann, and Iosif-Viorel Onut. “An Automatic Detection and Analysis of the Bitcoin Generator Scam.” In 2020 IEEE European Symposium on Security and Privacy Workshops (EuroS&PW), 407–16, 2020. https://doi.org/10.1109/EuroSPW51379.2020.00061.

This data will be used in:
1. [Feature Generation](#feature-generation): generating dataset from data tables.
    - [lib](./lib): feature generation code
2. [Model Training](#model-training): training models from dataset.
    - [train.py](./train.py): model training script

## Data Preparation
You have the option to run `./unzip_data.sh` and skip step 1 for both `Feature Generation` and `Pre-populating the dataset`. 

Before running `./unzip_data.sh`,

1. Download the data from Box
2. Place the tar files in the folders `tables` and `dataset` into `data/`. e.g. `data/tables/` & `data/dataset/`
3. Run `./unzip_data.sh`

## Feature Generation

Download the `tables` folder from Box and unzip them. 
- during feature generation, *additional Tables* (see below) will be generated.

Size (uncompressed) | Tables 
--|--
 to be generated |  1day_window
 to be generated |  not_windowed
3.4G |  txs
11G  |   vin
15G  |   vout
4.4G | database.db 

To generate features:
1. Uncompress `vin,vout,txs` tables:
    ```sh
    export DATA_TABLES=data/tables

    # download the compressed folders in DATA_TABLES path given above

    # no decompression needed for database.db
    tar -zxvf vin.tar.gz
    tar -zxvf vout.tar.gz
    tar -zxvf txs.tar.gz
    ```
2. Execute [run_feature_generation.py](./run_feature_generation.py) as follows:

    ```sh
    export DATA_TABLES=data/tables

    python run_feature_generation.py \
        -path $DATA_TABLES \
        -feat features_not_windowed_one \
            features_windowed_one \
            features_addr_one \
            features_addr_two 
    ```

Note:
- `features_not_windowed_one, features_windowed_one, features_addr_one, features_addr_twe` are additional sub-Tables found in `not_windowed, 1day_windowed`.
- They will be populated in the `not_windowed, 1day_windowed` folders.
- Expect this process to be slow; e.g., `features_not_windowed_one` took us around 16 hours on a beefy machine.

## Model Training

To get the **DataSet** to train the model, either:
1. Generate it from **Tables**.
2. Download it from Box.

### Pre-populating the dataset

Size (Uncompressed) | Dataset
--|--
24K | partitions.parquet
7.2G | cache/features
675M | cache/edges
3.6G | labels

1. [OPTIONAL] if downloading **Dataset** from Box, do:
    ```sh
    export DATA_SET=data/dataset

    # download the compressed folders in DATA_SET path given above
    mkdir -p $DATA_SET/cache
    tar -xvzf $DATA_SET/edges.tar.gz -C $DATA_SET/cache &
    tar -xvzf $DATA_SET/features.tar.gz -C $DATA_SET/cache & 
    tar -xvzf $DATA_SET/labels.tar.gz -C $DATA_SET/ &
    ```
2. Train the model
    ```sh
    # point to a model, say GAT
    export MODEL=gat

    # the class weight is the num_of_class_1 / num_of_class_0
    # we ignore the unlabelled class 2 when computing this

    # DATA_SET will be regenerated from DATA_TABLES if step 1 is skipped
    python train.py \
        $MODEL \
        -model_path data/model \
        -data_path $DATA_SET \
        -features_dir $DATA_TABLES \
        -class_weight 137

    # will populate the model in data/models/$MODEL.pt
    ```

Neural network models taken from [torch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/):
- [GCN](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCNConv): Graph convoluntional network, Kipf & Welling, [paper](https://arxiv.org/abs/1609.02907).
- [GAT](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GATConv): Graph Attention, Veličković, et. al, [paper](https://arxiv.org/abs/1710.10903).

