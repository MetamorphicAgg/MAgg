### Training the Base Model
[This implementation of TSP-GNN](https://github.com/machine-reasoning-ufrgs/TSP-GNN) is used.

#### Generate Train and Test Set for Base Model
Install a TSP solver [pyconcorde](https://github.com/jvkersch/pyconcorde).

Run train.py to generate train and test data:
```bash
python3 train.py -dev 0.01 -epochs 1
```

#### Training
For dev=0.01:
```bash
python3 train.py -dev 0.01 -epochs 1001
```
For dev=0.02:
```bash
python3 train.py -dev 0.02 -epochs 1001
```

#### Or Use a Trained Checkpoint
For dev=0.01, the TSP-GNN checkpoint is `training/dev=0.01/checkpoints/epoch=1000/`.
For dev=0.02, the TSP-GNN checkpoint is `training/dev=0.02/checkpoints/epoch=1000/`.

### Generate Test Set and Train Set for Aggregation Model

#### preprocess
```bash
mkdir -p data/val
mkdir -p data/test
python3 preprocess.py 0.01
python3 preprocess.py 0.02
```

#### Test Set
Build Ego Metamorphic Graphs for test set in MAgg(n) setting, for dev=0.01, 0.02:
```bash
python3 gendata.py amm,0.500 0.02 1000 test
python3 gendata.py amm,0.500 0.01 1000 test
```

#### Train Set
Build Ego Metamorphic Graphs for train set in MAgg(n) setting, for dev=0.01,0.02:
```bash
python3 gendata.py amm,0.500 0.02 1000 val
python3 gendata.py amm,0.500 0.01 1000 val
```

### Train and Test the Aggregation Model
See `magg1.ipynb`.