### Training the Base Model
[This implementation of NeuroSAT](https://github.com/ryanzhangfan/NeuroSAT) is used.

#### Training
Install a SAT solver:
```bash
cd src
git clone https://github.com/liffiton/PyMiniSolvers.git
cd PyMiniSolvers
make
```

Generate validation set:
```bash
mkdir data log model
mkdir data/val
python3 src/data_maker.py 'data/val/val.pkl' 'log/gen.log' 5000 12000 --min_n 40 --max_n 40
```

Train:
```bash
python3 src/train.py \
  --dim 128 \
  --n_rounds 26 \
  --epochs 200 \
  --n_pairs 100000 \
  --max_nodes_per_batch 12000 \
  --min_n 10 \
  --max_n 40 \
  --val-file 'val.pkl'
```

#### Or Use a Trained Checkpoint
The pretrained NeuroSAT checkpoint is `model/neurosat_best.pth.tar`.

### Generate Test Set and Train Set for Aggregation Model
#### Setup
Install a SAT solver:
```bash
cd src
git clone https://github.com/liffiton/PyMiniSolvers.git
cd PyMiniSolvers
make
```

#### Test Set
Generate test set following $SR(40)$:
```bash
mkdir -p data/test
python3 src/data_maker.py 'data/test/test_40.pkl' 'log/gen.log' 5000 12000 --min_n 40 --max_n 40
```

Build Ego Metamorphic Graphs for test set in MAgg(n) setting:
Change `n_vars` to $40$, `n_samples` to $n$ in `src/gendata1.py`
Change `ver` to `'test'` in `src/gendata1.py`
```bash
python3 src/gendata1.py
```


Building Ego Metamorphic Graphs for test set in MAgg(n, m) setting:
Change `n_vars` to $40$, `n_samples_lvl1` to $n$, `n_samples_lvl2` to $m$ in `src/gendata2.py`
Change `ver` to `'test'` in `src/gendata2.py`
```bash
python3 src/gendata2.py
```

#### Train Set
Generate train set following $SR(40)$:
```bash
mkdir -p data/val
python3 src/data_maker.py 'data/val/val_40.pkl' 'log/gen.log' 5000 12000 --min_n 40 --max_n 40
```

Build Ego Metamorphic Graphs for train set in MAgg(n) setting:
Change `n_vars` to $40$, `n_samples` to $n$ in `src/gendata1.py`
Change `ver` to `'val'` in `src/gendata1.py`
```bash
python3 src/gendata1.py
```


Building Ego Metamorphic Graphs for train set in MAgg(n, m) setting:
Change `n_vars` to $40$, `n_samples_lvl1` to $n$, `n_samples_lvl2` to $m$ in `src/gendata2.py`
Change `ver` to `val` in `src/gendata2.py`
```bash
python3 src/gendata2.py
```

### Train and Test the Aggregation Model
For MAgg(n) setting, see `magg1.ipynb`
For MAgg(n, m) setting, see `magg2.ipynb`
