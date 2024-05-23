### Training the Base Model
[This implementation of GREED](https://github.com/idea-iitd/greed) is used.
Download the pretrained GREED model and datasets following instructions in [this link](https://github.com/idea-iitd/greed).

### Generate Test Set and Train Set for Aggregation Model

#### Test Set
Build Ego Metamorphic Graphs for test set in MAgg(n) setting, for AIDS, LINUX, IMDB:
```bash
mkdir -p data/test
python3 gendata.py aids hv2 test 0.0
python3 gendata.py linux hv2 test 0.0
python3 gendata.py imdb hv2 test 0.0
```

#### Train Set
Build Ego Metamorphic Graphs for train set in MAgg(n) setting, for AIDS, LINUX, IMDB:
```bash
mkdir -p data/val
python3 gendata.py aids hv2 val 0.5
python3 gendata.py linux hv2 val 0.5
python3 gendata.py imdb hv2 val 0.5
```

### Train and Test the Aggregation Model
See `magg1.ipynb`.