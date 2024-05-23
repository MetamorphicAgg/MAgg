CUDA_INDEX = 0
import sys
import pickle
import numpy as np
import torch
torch.cuda.set_device(CUDA_INDEX)
torch.backends.cudnn.benchmark = True
import torch.optim
from tqdm.auto import tqdm
from neuro import models
from trans import *
from sklearn.utils import shuffle

if __name__ == '__main__':
    np.random.seed(0)
    # python3 gendata.py aids hv2 val 0.5
    print(sys.argv)

    if sys.argv[1] == 'aids':
        dataset = 'GED_AIDS700nef'
        CLASSES = 29
    elif sys.argv[1] == 'linux':
        dataset = 'GED_LINUX'
        CLASSES = 1
    elif sys.argv[1] == 'imdb':
        dataset = 'GED_IMDBMulti'
        CLASSES = 1
    
    trans = sys.argv[2]
    ver = sys.argv[3]
    portion = float(sys.argv[4])

    if ver == 'test':
        n_samples = 50
    else:
        n_samples = 50

    trans_func = trans_mp2[trans]
    
    model = models.NormGEDModel(8, CLASSES, 64, 64)
    if dataset == 'GED_AIDS700nef':
        model.load_state_dict(torch.load(f'expts/runlogs/{dataset}/1621925721.9678917/best_model.pt', map_location='cpu'))
    elif dataset == 'GED_LINUX':
        model.load_state_dict(torch.load(f'expts/runlogs/{dataset}/1621926136.1334405/best_model.pt', map_location='cpu'))
    elif dataset == 'GED_IMDBMulti':
        model.load_state_dict(torch.load(f'expts/runlogs/{dataset}/1621926070.5699093/best_model.pt', map_location='cpu'))
    else:
        raise

    if ver == 'val':
        inner_set, _ = torch.load(f'expts/data/{dataset}/train.pt', map_location='cpu')
        queries, targets, lb, ub = inner_set
        lb = lb.cpu().numpy()
        ub = ub.cpu().numpy()
        queries, targets, lb, ub = shuffle(queries, targets, lb, ub, random_state=0)
        lb = torch.tensor(lb)
        ub = torch.tensor(ub)
        n_train = int(len(inner_set[0]) * portion)
        inner_set = queries[:n_train], targets[:n_train], lb[:n_train], ub[:n_train]
    else:
        inner_set, _ = torch.load(f'expts/data/{dataset}/inner_test.pt', map_location='cpu')

    print(dataset, len(inner_set[0]))
    inner_queries, inner_targets, lb, ub = inner_set

    pairs = [(inner_queries[i], inner_targets[i]) for i in range(len(inner_queries))]
    new_pairs = []
    for g1, g2 in tqdm(pairs):
        new_pairs.append((g1, g2))
        for _ in range(n_samples):
            new_pairs.append(trans_func(g1, g2))
    new_inner_queries = list(map(lambda x: x[0], new_pairs))
    new_inner_targets = list(map(lambda x: x[1], new_pairs))
    
    preds = model.predict_inner(new_inner_queries, new_inner_targets, batch_size=1000)
    gts = (ub + lb) / 2
    print(preds)

    if ver == 'val':
        pickle.dump(preds.cpu().numpy(), open('data/%s/%s_%s_%s_%d_%.2ftrain_preds.pkl' % (ver, ver, dataset, trans, n_samples, portion), 'wb'))
        pickle.dump(gts.cpu().numpy(), open('data/%s/%s_%s_%.2ftrain_gts.pkl' % (ver, ver, dataset, portion), 'wb'))
    else:
        pickle.dump(preds.cpu().numpy(), open('data/%s/%s_%s_%s_%d_preds.pkl' % (ver, ver, dataset, trans, n_samples), 'wb'))
        pickle.dump(gts.cpu().numpy(), open('data/%s/%s_%s_gts.pkl' % (ver, ver, dataset), 'wb'))
