import numpy as np
import torch
import torch.nn.functional as F

lkup = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)

def predict_agg1(gnn, xs, n_samples=10):
    vec_cur = torch.tensor(lkup[xs[:, 0]]).cuda()
    vec_lvl1 = torch.tensor(lkup[xs[:, 1: n_samples + 1]]).cuda()
    out = gnn(vec_cur, vec_lvl1)
    return out

def predict_agg2(gnnl1, gnnl2, xs, n_samples_lvl1 = 10, n_samples_lvl2 = 10):
    vec_cur = torch.tensor(lkup[xs[:, 0]]).cuda()

    outs = []
    for idx in range(n_samples_lvl1):
        vec_lvl1 = lkup[xs[:, idx + 1]]
        vec_lvl2 = lkup[xs[:,  1 + n_samples_lvl1 + idx * n_samples_lvl2: 1 + n_samples_lvl1 + (idx + 1) * n_samples_lvl2]]
        vec_lvl1 = torch.tensor(vec_lvl1).cuda()
        vec_lvl2 = torch.tensor(vec_lvl2).cuda()

        out_lvl1 = gnnl2(vec_lvl1, vec_lvl2)
        out_lvl1 = F.softmax(out_lvl1, dim=1)
        outs.append(out_lvl1)
    outs = torch.stack(outs, dim=1)
    out = gnnl1(vec_cur, outs)

    return out

def gts2onehot(gts):
    return torch.tensor(lkup[gts]).cuda()

def trunc_pred_lvl1(xs, n_samples = 10, n_samples_all = 50):
    x_cur = xs[:, 0:1]
    x_lvl1 = xs[:, 1: 1 + n_samples_all]

    lvl1_idxs = np.random.choice(list(range(n_samples_all)), size=(n_samples, ), replace=False)
    x_lvl1 = x_lvl1[:, lvl1_idxs]
    
    return np.concatenate([x_cur, x_lvl1], axis=1)

def trunc_pred(xs, n_samples_lvl1 = 10, n_samples_lvl2 = 10, n_samples_lvl1_all = 30, n_samples_lvl2_all = 30):
    x_cur = xs[:, 0:1]
    x_lvl1 = xs[:, 1: 1 + n_samples_lvl1_all]
    x_lvl2 = np.reshape(xs[:, 1 + n_samples_lvl1_all:], (-1, n_samples_lvl1_all, n_samples_lvl2_all))
    
    lvl1_idxs = np.random.choice(list(range(n_samples_lvl1_all)), size=(n_samples_lvl1, ), replace=False)
    lvl2_idxs = np.random.choice(list(range(n_samples_lvl2_all)), size=(n_samples_lvl2, ), replace=False)
    x_lvl1 = x_lvl1[:, lvl1_idxs]
    x_lvl2 = x_lvl2[:, lvl1_idxs, :][:, :, lvl2_idxs]
    
    return np.concatenate([x_cur, x_lvl1, np.reshape(x_lvl2, (-1, n_samples_lvl1 * n_samples_lvl2))], axis=1)

