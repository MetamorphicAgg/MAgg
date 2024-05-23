import torch
import numpy as np
from neuro.metrics import rmse
from scipy.stats import kendalltau

bins = np.arange(0.0, 1.0, 0.005)
lkup = np.identity(bins.shape[0] + 1)
n_disc = bins.shape[0]

def disc(preds):
    idxs = np.digitize(preds, bins=bins)
    return np.mean(lkup[idxs], axis=1)
def prd2XY(preds, gts, n_samples=10):
    n_train = gts.shape[0]
    preds = np.reshape(preds, (n_train, n_samples + 1))
    mi = np.min(preds, axis=1)
    ma = np.max(preds, axis=1)

    Xs = (preds - mi[:, np.newaxis]) / (ma[:, np.newaxis] - mi[:, np.newaxis])
    Xs = disc(Xs[:, 1:])
    oYs = (preds[:, 0] - mi) / (ma - mi)
    Xs = np.concatenate([disc(oYs[:, np.newaxis]), Xs], axis=1)

    Ys = (gts - mi) / (ma - mi)

    return Xs, Ys, mi, ma, oYs

def test(model, Xs, lb, ub, mi, ma, oYs):
    Ys = model.predict(Xs)
    y_pred = Ys * (ma - mi) + mi
    orig_pred = oYs * (ma - mi) + mi

    err = rmse(lb, ub, torch.tensor(y_pred))
    err_orig = rmse(lb, ub, torch.tensor(orig_pred))

    return err_orig.cpu().numpy(), err.cpu().numpy()

def test_tau(model, Xs, lb_outer, ub_outer, mi, ma, oYs):
    Ys = model.predict(Xs)
    y_pred = Ys * (ma - mi) + mi
    orig_pred = oYs * (ma - mi) + mi
    
    n_query, n_target = lb_outer.size()
    outer_pred = np.reshape(y_pred, (n_target, n_query)).T
    orig_outer_pred = np.reshape(orig_pred, (n_target, n_query)).T
    
    ged = (lb_outer + ub_outer)/2

    tmp = []
    for i in range(n_query):
        tmp.append(kendalltau(ged[i], outer_pred[i], nan_policy='omit')[0])
    tau = sum(tmp)/len(tmp)
    tmp = []
    for i in range(n_query):
        tmp.append(kendalltau(ged[i], orig_outer_pred[i], nan_policy='omit')[0])
    orig_tau = sum(tmp)/len(tmp)
    return orig_tau, tau
    #return err_orig.cpu().numpy(), err.cpu().numpy()

def trunc_pred_lvl1(xs, n_samples = 10, n_samples_all = 50):
    x_cur = xs[:, 0:1]
    x_lvl1 = xs[:, 1: 1 + n_samples_all]

    lvl1_idxs = np.random.choice(list(range(n_samples_all)), size=(n_samples, ), replace=False)
    x_lvl1 = x_lvl1[:, lvl1_idxs]
    
    return np.concatenate([x_cur, x_lvl1], axis=1)

dataset_mp = {
    'aids': 'GED_AIDS700nef',
    'linux': 'GED_LINUX',
    'imdb': 'GED_IMDBMulti',
}
