import pickle
from tqdm import tqdm
from solver import solve_sat
from trans import expand_sat_n_batch_cut
from random import randint, seed
from neurosat import NeuroSAT
from config import parser
import sys
import numpy as np

import torch
import torch.nn
seed(0)

def predict(net, data):
    outputs = net(data)
    return outputs

if __name__ == '__main__':
    args = parser.parse_args()
    net = NeuroSAT(args)
    net = net.cuda()
    model = torch.load('model/neurosat_best.pth.tar')
    net.load_state_dict(model['state_dict'])
    
    n_vars = 40
    n_samples = 10
    ver = 'test'

    xs = pickle.load(open('data/%s/%s_%d.pkl' % (ver, ver, n_vars), 'rb'))[:5000]

    probs = []
    preds = []
    gts = []
    net.eval()
    with torch.no_grad():
        for prb in tqdm(xs):
            prbs_batches = expand_sat_n_batch_cut(prb, n_samples=n_samples, n_cut=17)
            probs.append(prbs_batches)

            all_pred = [predict(net, prbs_batch) for prbs_batch in prbs_batches]
            pred = torch.concat(all_pred, dim=0)
            preds.append(pred)

            gts.append(int(prb.is_sat[0]))

    preds = [list(pred.cpu().numpy() > 0.5) for pred in preds]
    preds = [[int(itm) for itm in pred] for pred in preds]

    pickle.dump(preds, open('data/%s/%s_%d_rp_lv1_%d_5000_preds.pkl' % (ver, ver, n_vars, n_samples), 'wb'))
    pickle.dump(gts, open('data/%s/%s_%d_rp_lv1_%d_5000_gts.pkl' % (ver, ver, n_vars, n_samples), 'wb'))

