import numpy as np
import pickle
from model import build_network
import os
#import warnings
#warnings.filterwarnings("ignore")
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from util import load_weights
from tqdm import tqdm
import sys
from trans import *

def create_batch(instances):
    # instance = (Ma, Mw, cost, sat)
    n_vertices  = np.array([ x[0].shape[0] for x in instances ])
    n_edges     = np.array([ len(np.nonzero(x[0])[0]) for x in instances ])
    total_vertices  = sum(n_vertices)
    total_edges     = sum(n_edges)

    EV              = np.zeros((total_edges,total_vertices))
    W               = np.zeros((total_edges,1))
    C               = np.zeros((total_edges,1))

    route_exists = np.array([x[3] for x in instances])

    for i, (Ma, Mw, cost, sat) in enumerate(instances):
        n, m = n_vertices[i], n_edges[i]
        n_acc = sum(n_vertices[0:i])
        m_acc = sum(n_edges[0:i])
        edges = list(zip(np.nonzero(Ma)[0], np.nonzero(Ma)[1]))
        for e,(x,y) in enumerate(edges):
            EV[m_acc+e,n_acc+x] = 1
            EV[m_acc+e,n_acc+y] = 1
            W[m_acc+e] = Mw[x,y]
        C[m_acc:m_acc+m,0] = cost

    return EV, W, C, route_exists, n_vertices, n_edges

def model_predict(sess, model, batch):
    batch = create_batch(batch)
    EV, W, C, route_exists, n_vertices, n_edges = batch
    feed_dict = {
        model['EV']: EV,
        model['W']: W,
        model['C']: C,
        model['time_steps']: 32,
        model['route_exists']: route_exists,
        model['n_vertices']: n_vertices,
        model['n_edges']: n_edges
    }
    preds = sess.run(model['predictions'], feed_dict = feed_dict)

    return (preds >= 0.5).astype(int)

def trans_batched(instance, n_samples=10, del_portion = 2, n_cut = 10, trans_func=del_edge):
    all_res = [[instance]]
    res = []
    for _ in range(n_samples):
        res.append(trans_func(instance, del_portion = del_portion))
        if len(res) == n_cut:
            all_res.append(res)
            res = []
    if len(res) != 0:
        all_res.append(res)
        res = []
    return all_res

if __name__ == '__main__':
    np.random.seed(0)

    print(sys.argv)
    # python3 gendata.py amm,0.500 0.02 1000 test
    
    trans = sys.argv[1]
    dev = float(sys.argv[2])
    checkpoint_version = str(sys.argv[3])
    if len(sys.argv) >= 5:
        ver = sys.argv[4]
    else:
        ver = 'val'
    
    isTest = False
    if len(sys.argv) >= 6 and sys.argv[5] == 't':
        isTest = True
    
    n_samples = 10 #if ver == 'val' else 50
    checkpoint_dir = 'training/dev=%.2f/checkpoints/epoch=%s' % (dev, checkpoint_version)

    xs = pickle.load(open('data/%s/%s_%.2f.pkl' % (ver, ver, dev), 'rb'))

    preds = []
    gts = []
    
    config = tf.ConfigProto( device_count = {'GPU':0})
    with tf.Session(config=config) as sess:
        GNN = build_network(64)
        sess.run( tf.global_variables_initializer() )
        load_weights(sess, checkpoint_dir)

        for instance in tqdm(xs):
            trans_func, del_portion = trans.split(',')
            del_portion = float(del_portion)

            batches = trans_batched(instance, n_samples=n_samples, del_portion=del_portion, n_cut=17, trans_func=trans_mp[trans_func])
            pred = [model_predict(sess, GNN, batch) for batch in batches]
            pred = np.concatenate(pred, axis=0)

            if isTest:
                print(pred, instance[3])
                input()

            preds.append(pred)
            gts.append(instance[3])
        
    pickle.dump(preds, open('data/%s/%s_%.2f_%d_%s_%s_%d_preds.pkl' % (ver, ver, dev, len(xs), checkpoint_version, trans, n_samples), 'wb'))
    pickle.dump(gts, open('data/%s/%s_%.2f_%d_gts.pkl' % (ver, ver, dev, len(xs)), 'wb'))
