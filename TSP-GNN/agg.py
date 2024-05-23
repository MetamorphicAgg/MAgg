import torch
import torch.nn
from torch import Tensor
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from tqdm import tqdm
import torch.nn.functional as F
#from torch_geometric.logging import init_wandb, log
import numpy as np
import pickle
import os
import torch.nn.functional as F
from utils import gts2onehot, predict_agg1, predict_agg2

class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        dim = 2
        self.L1_agg = torch.nn.Identity()
        self.L1 = torch.nn.Linear(dim + 2, 2, bias=False)

    def forward(self, vec_cur, vec_prds):
        agged = torch.mean(self.L1_agg(vec_prds), dim=1)
        concat = torch.concat([vec_cur, agged], dim=1)
        L1 = self.L1(concat)
        return L1

def train_agg1(xs_val, gts_val, n_samples=10, batch_size=100, lr = 0.0001, epochs=1000, dump_path=os.path.join('agg_model/', 'gnn_lvl1.pt')):
    assert(xs_val.shape[0] % batch_size == 0)
    batches = [(xs_val[i * batch_size : (i + 1) * batch_size, :], 
                gts_val[i * batch_size : (i + 1) * batch_size]) 
                    for i in range(xs_val.shape[0] // batch_size)]
    
    gnn = GNN().cuda()
    optimizer = torch.optim.Adam([dict(params=gnn.parameters(), weight_decay=0.0)], lr=lr)
    gnn.load_state_dict(torch.load(os.path.join('agg_model/', 'gnn_init.pt')))

    best_train_acc = 0.0

    bar = tqdm(list(range(epochs)))
    for epoch in bar:
        total_loss = 0.0
        n_correct = 0
        
        gnn.train()
        for idx, batch in enumerate(batches):
            xs, gts = batch
            out = predict_agg1(gnn, xs, n_samples=n_samples)
            vec_lb = gts2onehot(gts)
            
            loss = F.cross_entropy(out, vec_lb)
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().cpu().numpy()
            pred = np.argmax(out.detach().cpu().numpy(), axis=1)

            n_correct += np.sum(pred == (1 - gts))
        train_acc = n_correct / len(xs_val)

        if best_train_acc <= train_acc:
            best_train_acc = train_acc
            torch.save(gnn.state_dict(), dump_path)
        
        bar.set_postfix(
            loss='%.4f' % (total_loss / len(xs_val)),
            train_acc='%.4f' % train_acc,
            best_train_acc='%.4f' % best_train_acc,)

def test_agg1(xs_test, gts_test, n_samples=10, batch_size=100, dump_path=os.path.join('agg_model/', 'gnn_lvl1.pt')):
    assert(xs_test.shape[0] % batch_size == 0)

    batches = [(xs_test[i * batch_size : (i + 1) * batch_size, :], 
                gts_test[i * batch_size : (i + 1) * batch_size]) 
                    for i in range(xs_test.shape[0] // batch_size)]
    
    gnn = GNN().cuda()
    gnn.load_state_dict(torch.load(dump_path))

    n_correct = 0
    bar = tqdm(list(enumerate(batches)))

    gnn.eval()
    for idx, batch in bar:
        xs, gts = batch
        out = predict_agg1(gnn, xs, n_samples=n_samples)
        pred = np.argmax(out.detach().cpu().numpy(), axis=1)
        n_correct += np.sum(pred == (1 - gts))
    test_acc = n_correct / len(xs_test)

    return test_acc

def train_agg2(xs_val, gts_val, n_samples_lvl1=10, n_samples_lvl2=10, batch_size=100, lr = 0.0001, epochs=1000, dump_path=os.path.join('agg_model/', 'gnn_lvl2.pt')):
    assert(xs_val.shape[0] % batch_size == 0)
    batches = [(xs_val[i * batch_size : (i + 1) * batch_size, :], 
                gts_val[i * batch_size : (i + 1) * batch_size]) 
                    for i in range(xs_val.shape[0] // batch_size)]
    
    gnn = GNN().cuda()
    optimizer = torch.optim.Adam([dict(params=gnn.parameters(), weight_decay=0.0)], lr=lr)
    gnn.load_state_dict(torch.load(os.path.join('agg_model/', 'gnn_init.pt')))

    best_train_acc = 0.0

    bar = tqdm(list(range(epochs)))
    for epoch in bar:
        total_loss = 0.0
        n_correct = 0
        
        gnn.train()
        for idx, batch in enumerate(batches):
            xs, gts = batch
            out = predict_agg2(gnn, gnn, xs, n_samples_lvl1=n_samples_lvl1, n_samples_lvl2=n_samples_lvl2)
            vec_lb = gts2onehot(gts)
            
            loss = F.cross_entropy(out, vec_lb)
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().cpu().numpy()
            pred = np.argmax(out.detach().cpu().numpy(), axis=1)

            n_correct += np.sum(pred == (1 - gts))
        train_acc = n_correct / len(xs_val)

        if best_train_acc <= train_acc:
            best_train_acc = train_acc
            torch.save(gnn.state_dict(), dump_path)
        
        bar.set_postfix(
            loss='%.4f' % (total_loss / len(xs_val)),
            train_acc='%.4f' % train_acc,
            best_train_acc='%.4f' % best_train_acc,)

def test_agg2(xs_test, gts_test, n_samples_lvl1=10, n_samples_lvl2=10, batch_size = 100, dump_path=os.path.join('agg_model/', 'gnn_lvl2.pt')):
    assert(xs_test.shape[0] % batch_size == 0)

    batches = [(xs_test[i * batch_size : (i + 1) * batch_size, :], 
                gts_test[i * batch_size : (i + 1) * batch_size]) 
                    for i in range(xs_test.shape[0] // batch_size)]
    
    gnn = GNN().cuda()
    gnn.load_state_dict(torch.load(dump_path))

    n_correct = 0
    bar = tqdm(list(enumerate(batches)))

    gnn.eval()
    for idx, batch in bar:
        xs, gts = batch
        out = predict_agg2(gnn, gnn, xs, n_samples_lvl1=n_samples_lvl1, n_samples_lvl2=n_samples_lvl2)
        pred = np.argmax(out.detach().cpu().numpy(), axis=1)
        n_correct += np.sum(pred == (1 - gts))
    test_acc = n_correct / len(xs_test)

    return test_acc

if __name__ == '__main__':
    pass

