from torch_geometric.data.data import Data
import numpy as np
import torch

def del_edge(g):
    dct = g.to_dict()
    eds = dct['edge_index']
    x = dct['x']

    eds = eds.numpy()
    del_id =  np.random.randint(0, eds.shape[1])
    neds = np.concatenate([eds[:, :del_id], eds[:, del_id + 1:]], axis=1)
    
    ng = Data.from_dict({
        'edge_index': torch.tensor(neds),
        'x': x,
    })

    return ng

def add_edge(g):
    dct = g.to_dict()
    eds = dct['edge_index']
    x = dct['x']
    
    n_nodes = x.size()[0]
    u, v = np.random.choice(list(range(n_nodes)), (2, ), replace=False)

    eds = eds.numpy()
    neds = np.concatenate([eds, np.array([[u], [v]])], axis=1)
    
    ng = Data.from_dict({
        'edge_index': torch.tensor(neds),
        'x': x,
    })

    return ng

def add_node(g):
    dct = g.to_dict()
    eds = dct['edge_index']
    x = dct['x']
    
    n_nodes = x.size()[0]
    u = np.random.randint(0, n_nodes)
    x = x.numpy()
    x = np.concatenate([x, x[u:u+1, :]], axis=0)

    ng = Data.from_dict({
        'edge_index': eds,
        'x': torch.tensor(x),
    })

    return ng

def relabel(g):
    dct = g.to_dict()
    eds = dct['edge_index']
    x = dct['x']
    
    n_nodes = x.size()[0]
    u, v = np.random.randint(0, n_nodes, size=(2, ))
    
    x = x.numpy().copy()
    x[u, :] = x[v, :]

    ng = Data.from_dict({
        'edge_index': eds,
        'x': torch.tensor(x),
    })

    return ng

def havoc(g, all_trans=[del_edge, add_edge, add_node]):

    idx = np.random.randint(0, len(all_trans))
    trans_func = all_trans[idx]

    return trans_func(g)

def havoc2(g1, g2):
    if np.random.randint(0, 2) == 0:
        return havoc(g1), g2
    else:
        return g1, havoc(g2)

def havoc3(g1, g2, all_trans=[del_edge, add_edge, add_node, relabel]):
    if np.random.randint(0, 2) == 0:
        return havoc(g1, all_trans=all_trans), g2
    else:
        return g1, havoc(g2, all_trans=all_trans)


trans_mp = {
    'de': del_edge,
    'ae': add_edge,
    'an': add_node,
    'hv': havoc,
}

trans_mp2 = {
    'rl': lambda g1, g2: havoc3(g1, g2, all_trans=[relabel]),
    'hv2': havoc2,
    'hv3': havoc3,
}
