import numpy as np

def del_edge(instance, del_portion = 0.01):
    Ma, Mw, cost, sat = instance
    Ma = Ma.copy()
    Mw = Mw.copy()

    idxs = np.array(np.where(Ma)).T
    #idxs = [(pr[0], pr[1]) for pr in idxs]
    edges = list(range(len(idxs)))
    del_idxs = np.random.choice(edges, size=(int(len(edges) * del_portion), ), replace=False)
    del_idxs = idxs[del_idxs]
    for x, y in del_idxs:
        Ma[x, y] = 0
        Mw[x, y] = np.inf
    return Ma, Mw, cost, sat
  
def del_surr(instance, del_portion = 0.5):
    Ma, Mw, cost, sat = instance
    Ma = Ma.copy()
    Mw = Mw.copy()

    idx = np.random.randint(0, Ma.shape[0])
    eds = np.concatenate([np.where(Ma[idx, :])[0], np.where(Ma[:, idx])[0]], axis=0)
    #print(eds)
    dels = np.random.choice(eds, size=(int(del_portion * len(eds))), replace=False)
    for idy in dels:
        Ma[idx, idy] = 0
        Ma[idy, idx] = 0
        Mw[idx, idy] = np.inf
        Mw[idy, idx] = np.inf
    return Ma, Mw, cost, sat

def del_sing(instance, del_portion = 0.5):
    Ma, Mw, cost, sat = instance
    Ma = Ma.copy()
    Mw = Mw.copy()

    idx = 0 #np.random.randint(0, Ma.shape[0])
    eds = np.concatenate([np.where(Ma[idx, :])[0], np.where(Ma[:, idx])[0]], axis=0)
    #print(eds)
    dels = np.random.choice(eds, size=(int(del_portion * len(eds))), replace=False)
    for idy in dels:
        Ma[idx, idy] = 0
        Ma[idy, idx] = 0
        Mw[idx, idy] = np.inf
        Mw[idy, idx] = np.inf
    return Ma, Mw, cost, sat

def del_remain(instance, del_portion = 2):
    Ma, Mw, cost, sat = instance
    Ma = Ma.copy()
    Mw = Mw.copy()

    idx = 0 #np.random.randint(0, Ma.shape[0])
    eds = np.concatenate([np.where(Ma[idx, :])[0], np.where(Ma[:, idx])[0]], axis=0)
    #print(eds)
    dels = np.random.choice(eds, size=(len(eds) - int(del_portion)), replace=False)
    for idy in dels:
        Ma[idx, idy] = 0
        Ma[idy, idx] = 0
        Mw[idx, idy] = np.inf
        Mw[idy, idx] = np.inf
    return Ma, Mw, cost, sat

def del_max(instance, del_portion = 0.01):
    Ma, Mw, cost, sat = instance
    Ma = Ma.copy()
    Mw = Mw.copy()

    idxs = np.array(np.where(Ma)).T
    #idxs = [(pr[0], pr[1]) for pr in idxs]
    edges = list(range(len(idxs)))
    edges = sorted(edges, key = lambda e: Mw[idxs[e, 0], idxs[e, 1]])
    edges = edges[len(edges) // 3:]

    del_idxs = np.random.choice(edges, size=(int(len(edges) * del_portion), ), replace=False)
    del_idxs = idxs[del_idxs]
    for x, y in del_idxs:
        Ma[x, y] = 0
        Mw[x, y] = np.inf
    return Ma, Mw, cost, sat

def merge_edge(instance, del_portion = 1.0):
    Ma, Mw, cost, sat = instance
    n_nodes = Ma.shape[0]
    
    Man = np.zeros(shape=(n_nodes - 1, n_nodes - 1), dtype=Ma.dtype)
    Mwn = np.zeros(shape=(n_nodes - 1, n_nodes - 1), dtype=Mw.dtype)

    idxs = np.array(np.where(Ma)).T
    d_edge = idxs[np.random.randint(0, len(idxs))]
    
    u, v = d_edge
    w = n_nodes - 2
    def newid(x):
        if x < min(u, v):
            return x
        elif x > max(u, v):
            return x - 2
        else:
            return x - 1
    other_nodes = set(range(n_nodes)) - set([u, v])
    #print(n_nodes, u, v, w, other_nodes)
    for x in other_nodes:
        for y in other_nodes:
            nx, ny = newid(x), newid(y)
            Man[nx, ny] = Ma[x, y]
            Mwn[nx, ny] = Mw[x, y]
    for x in other_nodes:
        nx = newid(x)
        Man[min(nx, w), max(nx, w)] = min(Ma[min(x, u), max(x, u)], Ma[min(x, v), max(x, v)])
        Mwn[min(nx, w), max(nx, w)] = min(Mw[min(x, u), max(x, u)], Mw[min(x, v), max(x, v)])
    #print(Ma)
    #print(Man)
    #input()
    return Man, Mwn, (cost * n_nodes - Mw[u, v]) / (n_nodes - 1), sat

def dec_cost(instance, del_portion = 0.02):
    Ma, Mw, cost, sat = instance
    return Ma, Mw, cost * (1 - np.random.uniform(0.0, del_portion)), sat

def inc_si(instance, del_portion = 0.1):
    Ma, Mw, cost, sat = instance
    n_nodes = Ma.shape[0]
    if del_portion >= 1.0:
        ch_nodes = np.random.choice(range(n_nodes), (int(del_portion), ), replace=False)
    else:
        ch_nodes = np.random.choice(range(n_nodes), (int(n_nodes * del_portion), ), replace=False)
    
    cost = cost * n_nodes
    Ma, Mw = Ma.copy(), Mw.copy()  
    for nd in ch_nodes:
        Ws = Mw[nd, :] + Mw[:, nd]
        mi = min(0.1, np.min(Ws[Ws != 0]))
        ch_w = np.random.uniform(-mi, mi)
        for u in range(n_nodes):
            if u != nd:
                Mw[min(u, nd), max(u, nd)] += ch_w
        cost += 2 * ch_w
    cost = cost / n_nodes
    return Ma, Mw, cost, sat

def del_node(instance, del_portion = 1.0):
    Ma, Mw, cost, sat = instance
    n_nodes = Ma.shape[0]
    nd = np.random.randint(0, n_nodes)

    def newid(x):
        if x < nd:
            return x
        else:
            return x - 1

    cost = cost * n_nodes
    Ma, Mw = Ma.copy(), Mw.copy()
    Man = np.zeros(shape=(n_nodes - 1, n_nodes - 1), dtype=Ma.dtype)
    Mwn = np.zeros(shape=(n_nodes - 1, n_nodes - 1), dtype=Mw.dtype)
    Mw2 = np.zeros(shape=(n_nodes - 1, n_nodes - 1), dtype=Mw.dtype)

    for u in range(n_nodes):
        for v in range(u + 1, n_nodes):
            if u != nd and v != nd:
                nu, nv = newid(u), newid(v)
                Man[nu, nv] = Ma[u, v]
                Mwn[nu, nv] = Mw[u, v]
                Mw2[nu, nv] = Mw[min(u, nd), max(u, nd)] + Mw[min(v, nd), max(v, nd)]

    dd = Mw2 - Mwn
    #dd = np.sort(dd[dd != 0])
    #chw = dd[int((len(dd) - 1) * del_portion)]
    chw = min([dd[u, v] for u in range(n_nodes - 1) for v in range(u + 1, n_nodes - 1)])
    cost = (cost * n_nodes - chw) / (n_nodes - 1)
    return Man, Mwn, cost, sat

def scale_weight(instance, del_portion = 1.0):
    Ma, Mw, cost, sat = instance
    rt = np.random.uniform(del_portion, 1.0)
    return Ma, Mw * rt, cost * rt, sat

def inc_sm(instance, del_portion = 0.1):
    Ma, Mw, cost, sat = instance
    n_nodes = Ma.shape[0]
    ch_nodes = np.random.choice(range(n_nodes), (1, ), replace=False)

    cost = cost * n_nodes
    Ma, Mw = Ma.copy(), Mw.copy()  
    for nd in ch_nodes:
        Ws = Mw[nd, :] + Mw[:, nd]
        mi = min(del_portion, np.min(Ws[Ws != 0]))
        ch_w = np.random.uniform(-mi, mi)
        for u in range(n_nodes):
            if u != nd:
                Mw[min(u, nd), max(u, nd)] += ch_w
        cost += 2 * ch_w
    cost = cost / n_nodes
    return Ma, Mw, cost, sat

def inc_sm2(instance, del_portion = 0.1):
    Ma, Mw, cost, sat = instance
    n_nodes = Ma.shape[0]
    ch_nodes = np.random.choice(range(n_nodes), (1, ), replace=False)

    cost = cost * n_nodes
    Ma, Mw = Ma.copy(), Mw.copy()  
    for nd in ch_nodes:
        Ws = Mw[nd, :] + Mw[:, nd]
        mi = min(del_portion, np.min(Ws[Ws != 0]))
        ch_w = mi #np.random.uniform(-mi, mi)
        for u in range(n_nodes):
            if u != nd:
                Mw[min(u, nd), max(u, nd)] += ch_w
        cost += 2 * ch_w
    cost = cost / n_nodes
    return Ma, Mw, cost, sat

def dec_sm2(instance, del_portion = 0.1):
    Ma, Mw, cost, sat = instance
    n_nodes = Ma.shape[0]
    ch_nodes = np.random.choice(range(n_nodes), (1, ), replace=False)

    cost = cost * n_nodes
    Ma, Mw = Ma.copy(), Mw.copy()  
    for nd in ch_nodes:
        Ws = Mw[nd, :] + Mw[:, nd]
        mi = min(del_portion, np.min(Ws[Ws != 0]))
        ch_w = mi #np.random.uniform(-mi, mi)
        for u in range(n_nodes):
            if u != nd:
                Mw[min(u, nd), max(u, nd)] -= ch_w
        cost -= 2 * ch_w
    cost = cost / n_nodes
    return Ma, Mw, cost, sat


def add_mid_once(instance):
    Ma, Mw, cost, sat = instance
    n_nodes = Ma.shape[0]
    u, v = np.random.choice(range(n_nodes), (2, ), replace=False)

    Man = np.zeros(shape=(n_nodes + 1, n_nodes + 1), dtype=Ma.dtype)
    Mwn = np.zeros(shape=(n_nodes + 1, n_nodes + 1), dtype=Mw.dtype)

    for x in range(n_nodes):
        for y in range(x + 1, n_nodes):
            Man[x, y] = Ma[x, y]
            Mwn[x, y] = Mw[x, y]
    
    t = n_nodes

    Man[u, t] = 1
    Mwn[u, t] = Mw[min(u, v), max(u, v)] / 2
    Man[v, t] = 1
    Mwn[v, t] = Mw[min(u, v), max(u, v)] / 2

    for x in range(n_nodes):
        if x != u and x != v:
            a = Mw[min(x, u), max(x, u)]
            b = Mw[min(x, v), max(x, v)]
            c = Mw[min(u, v), max(u, v)]
            Mwn[x, t] = 0.5 * np.sqrt(2 * (a ** 2 + b ** 2) - c ** 2)
            Man[x, t] = 1
    
    return Man, Mwn, cost * n_nodes / (n_nodes + 1), sat

def add_mid(instance, del_portion = 1.0):
    n_iter = int(del_portion)
    for _ in range(n_iter):
        instance = add_mid_once(instance)
    return instance

def add_mid_min(instance, del_portion = 0.1):
    Ma, Mw, cost, sat = instance
    n_nodes = Ma.shape[0]
    
    idxs = np.array(np.where(Ma)).T
    edges = list(range(len(idxs)))
    edges = sorted(edges, key=lambda x: Mw[idxs[x, 0], idxs[x, 1]])
    del_ed = np.random.choice(edges[:int(del_portion * len(edges))], size=(1, ), replace=False)[0]
    u, v = idxs[del_ed, 0], idxs[del_ed, 1]
    
    Man = np.zeros(shape=(n_nodes + 1, n_nodes + 1), dtype=Ma.dtype)
    Mwn = np.zeros(shape=(n_nodes + 1, n_nodes + 1), dtype=Mw.dtype)

    for x in range(n_nodes):
        for y in range(x + 1, n_nodes):
            Man[x, y] = Ma[x, y]
            Mwn[x, y] = Mw[x, y]
    
    t = n_nodes

    Man[u, t] = 1
    Mwn[u, t] = Mw[min(u, v), max(u, v)] / 2
    Man[v, t] = 1
    Mwn[v, t] = Mw[min(u, v), max(u, v)] / 2

    for x in range(n_nodes):
        if x != u and x != v:
            a = Mw[min(x, u), max(x, u)]
            b = Mw[min(x, v), max(x, v)]
            c = Mw[min(u, v), max(u, v)]
            Mwn[x, t] = 0.5 * np.sqrt(max(2 * (a ** 2 + b ** 2) - c ** 2, 0))
            Man[x, t] = 1
    
    return Man, Mwn, cost * n_nodes / (n_nodes + 1), sat

def add_div_min(instance, del_portion = 0.1):
    Ma, Mw, cost, sat = instance
    n_nodes = Ma.shape[0]
    
    idxs = np.array(np.where(Ma)).T
    edges = list(range(len(idxs)))
    edges = sorted(edges, key=lambda x: Mw[idxs[x, 0], idxs[x, 1]])
    del_ed = np.random.choice(edges[:int(del_portion * len(edges))], size=(1, ), replace=False)[0]
    u, v = idxs[del_ed, 0], idxs[del_ed, 1]
    
    Man = np.zeros(shape=(n_nodes + 1, n_nodes + 1), dtype=Ma.dtype)
    Mwn = np.zeros(shape=(n_nodes + 1, n_nodes + 1), dtype=Mw.dtype)

    for x in range(n_nodes):
        for y in range(x + 1, n_nodes):
            Man[x, y] = Ma[x, y]
            Mwn[x, y] = Mw[x, y]
    
    t = n_nodes

    Man[u, t] = 1
    Mwn[u, t] = Mw[min(u, v), max(u, v)] / 2
    Man[v, t] = 1
    Mwn[v, t] = Mw[min(u, v), max(u, v)] / 2
    
    l = np.random.uniform(0.0, 1.0)

    for x in range(n_nodes):
        if x != u and x != v:
            a = Mw[min(x, u), max(x, u)]
            b = Mw[min(x, v), max(x, v)]
            c = Mw[min(u, v), max(u, v)]
            Mwn[x, t] = np.sqrt(
                max(
                    l ** 2 * a ** 2 + \
                    (1 - l) ** 2 * b ** 2 + \
                    l * (1 - l) * (a ** 2 + b ** 2 - c ** 2), 0))
            Man[x, t] = 1
    
    return Man, Mwn, cost * n_nodes / (n_nodes + 1), sat


def repeat(n_repeat, instance, del_portion, trans_func=add_mid_min):
    for _ in range(n_repeat):
        #print(instance)
        instance = trans_func(instance, del_portion=del_portion)
        #print(instance)
    return instance

def func_nrepeat(n_repeat, trans_func):
    return lambda instance, del_portion: repeat(n_repeat, instance, del_portion, trans_func=trans_func)

trans_mp = {
    'dl': del_edge,
    'sr': del_surr,
    'si': del_sing,
    're': del_remain,
    'dm': del_max,
    'mr': merge_edge,
    'dc': dec_cost,
    'ii': inc_si,
    'dn': del_node,
    'sw': scale_weight,
    'im': inc_sm,
    'im2': inc_sm2,
    'am': add_mid,
    'amm': add_mid_min,
    'adm': add_div_min,
    'dm2': dec_sm2,
}

for i in range(1, 31):
    trans_mp['rp%damm' % i] = func_nrepeat(i, add_mid_min)
    trans_mp['rp%dadm' % i] = func_nrepeat(i, add_div_min)
