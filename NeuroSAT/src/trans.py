from mk_problem import Problem, mk_batch_problem
from random import randint
import numpy as np

def idxs2clause(L_unpack_indices, n_vars):
    iclauses = []
    cur_clause = []
    cur_clause_id = None
    for var_id, clause_id in L_unpack_indices:
        if clause_id != cur_clause_id:
            if len(cur_clause):
                iclauses.append(cur_clause)
            cur_clause = []
            cur_clause_id = clause_id
        
        v_id = (var_id % n_vars) + 1
        v_sign = 1 if var_id < n_vars else -1
        cur_clause.append(v_sign * v_id)
    if len(cur_clause):
        iclauses.append(cur_clause)
    return iclauses
def expand_sat(prb, var_id):
    assert(var_id >= 1 and var_id <= prb.n_vars)
    iclauses = idxs2clause(prb.L_unpack_indices, prb.n_vars)
    
    # let var_id be 0
    n_iclauses = []
    NotSat = False
    for c in iclauses:
        nc = []
        for w in c:
            if abs(w) != var_id:
                if abs(w) > var_id:
                    w = w - 1 if w > 0 else w + 1
                nc.append(w)
            elif w == var_id:
                continue
            elif w == -var_id:
                nc = True
                break
        if nc == True:
            continue
        elif len(nc) == 0:
            NotSat = True
            break
        else:
            n_iclauses.append(nc)
    if NotSat:
        prb1 = False
    else:
        #prb1 = get_problem(n_iclauses, prb.n_vars - 1)
        prb1 = get_problem(n_iclauses, prb.n_vars)

    # let var_id be 1
    n_iclauses = []
    NotSat = False
    for c in iclauses:
        nc = []
        for w in c:
            if abs(w) != var_id:
                if abs(w) > var_id:
                    w = w - 1 if w > 0 else w + 1
                nc.append(w)
            elif w == -var_id:
                continue
            elif w == var_id:
                nc = True
                break
        if nc == True:
            continue
        elif len(nc) == 0:
            NotSat = True
            break
        else:
            n_iclauses.append(nc)
    if NotSat:
        prb2 = False
    else:
        #prb2 = get_problem(n_iclauses, prb.n_vars - 1)
        prb2 = get_problem(n_iclauses, prb.n_vars)

    return prb1, prb2

def expand_sat_n(prb, n_samples):
    all_prbs = []
    for _ in range(n_samples):
        var_id = randint(1, prb.n_vars)
        prb1, prb2 = expand_sat(prb, var_id)
        all_prbs.append([prb1, prb2][randint(0, 1)])
    return all_prbs

def del_sat(prb, del_ratio = 0.02):
    iclauses = idxs2clause(prb.L_unpack_indices, prb.n_vars)
    n_del = int(len(iclauses) * del_ratio)
    idxs = set(np.random.choice(list(range(len(iclauses))), size=(n_del, ), replace=False))
    n_iclauses = [iclauses[idx] for idx in range(len(iclauses)) if idx not in idxs]
    return get_problem(n_iclauses, prb.n_vars)

def del_sat_n(prb, n_samples, del_ratio = 0.02):
    return [del_sat(prb, del_ratio) for _ in range(n_samples)]

#def del_sat_n(prb, n_samples=10):
#    pass

def get_problem(iclauses, n_vars):
    n_cells_per_batch = [sum(map(len, iclauses))]
    return Problem(n_vars, iclauses, [None], n_cells_per_batch, "dimacs")    

# batch problem generation
def problems_to_batch(prbs):
    #from time import monotonic as _time
    #t0 = _time()
    all_prbs = []
    for prb in prbs: 
        iclauses = idxs2clause(prb.L_unpack_indices, prb.n_vars)
        all_prbs.append(('dimacs', prb.n_vars, iclauses, None))
    #t1 = _time()
    res = mk_batch_problem(all_prbs)
    #t2 = _time()
    return res#, t1 - t0, t2 - t1

def expand_sat_for_batch(prb, var_id, iclauses):
    assert(var_id >= 1 and var_id <= prb.n_vars)
    
    if randint(0, 1) == 0:
        # let var_id be 0
        n_iclauses = []
        NotSat = False
        for c in iclauses:
            nc = []
            for w in c:
                if abs(w) != var_id:
                    if abs(w) > var_id:
                        w = w - 1 if w > 0 else w + 1
                    nc.append(w)
                elif w == var_id:
                    continue
                elif w == -var_id:
                    nc = True
                    break
            if nc == True:
                continue
            elif len(nc) == 0:
                NotSat = True
                break
            else:
                n_iclauses.append(nc)
        if NotSat:
            prb1 = False
        else:
            #prb1 = get_problem(n_iclauses, prb.n_vars - 1)
            prb1 = n_iclauses
        return prb1
    else:
        # let var_id be 1
        n_iclauses = []
        NotSat = False
        for c in iclauses:
            nc = []
            for w in c:
                if abs(w) != var_id:
                    if abs(w) > var_id:
                        w = w - 1 if w > 0 else w + 1
                    nc.append(w)
                elif w == -var_id:
                    continue
                elif w == var_id:
                    nc = True
                    break
            if nc == True:
                continue
            elif len(nc) == 0:
                NotSat = True
                break
            else:
                n_iclauses.append(nc)
        if NotSat:
            prb2 = False
        else:
            #prb2 = get_problem(n_iclauses, prb.n_vars - 1)
            prb2 = n_iclauses
        return prb2

def expand_sat_n_batch(prb, n_samples):
    all_prbs = []
    iclauses = idxs2clause(prb.L_unpack_indices, prb.n_vars)
    all_prbs.append(('dimacs', prb.n_vars, iclauses, None))
    for _ in range(n_samples):
        var_id = randint(1, prb.n_vars)
        n_iclauses = expand_sat_for_batch(prb, var_id, iclauses)
        all_prbs.append(('dimacs', prb.n_vars, n_iclauses, None))
    return mk_batch_problem(all_prbs)

def expand_sat_n_batch_cut(prb, n_samples, n_cut=10):
    problem_batches = []
    all_prbs = []
    iclauses = idxs2clause(prb.L_unpack_indices, prb.n_vars)
    all_prbs.append(('dimacs', prb.n_vars, iclauses, None))
    for _ in range(n_samples):
        var_id = randint(1, prb.n_vars)
        n_iclauses = expand_sat_for_batch(prb, var_id, iclauses)
        all_prbs.append(('dimacs', prb.n_vars, n_iclauses, None))
        if len(all_prbs) == n_cut:
            problem_batches.append(mk_batch_problem(all_prbs))
            all_prbs = []
    if len(all_prbs) != 0:
        problem_batches.append(mk_batch_problem(all_prbs))
    return problem_batches

def expand_sat_n_batch_cut_lvl2(prb, n_samples, n_samples_2, n_cut=10):
    problem_batches = []
    all_prbs = []
    iclauses = idxs2clause(prb.L_unpack_indices, prb.n_vars)
    all_prbs.append(('dimacs', prb.n_vars, iclauses, None))
    all_lvl1_clauses = []
    for _ in range(n_samples):
        n_iclauses = False
        while not n_iclauses:
            var_id = randint(1, prb.n_vars)
            n_iclauses = expand_sat_for_batch(prb, var_id, iclauses)
        all_lvl1_clauses.append(n_iclauses)
        all_prbs.append(('dimacs', prb.n_vars, n_iclauses, None))
        if len(all_prbs) == n_cut:
            problem_batches.append(mk_batch_problem(all_prbs))
            all_prbs = []
    for lvl1_iclauses in all_lvl1_clauses:
        for _ in range(n_samples_2):
            n_iclauses = False
            while not n_iclauses:
                var_id = randint(1, prb.n_vars - 1)
                n_iclauses = expand_sat_for_batch(prb, var_id, lvl1_iclauses)
            all_prbs.append(('dimacs', prb.n_vars, n_iclauses, None))
            if len(all_prbs) == n_cut:
                problem_batches.append(mk_batch_problem(all_prbs))
                all_prbs = []
    if len(all_prbs) != 0:
        problem_batches.append(mk_batch_problem(all_prbs))
    return problem_batches

def aul_sat_for_batch(prb, iclauses):
    n_vars = prb.n_vars
    new_l = n_vars + 1

    n_iclauses = []
    for c in iclauses:
        nc = [l for l in c]
        if randint(0, 10) == 0:
            n_iclauses.append(nc + [-new_l])
        else:
            n_iclauses.append(nc)
    n_iclauses.append([new_l])
    return n_iclauses

def aul_sat_n_batch_cut(prb, n_samples, n_cut=10):
    problem_batches = []
    all_prbs = []
    iclauses = idxs2clause(prb.L_unpack_indices, prb.n_vars)
    all_prbs.append(('dimacs', prb.n_vars + 1, iclauses, None))
    for _ in range(n_samples):
        n_iclauses = aul_sat_for_batch(prb, iclauses)
        all_prbs.append(('dimacs', prb.n_vars + 1, n_iclauses, None))
        if len(all_prbs) == n_cut:
            problem_batches.append(mk_batch_problem(all_prbs))
            all_prbs = []
    if len(all_prbs) != 0:
        problem_batches.append(mk_batch_problem(all_prbs))
    return problem_batches


if __name__ == '__main__':
    from time import monotonic as _time
    import pickle
    batches = pickle.load(open('val.pkl', 'rb'))
    prb = batches[0]
    prb1, prb2 = expand_sat(prb, 1)

    print(idxs2clause(prb.L_unpack_indices, prb.n_vars))
    print(idxs2clause(prb1.L_unpack_indices, prb1.n_vars))
    print(idxs2clause(prb2.L_unpack_indices, prb2.n_vars))
