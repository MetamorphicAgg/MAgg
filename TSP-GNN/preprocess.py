from myloader import InstanceLoader
import pickle
import sys

if __name__ == '__main__':
    dev = float(sys.argv[1])

    # test data
    test_loader = InstanceLoader('instances/test')
    data_lst = test_loader.get_instances()
    
    n_data_lst = []
    for Ma, Mw, route in data_lst:
        n = Ma.shape[0]
        cost = sum([ Mw[min(x,y),max(x,y)] for (x,y) in zip(route,route[1:]+[route[0]]) ]) / n
        n_data_lst.append((Ma, Mw, (1 - dev) * cost, 0))
        n_data_lst.append((Ma, Mw, (1 + dev) * cost, 1))
    
    pickle.dump(n_data_lst, open('data/test/test_%.2f.pkl' % dev, 'wb'))

    # val data
    n_samples = 1024
    test_loader = InstanceLoader('instances/train')
    data_lst = test_loader.get_instances(n_samples=n_samples)
    data_lst = data_lst[:n_samples]
    
    n_data_lst = []
    for Ma, Mw, route in data_lst:
        n = Ma.shape[0]
        cost = sum([ Mw[min(x,y),max(x,y)] for (x,y) in zip(route,route[1:]+[route[0]]) ]) / n
        n_data_lst.append((Ma, Mw, (1 - dev) * cost, 0))
        n_data_lst.append((Ma, Mw, (1 + dev) * cost, 1))
    
    pickle.dump(n_data_lst, open('data/val/val_%.2f.pkl' % dev, 'wb'))
    
    