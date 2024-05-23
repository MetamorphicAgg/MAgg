import matplotlib.pyplot as plt
import torch
from utils import n_disc
from tqdm import tqdm
import numpy as np

class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes = [50], output_size=1, p=0.5):
        super(MLP, self).__init__()
        sizes = [input_size] + hidden_sizes

        for i in range(len(hidden_sizes)):
            setattr(self, 'fc%d' % i, torch.nn.Linear(sizes[i], sizes[i + 1]))
        
        self.fcs = [getattr(self, 'fc%d' % i) for i in range(len(hidden_sizes))]
        self.act = torch.nn.ReLU()
        self.final_fc = torch.nn.Linear(sizes[-1], output_size)
        self.dropout = torch.nn.Dropout(p=p)
    
    def forward(self, x):
        for fc in self.fcs:
            x = fc(x)
            x = self.act(x)
            x = self.dropout(x)
        x = self.final_fc(x)
        return x

class MLPRegressor:
    def __init__(self, hidden_sizes=[50], p=0.5, input_size=(n_disc + 1) * 2):
        self.mlp = MLP(input_size=input_size, hidden_sizes=hidden_sizes, p=p).cuda()
    def predict(self, X):
        self.mlp.eval()
        X = torch.tensor(X, dtype=torch.float32).cuda()
        Y = self.mlp(X)
        return Y.detach().cpu().numpy()[:, 0]
    def fit(self, X, Y, batch_size=10000, n_epochs=100, lr=0.001, weight_decay=0.001, plot=True):
        n_batch = X.shape[0] // batch_size \
            if X.shape[0] % batch_size == 0 \
            else X.shape[0] // batch_size + 1
        self.X_batch = [torch.tensor(X[i * batch_size: (i + 1) * batch_size, :], dtype=torch.float32).cuda() for i in range(n_batch)]
        self.Y_batch = [torch.tensor(Y[i * batch_size: (i + 1) * batch_size, np.newaxis], dtype=torch.float32).cuda() for i in range(n_batch)]

        optimizer = torch.optim.Adam(self.mlp.parameters(), lr=lr, weight_decay=weight_decay)
        
        loss_fn =  torch.nn.MSELoss()
        
        losses_train = []
        epochs = tqdm(list(range(n_epochs)))
        
        for epoch in epochs:
            all_loss = 0.0
            all_size = 0
            self.mlp.train()
            bar = list(range(n_batch))
            for i in bar:
                Xs, Ys = self.X_batch[i], self.Y_batch[i]
                pred = self.mlp(Xs)
                loss = loss_fn(pred, Ys)

                #scheduler.step()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                all_loss += loss.detach().cpu().numpy() * Ys.size()[0]
                all_size += Ys.size()[0]
            train_loss = all_loss / all_size
            
            losses_train.append(train_loss)

            epochs.set_postfix(train_mse=train_loss)    
            torch.cuda.empty_cache()
        if plot:
            plt.plot(list(epochs), losses_train, label='train')
            plt.legend()
        