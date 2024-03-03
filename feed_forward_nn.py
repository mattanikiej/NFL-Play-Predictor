import torch
from torch import nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import datetime
from uuid import uuid4
import numpy as np
import matplotlib.pyplot as plt

class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(Model, self).__init__()

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.layers = nn.Sequential(
          nn.Linear(input_size, hidden_dim),
          nn.LeakyReLU(),
        )

        for _ in range(n_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.LeakyReLU())

        self.layers.append(nn.Linear(hidden_dim, output_size))
        self.layers.append(nn.Sigmoid())

    def forward(self, x):
        x = self.layers(x)
        return x


class PassRunDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):

        self.X = x.values
        self.y = y.values

    def __len__(self):
        return self.X.shape[0]
    
    def n_features(self):
        return self.X.shape[1]

    def __getitem__(self, idx):
        data = self.X[idx]
        label = self.y[idx]

        data = torch.tensor(data, dtype=torch.float32)
        # print("LABEL:", label)
        label = torch.tensor(int(label[0]), dtype=torch.long)
        return data, label
    

def get_data(cols=None):
    features = pd.read_csv("./data/scaled_features.csv")

    if cols is not None:
        features = features[cols]

    labels = pd.read_csv("./data/labels.csv")

    return train_test_split(features, labels)

def train(dataloader, model, lf, opt, val_dataloader):
    
    train_loss = []
    train_acc = []

    valid_loss = []
    valid_acc = []

    now = datetime.datetime.now()
    for batch, (X, y) in enumerate(dataloader):
        model.train()

        X = X.to(model.device)
        y = y.to(model.device)

        # make some predictions and get the error
        pred = torch.round(model(X)).squeeze()
        loss = lf(pred.float(), y.float())

        # backpropogation
        opt.zero_grad()
        loss.backward()
        opt.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            correct = torch.eq(y, pred).sum().item() 
            accuracy = (correct / len(pred)) * 100 
            v_loss, v_acc, _, _ = test(val_dataloader, model, lf)
            iters = 10 * len(X)
            then = datetime.datetime.now()
            iters /= (then - now).total_seconds()
            print(f"train loss: {loss:>6f} train accuracy: {accuracy:>6f} valid loss: {v_loss:>6f} valid accuracy: {v_acc:>6f} [{current:>5d}/{19000}] ({iters:.1f} its/sec)")
            now = then
            train_loss.append(loss)
            train_acc.append(accuracy)
            valid_loss.append(v_loss)
            valid_acc.append(v_acc)

    return np.mean(train_loss), np.mean(train_acc), np.mean(valid_loss), np.mean(valid_acc)

def test(dataloader, model, lf):
    num_batches = 0
    model.eval()
    correct = 0
    test_acc = []
    test_loss = []
    # get test labels and predictions
    test_labels = []
    pred_labels = []
    for X, y in dataloader:
        X = X.to(model.device)
        y = y.to(model.device)
        pred = torch.round(model(X)).squeeze()
        loss = lf(pred.float(), y.float()).item()
        test_loss.append(loss)
        correct = torch.eq(y, pred).sum().item() 
        accuracy = (correct / len(pred)) * 100 
        test_acc.append(accuracy)
        num_batches = num_batches + 1
        test_labels += y.float().tolist()
        pred_labels += pred.float().tolist()


    return np.mean(test_loss), np.mean(test_acc), test_labels, pred_labels

if __name__ == "__main__":
    test_results = {"Model": [], "Accuracy": []}
    hidden_dims = [32, 64, 128]
    hidden_layers = [1, 2, 3]

    c1 = ['down', 'no_huddle', 'goal_to_go', 'defteam_score', 'half_seconds_remaining', 'quarter_seconds_remaining', 'posteam_timeouts_remaining', 'score_differential', 'ydstogo', 'posteam_score', 'game_seconds_remaining', 'total_away_score']
    c2 = ['down', 'no_huddle', 'goal_to_go', 'defteam_score', 'half_seconds_remaining', 'quarter_seconds_remaining', 'posteam_timeouts_remaining', 'score_differential', 'ydstogo', 'posteam_score', 'game_seconds_remaining', 'total_away_score', 'away_timeouts_remaining', 'yardline_100', 'home_timeouts_remaining', 'drive']
    cols = {'0.001': c1, '0.05': c2}
    p_values = ['0.001', '0.05']

    for i, hidden_dim in enumerate(hidden_dims):
        for j, n_layers in enumerate(hidden_layers):
            for k, p in enumerate(p_values):
                # unique id number for this run
                # used to identify saved model in chekpoints/
                session = str(uuid4())[:5]

                c = cols[p]
                train_x, test_x, train_y, test_y = get_data(c)

                train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.2)

                train_data = PassRunDataset(train_x, train_y)
                train_dataloader = torch.utils.data.DataLoader(
                    train_data, batch_size=10, shuffle=True
                )

                valid_data = PassRunDataset(valid_x, valid_y)
                valid_dataloader = torch.utils.data.DataLoader(
                    valid_data, batch_size=10, shuffle=True
                )

                print(train_data.n_features())
                model = Model(input_size=train_data.n_features(), output_size=1, hidden_dim=hidden_dim, n_layers=n_layers)

                # load in pretrained model if load is true
                load = False
                if load:
                    load_path = 'checkpoints/ad3bc'
                    model.load_state_dict(torch.load(load_path))
                
                model.to(model.device)

                # set training params
                n_epochs = 100
                lr=1e-5

                lf = nn.BCELoss()
                opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

                train_losses = []
                train_accs = []

                valid_losses = []
                valid_accs = []

                train_model = not load # don't train model if one is loaded in
                if train_model:
                    for i in range(n_epochs):
                        l, a, vl, va = train(train_dataloader, model, lf, opt, valid_dataloader)

                        print()
                        print(f"EPOCH: {i}/{n_epochs} train loss: {l:>6f} train accuracy: {a:>6f} valid loss: {vl:>6f} valid accuracy: {va:>6f}")
                        print()

                        train_losses.append(l)
                        train_accs.append(a)
                        valid_losses.append(vl)
                        valid_accs.append(va)

                    plt.plot(train_losses, label="Train Loss")
                    plt.plot(valid_losses, label="Validation Loss")
                    plt.legend()
                    plt.title(f"Loss Plots For Model: {session}")
                    plt.savefig(f"figures/loss_{session}_{n_layers}_{hidden_dim}_{p}.png")

                    plt.clf()

                    plt.plot(train_accs, label="Train Accuracy")
                    plt.plot(valid_accs, label="Validation Accuracy")
                    plt.legend()
                    plt.title(f"Accuracy Plots For Model: {session}")
                    plt.savefig(f"figures/acc_{session}_{n_layers}_{hidden_dim}_{p}.png")

                    plt.clf()


                test_data = PassRunDataset(test_x, test_y)
                test_dataloader = torch.utils.data.DataLoader(
                    test_data, batch_size=10, shuffle=True
                )

                acc, _, y, pred, = test(test_dataloader, model, lf)

                test_results['Model'].append(session)
                test_results['Accuracy'].append(acc)

                conf = confusion_matrix(y, pred)
                p, r, f, s = precision_recall_fscore_support(y, pred, pos_label=1, average='binary')
                

                print("Precision: {:.3f} Recall: {:.3f} F1-Score: {:.3f}".format(p, r, f))
                print(conf)

                save = not load # save new model if one was not loaded in
                if save:
                    save_path = f'checkpoints/{session}'
                    torch.save(model.state_dict(), save_path)

    pd.DataFrame(test_results).to_csv('test_results.csv', index=False)
