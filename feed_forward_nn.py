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
          nn.ReLU(),
          nn.Dropout(p=0.1)
        )

        for _ in range(n_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(hidden_dim, output_size))
        # self.layers.append(nn.Sigmoid())

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
    model.train()

    train_loss = []
    train_acc = []

    valid_loss = []
    valid_acc = []

    now = datetime.datetime.now()
    for batch, (X, y) in enumerate(dataloader):

        X = X.to(model.device)
        y = y.to(model.device)

        # make some predictions and get the error
        logits = model(X).squeeze()
        pred = torch.round(torch.sigmoid(logits))
        loss = lf(logits, y.float())

        # backpropogation
        opt.zero_grad()
        loss.backward()
        opt.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            correct = torch.eq(y, pred).sum().item() 
            accuracy = (correct / len(pred)) * 100 
            v_loss, v_acc, _, _ = test(val_dataloader, model, lf)
            iters = 10 * len(X)
            then = datetime.datetime.now()
            iters /= (then - now).total_seconds()
            print(f"train loss: {loss:>6f} train accuracy: {accuracy:>6f} valid loss: {v_loss:>6f} valid accuracy: {v_acc:>6f} [{current:>5d}/{17920}] ({iters:.1f} its/sec)")
            now = then
            train_loss.append(loss)
            train_acc.append(accuracy)
            valid_loss.append(v_loss)
            valid_acc.append(v_acc)

    return np.mean(train_loss), np.mean(train_acc), np.mean(valid_loss), np.mean(valid_acc)

def test(dataloader, model, lf):
    num_batches = 0
    correct = 0
    test_loss = []
    # get test labels and predictions
    test_labels = []
    pred_labels = []
    for X, y in dataloader:
        X = X.to(model.device)
        y = y.to(model.device)
        logits = model(X).squeeze()
        pred = torch.round(torch.sigmoid(logits))
        loss = lf(logits, y.float()).item()
        test_loss.append(loss)
        correct += torch.eq(y, pred).sum().item() 
        num_batches = num_batches + 1
        test_labels += y.float().tolist()
        pred_labels += pred.float().tolist()

    test_acc = (correct / len(pred_labels)) * 100 

    return np.mean(test_loss), test_acc, test_labels, pred_labels

if __name__ == "__main__":
    test_results = {"Model": [], 
                    "Accuracy": [], 
                    "HiddenDims":[], 
                    "HiddenLayers":[], 
                    "PValue": [], 
                    "LearningRate": [], 
                    "Epochs": []
                    }
    
    hidden_dims = [64, 128, 256]
    hidden_layers = [1, 2, 3]
    learning_rates = [1e-2, 1e-3, 1e-4]

    c1 = ['down', 'no_huddle', 'goal_to_go', 'defteam_score', 'half_seconds_remaining', 'quarter_seconds_remaining', 'posteam_timeouts_remaining', 'score_differential', 'ydstogo', 'posteam_score', 'game_seconds_remaining', 'total_away_score']
    c2 = ['down', 'no_huddle', 'goal_to_go', 'defteam_score', 'half_seconds_remaining', 'quarter_seconds_remaining', 'posteam_timeouts_remaining', 'score_differential', 'ydstogo', 'posteam_score', 'game_seconds_remaining', 'total_away_score', 'away_timeouts_remaining', 'yardline_100', 'home_timeouts_remaining', 'drive']
    cols = {'0.001': c1, '0.05': c2, '1': None}
    p_values = ['0.001', '0.05', '1']
    for hidden_dim in hidden_dims:
        for n_layers in hidden_layers:
            for p in p_values:
                for lr in learning_rates:
                    # unique id number for this run
                    # used to identify saved model in chekpoints/
                    session = str(uuid4())[:5]

                    c = cols[p]
                    train_x, test_x, train_y, test_y = get_data(c)

                    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.2)

                    train_data = PassRunDataset(train_x, train_y)
                    train_dataloader = torch.utils.data.DataLoader(
                        train_data, batch_size=256, shuffle=True
                    )

                    valid_data = PassRunDataset(valid_x, valid_y)
                    valid_dataloader = torch.utils.data.DataLoader(
                        valid_data, batch_size=256, shuffle=True
                    )

                    test_data = PassRunDataset(test_x, test_y)
                    test_dataloader = torch.utils.data.DataLoader(
                        test_data, batch_size=256, shuffle=True
                    )

                    model = Model(input_size=train_data.n_features(), output_size=1, hidden_dim=hidden_dim, n_layers=n_layers)
                    model.to(model.device)

                    # set training params
                    n_epochs = 100

                    lf = nn.BCEWithLogitsLoss()
                    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

                    train_losses = []
                    train_accs = []

                    valid_losses = []
                    valid_accs = []

                    early_acc = 0
                    early_stop = False
                    epochs_trained = 0
                    for i in range(n_epochs):

                        epochs_trained += 1

                        l, a, vl, va = train(train_dataloader, model, lf, opt, valid_dataloader)

                        print()
                        print(f"EPOCH: {i}/{n_epochs} train loss: {l:>6f} train accuracy: {a:>6f} valid loss: {vl:>6f} valid accuracy: {va:>6f}")
                        print()

                        train_losses.append(l)
                        train_accs.append(a)
                        valid_losses.append(vl)
                        valid_accs.append(va)

                        # initialize early stopping
                        if i == 0:
                            early_acc = va

                        # stop if accuracy has no improved in 5 epochs
                        if i % 5 == 0 and i != 0:
                            early_stop = early_acc > va
                            early_acc = va

                        if early_stop:
                            break


                    plt.plot(train_losses, label="Train Loss")
                    plt.plot(valid_losses, label="Validation Loss")
                    plt.legend()
                    plt.title(f"Loss Plots For Model: {session}")
                    plt.savefig(f"figures/loss_{session}_{n_layers}_{hidden_dim}_{p}_{lr}.png")

                    plt.clf()

                    plt.plot(train_accs, label="Train Accuracy")
                    plt.plot(valid_accs, label="Validation Accuracy")
                    plt.legend()
                    plt.title(f"Accuracy Plots For Model: {session}")
                    plt.savefig(f"figures/acc_{session}_{n_layers}_{hidden_dim}_{p}_{lr}.png")

                    plt.clf()

                    model.eval()
                    _, acc, y, pred, = test(test_dataloader, model, lf)

                    test_results['Model'].append(session)
                    test_results['Accuracy'].append(acc)
                    test_results['HiddenDims'].append(hidden_dim)
                    test_results['HiddenLayers'].append(n_layers)
                    test_results['PValue'].append(p)
                    test_results['LearningRate'].append(lr)
                    test_results['Epochs'].append(epochs_trained)

                    conf = confusion_matrix(y, pred)
                    pr, r, f, s = precision_recall_fscore_support(y, pred, pos_label=1, average='binary')

                    print("Precision: {:.3f} Recall: {:.3f} F1-Score: {:.3f}".format(pr, r, f))
                    print(conf)

                    save_path = f'checkpoints/{session}'
                    torch.save(model.state_dict(), save_path)

    pd.DataFrame(test_results).to_csv('test_results.csv', index=False)
