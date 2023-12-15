import torch
from torch import nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import datetime
from uuid import uuid4

class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(Model, self).__init__()

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.linear1 = nn.Linear(input_size, 64)
        self.relu1 = nn.LeakyReLU()
        self.linear2 = nn.Linear(64, 64)
        self.relu2 = nn.LeakyReLU()
        self.linear_out = nn.Linear(64, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear_out(x)
        x = self.softmax(x)
        return x
        
        
    
    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device)
        return hidden


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
    

def get_data():
    features = pd.read_csv("./data/scaled_features.csv")
    labels = pd.read_csv("./data/labels.csv")

    return train_test_split(features, labels)

def train(dataloader, model, lf, opt):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.train()
    train_loss = []
    train_acc = []

    correct = 0
    total = 0

    now = datetime.datetime.now()
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)

        # make some predictions and get the error
        pred = model(X)
        loss = lf(torch.log(pred), y)

        # backpropogation
        opt.zero_grad()
        loss.backward()
        opt.step()

        # Get predictions
        _, predicted = torch.max(pred.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            iters = 10 * len(X)
            then = datetime.datetime.now()
            iters /= (then - now).total_seconds()
            print(f"loss: {loss:>6f} [{current:>5d}/{23000}] ({iters:.1f} its/sec)")
            now = then
            train_loss.append(loss)
            accuracy = 100 * correct / total
            train_acc.append(accuracy)

    return train_loss, train_acc

def test(dataloader, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_batches = 0
    model.eval()
    correct = 0
    total = 0
    test_acc = 0
    # get test labels and predictions
    test_labels = []
    pred_labels = []
    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        # Get predictions
        _, predicted = torch.max(pred.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
        accuracy = 100 * correct / total
        test_acc += accuracy
        num_batches = num_batches + 1
        test_labels += y.float().tolist()
        pred_labels += predicted.float().tolist()

    test_acc /= num_batches
    print(f"Test Accuracy: {test_acc:>0.1f}%\n") 
    return test_acc, test_labels, pred_labels

if __name__ == "__main__":

    # unique id number for this run
    # used to identify saved model in chekpoints/
    session = str(uuid4())[:5]

    train_x, test_x, train_y, test_y = get_data()

    train_data = PassRunDataset(train_x, train_y)
    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=100, shuffle=True
    )

    model = Model(input_size=train_data.n_features(), output_size=2, hidden_dim=64, n_layers=4)

    # load in pretrained model if load is true
    load = True
    if load:
        load_path = 'checkpoints/ad3bc'
        model.load_state_dict(torch.load(load_path))
    
    model.to(model.device)

    # set training params
    n_epochs = 100
    lr=1e-4

    lf = nn.NLLLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    train_losses = []
    train_accs = []
    
    train_model = not load # don't train model if one is loaded in
    if train_model:
        for i in range(n_epochs):
            l, a = train(train_dataloader, model, lf, opt)

            train_losses.append(l)
            train_accs.append(a)


    test_data = PassRunDataset(test_x, test_y)
    test_dataloader = torch.utils.data.DataLoader(
        test_data, batch_size=100, shuffle=True
    )

    acc, y, pred = test(test_dataloader, model)
    conf = confusion_matrix(y, pred)
    p, r, f, s = precision_recall_fscore_support(y, pred, pos_label=1, average='binary')
    

    print("Precision: {:.3f} Recall: {:.3f} F1-Score: {:.3f}".format(p, r, f))
    print(conf)

    save = not load # save new model if one was not loaded in
    if save:
        save_path = f'checkpoints/{session}'
        torch.save(model.state_dict(), save_path)

