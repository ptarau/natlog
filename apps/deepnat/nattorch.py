import sys
import random
import torch.nn as nn
import torch.cuda
from torch import tensor
from sklearn.model_selection import train_test_split
import numpy as np
from numpy import array, transpose, shape
import torch.optim as optim
from natlog import Natlog, natprogs

sys.setrecursionlimit(1 << 14)

DTYPE = np.float32
TDTYPE = torch.float

shared = dict()


def share(f):
    shared[f.__name__] = f
    return f


def share_syms():
    for n, f in globals().items():
        if n not in {'add'}:
            shared[n] = f
    return shared


def load_dataset(features, seed):
    """
     a ^ b ... ^ ... synthetic boolean
     truth table dataset - known to be hard to learn
    """
    from itertools import product

    n = 2 * features
    layer_sizes = [n + n, n, n + n, n]

    def to_np(a):
        return np.array(a, dtype=DTYPE)

    def data_x():
        x=list(product([-1.0, 1.0], repeat=features))
        return x

    def data_y():
        m = 2 ** features
        rs = []
        for xs in data_x():
            r = 0
            for x in xs:
                x = int((x + 1) / 2)
                r = r ^ x
            rs.append(r)
        rs=[rs]
        return rs

    return split(data_x(), data_y(), seed), layer_sizes


def split(Xss, ys, seed, test_size=0.1):
    X=np.array(Xss)
    y=np.array(ys)
    y=y.T
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=test_size, random_state=seed)
    res = tuple(tensor(a, dtype=torch.float) for a in [X_train, X_test, y_train, y_test])
    # print('SPLIT:', res)
    return res


class LinearNet(nn.Module):
    def __init__(self, input_size, sizes, output_size):
        super(LinearNet, self).__init__()

        self.num_layers = len(sizes)

        self.linears = nn.ModuleList([nn.Linear(input_size, sizes[0])])
        print(input_size, sizes[0])
        for i in range(1, self.num_layers):
            s1 = sizes[i - 1]
            s2 = sizes[i]
            print('HIDDEN SIZES:', s1, s2)
            self.linears.append(nn.Linear(s1, s2))

        self.linears.append(nn.Linear(sizes[-1], output_size))
        print(sizes[-1], output_size)

    def forward(self, x):
        f = nn.ReLU()
        # f = nn.Tanh()
        for n in self.linears:
            x = n(x)
            x = f(x)
        # g = nn.Sigmoid()
        return x  # g(x)


def accuracy(Y, Y_hat):
    return (Y == Y_hat).sum() / len(Y)


def train_model(X_train, y_train, sizes, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = LinearNet(X_train.shape[1], sizes, 1)
    lossfun = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), weight_decay=0.001)

    inputs = X_train

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = lossfun(outputs, y_train)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0: print('LOSS AT ', epoch, '\t-->', round(loss.item(), 8))

    return net, lossfun


def test_model(net, lossfun, X, y):
    print('STATE DICT:', list(net.state_dict().keys()))
    with torch.inference_mode():
        y_hat = net(X)
        loss = np.sqrt(lossfun(y_hat, y).detach().numpy())
        preds = y_hat > 0.5
        y = y > 0.5
        acc = accuracy(y, preds)
        res = loss.tolist(), acc.tolist()
        # print('RES:', res)
        return res


def test_nattorch():
    epochs = 600
    features = 10
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    dataset, sizes = load_dataset(features, seed)
    X_train, X_test, y_train, y_test = [torch.from_numpy(d) for d in dataset]
    net, lossfun = train_model(X_train, y_train, sizes, epochs)
    train_loss, train_acc = test_model(net, lossfun, X_train, y_train)
    test_loss, test_acc = test_model(net, lossfun, X_test, y_test)

    print('\nRESULTS:\n')

    print('Train loss:', train_loss)
    print('Test loss:', test_loss)

    print()

    print('Train accuracy:', train_acc)
    print('Train accuracy:', test_acc)


def lin_test():
    n = LinearNet(4, [5, 10, 15, 6], 1)
    print(type(n))

    for d in load_dataset(4,42):
        print(d)
        print()


# Natlog activation
def run_natlog():
    share_syms()
    n = Natlog(file_name="nattorch.nat",
               with_lib=natprogs() + "lib.nat", callables=shared)
    # n.query("eq Started 'Natlog'.")
    # n.query("`eye 4 M, `matmul M M X, #print X, fail?")
    # n.query('go?'),
    # n.query('alt?'),
    n.repl()


if __name__ == "__main__":
    # test_nattorch()
    run_natlog()
