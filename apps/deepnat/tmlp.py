import torch.nn as nn
import torch.cuda
from sklearn.model_selection import train_test_split
import numpy as np
import torch.optim as optim

DTYPE = np.float32


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
        return np.array(list(product([-1.0, 1.0], repeat=features)), dtype=DTYPE)

    def data_y():
        m = 2 ** features
        rs = []
        for xs in data_x():
            r = 0
            for x in xs:
                x = int((x + 1) / 2)
                r = r ^ x
            rs.append(r)
        ys = to_np(rs).reshape(m, 1)
        return ys

    return split(data_x(), data_y(), seed), layer_sizes


def split(X, y, seed, test_size=0.1):
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=test_size, random_state=seed)
    return X_train, X_test, y_train, y_test


class LinearNet(nn.Module):
    def __init__(self, input_size, sizes, output_size):
        super(LinearNet, self).__init__()
        self.num_layers = len(sizes)

        self.linears = nn.ModuleList([nn.Linear(input_size, sizes[0])])
        print(input_size, sizes[0])
        for i in range(1, self.num_layers):
            s1 = sizes[i - 1]
            s2 = sizes[i]
            print('SIZES:', s1, s2)
            self.linears.append(nn.Linear(s1, s2))

        self.linears.append(nn.Linear(sizes[-1], output_size))
        print(sizes[-1], output_size)

    def forward(self, x):
        f = nn.ReLU()
        for n in self.linears:
            x = n(x)
            x = f(x)
        # g=nn.Sigmoid()
        return x  # g(x)


def accuracy(Y, Y_hat):
    return (Y == Y_hat).sum() / len(Y)


def train(X_train, y_train, sizes, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = LinearNet(X_train.shape[1], sizes, 1)
    lossfun = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), weight_decay=0.001)

    inputs = X_train

    outputs = inputs

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = lossfun(outputs, y_train)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0: print('LOSS AT ', epoch, '\t-->', round(loss.item(), 4))

    return net, lossfun


def test(net, lossfun, X, y):
    with torch.inference_mode():
        y_hat = net(X)
        loss = np.sqrt(lossfun(y_hat, y).detach().numpy())
        preds = y_hat > 0.5
        y = y > 0.5
        acc = accuracy(y, preds)
        return loss.tolist(), acc.tolist()


def run():
    epochs = 600
    features = 12
    seed = 0
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    dataset, sizes = load_dataset(features, seed)
    X_train, X_test, y_train, y_test = [torch.from_numpy(d) for d in dataset]
    net, lossfun = train(X_train, y_train, sizes, epochs)
    train_loss, train_acc = test(net, lossfun, X_train, y_train)
    test_loss, test_acc = test(net, lossfun, X_test, y_test)

    print('\nRESULTS:\n')

    print('Train loss:', train_loss)
    print('Test loss:', test_loss)

    print()

    print('Train accuracy:', train_acc)
    print('Train accuracy:', test_acc)


def t1():
    n = LinearNet(4, [5, 10, 15, 6], 1)
    print(type(n))

    for d in load_dataset(4):
        print(d)
        print()


if __name__ == "__main__":
    run()
