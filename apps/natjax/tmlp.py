import torch.nn as nn
import torch.cuda
from sklearn.model_selection import train_test_split
import numpy as np
import torch.optim as optim

DTYPE = np.float32
SEED = 42
EPOCHS = 600
FEATURES = 12

np.random.seed(SEED)
torch.random.manual_seed(SEED)


def load_dataset():
    """
     a ^ b ... ^ ... synthetic boolean
     truth table dataset - known to be hard to learn
    """
    from itertools import product

    features = FEATURES
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

    return split(data_x(), data_y()), layer_sizes


def split(X, y, test_size=0.1):
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=test_size, random_state=SEED)
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
            print(s1, s2)
            self.linears.append(nn.Linear(s1, s2))

        self.linears.append(nn.Linear(sizes[-1], output_size))
        print(sizes[-1], output_size)

    def forward(self, x):
        for n in self.linears:
            x = n(x)
            f = nn.ReLU()
            x = f(x)
        # g=nn.Sigmoid()
        return x  # g(x)


def t1():
    n = LinearNet(4, [5, 10, 15, 6], 1)
    print(type(n))

    for d in load_dataset():
        print(d)
        print()


def train():
    dataset, sizes = load_dataset()
    X_train, X_test, y_train, y_test = [torch.from_numpy(d) for d in dataset]

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = LinearNet(X_train.shape[1], sizes, 1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), weight_decay=0.0001)

    inputs = X_train

    outputs=inputs

    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        outputs = net(inputs)

        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0: print('LOSS AT ', epoch, '\t-->', round(loss.item(), 4))

    inputs = X_train

    train_outputs = net(inputs)
    train_res = np.sqrt(criterion(outputs, y_train).detach().numpy())

    inputs = X_test
    test_outputs = net(inputs)

    test_res = np.sqrt(criterion(test_outputs, y_test).detach().numpy())

    print("Root mean squared error:")
    print("Train", train_res)
    print("Test ", test_res)

    train_preds = (train_outputs > 0.5)
    test_preds = (test_outputs > 0.5)

    y_train = y_train > 0.5
    y_test = y_test > 0.5

    def acc(Y, Y_hat):
        # print(Y,Y_hat)
        return (Y == Y_hat).sum() / len(Y)

    print('ACCURACY TRAIN PREDS:', acc(y_train, train_preds))
    print('ACCURACY TEST PREDS: ', acc(y_test, test_preds))


if __name__ == "__main__":
    train()
