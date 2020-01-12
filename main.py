import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import pandas as pd


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        '''
        1 input layer
        5 output layer
        Kernel size 3x3
        '''
        self.conv1 = nn.Conv2d(1, 5, 3)

        # Kernel size 2x2
        self.pool = nn.MaxPool2d(2)

        '''
        5 input layers
        15 output layers
        Kernel size 3x3
        '''
        self.conv2 = nn.Conv2d(5, 15, 3)

        # Flatten into fully connected layer
        self.fc1 = nn.Linear(15 * 5 * 5, 120)

        self.fc2 = nn.Linear(120, 60)

        # 10 output classes
        self.fc3 = nn.Linear(60, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # Reshape, -1 is inferred from other dimensions
        x = x.view(-1, 15 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


PRINT_STEP = 1000


class ToCuda:
    def __call__(self, tensor):
        return tensor.cuda()


def train(net, loader, epochs, criterion, optimizer):

    correct = 0
    loss = 0.0

    for epoch in range(epochs):
        for idx, data in enumerate(loader):

            X, y = data

            optimizer.zero_grad()

            outputs = net(X)

            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, y)

            loss.backward()
            optimizer.step()

            correct += (predicted == y).sum().item()
            loss += loss.item()

            if idx % PRINT_STEP == PRINT_STEP - 1:
                print(f'epoch [{epoch + 1} {idx + 1}], loss {loss / 100:.4f}, accuracy {correct / (loader.batch_size * PRINT_STEP) * 100:.3f}')
                correct = 0
                loss = 0.0


def main():

    transformers = [transforms.ToTensor()]

    if torch.cuda.is_available():
        torch.cuda.set_device("cuda:0")
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        transformers.append(ToCuda())

    transform = transforms.Compose(transformers)

    trainset = torchvision.datasets.MNIST('.', download=True,
                                          train=True,
                                          transform=transform)

    loader = torch.utils.data.DataLoader(trainset,
                                         batch_size=6)

    net = ConvNet()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001)

    train(net, loader, 40, criterion, optimizer)

    testset = pd.read_csv('test.csv')
    X = testset.values.astype('float32')
    X = torch.tensor(X)
    X = X.view(X.shape[0], 1, 28, 28)
    print(X.shape)
    outputs = net(X)

    _, predicted = torch.max(outputs.data, 1)
    pd.DataFrame({
        "ImageId": [x for x in range(1, len(predicted) + 1)],
        "Label": predicted
    }).to_csv('results.csv', index=False)

if __name__ == '__main__':
    main()
