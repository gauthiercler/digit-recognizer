import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

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

def train(net, loader, epochs, criterion, optimizer):

    correct = 0
    loss = 0.0

    for epoch in range(5):
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

            if idx % 1000 == 999:
                print(epoch + 1, idx + 1, loss / 100, correct / 6000 * 100)
                correct = 0
                loss = 0.0

def main():
    net = ConvNet()

    transform = transforms.Compose(
    [
        transforms.ToTensor()
    ])

    trainset = torchvision.datasets.MNIST('.', download=True,
                                       train=True,
                                       transform=transform)

    loader = torch.utils.data.DataLoader(trainset,
                                         batch_size=6)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001)

    train(net, loader, 5, criterion, optimizer)

if __name__ == '__main__':
    main()
