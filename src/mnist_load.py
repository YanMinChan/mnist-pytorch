import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, ))])

batch_size = 1000

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                      download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size,
                         shuffle=True, num_workers=0)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                     download=True, transform=transform)
testloader = DataLoader(testset, batch_size=batch_size,
                        shuffle=False, num_workers=0)

# Print the MNIST image


def imshow(img):
    img = img / 2 + 0.5  # Undo normalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# dataiter = iter(trainloader)
# images, labels = next(dataiter)

# imshow(torchvision.utils.make_grid(images))
# print(' '.join(f'{labels[j]}' for j in range(batch_size)))

# [[Define the CNN]]


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(3, 3)
        self.fc1 = nn.Linear(6 * 8 * 8, 80)
        self.fc2 = nn.Linear(80, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = torch.flatten(x, 1)  # flatten all dimension except batch
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        return x


net = Net()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.100, momentum=0.9)

# [[Network training]]
# Select device (gpu if cuda is available)
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#
# net.to(device)

# Train the network
num_epochs = 2

start_time = time.time()
for epoch in range(num_epochs):

    correct = 0
    loss = torch.tensor(0.0)
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / len(trainset)
    print(f'[{epoch + 1}/{num_epochs}] loss: {loss.item()
          :.3f}, accuracy: {accuracy:.3f}')


end_time = time.time()
elapsed_time = end_time - start_time
print(f'Finished Training.\nTime taken: {elapsed_time:.4f} seconds')

PATH = './mnist_net.pth'
torch.save(net.state_dict(), PATH)

# [[Network performance]]
# Load the nn
net = Net()
net.load_state_dict(torch.load(PATH, weights_only=True))

# # Demo prediction
# dataiter = iter(testloader)
# images, labels = next(dataiter)
#
# # print images
# imshow(torchvision.utils.make_grid(images))
# print(' '.join(f'{labels[j]}' for j in range(4)))
#
# # Predict output (with test data)
# outputs = net(images)
# _, predicted = torch.max(outputs, 1)
# print('Predicted: ', ' '.join(f'{predicted[j]}' for j in range(4)))

# Overall performance
correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accucary of the network: {100 * correct // total}%')

# Performance for each class
correct_pred = {i: 0 for i in range(10)}
total_pred = {i: 0 for i in range(10)}
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)

        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[label.item()] += 1
            total_pred[label.item()] += 1

for digit, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[digit]
    print(f'Accuracy for class: {digit} is {accuracy:.1f}%')
