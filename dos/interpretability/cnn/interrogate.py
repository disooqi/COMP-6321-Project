import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from architecture import CNN

#
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
testset = CIFAR10(root='../../../data', train=False, download=False, transform=transform)
#
testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)

trainset = CIFAR10(root='../../../data', train=False, download=False, transform=transform)
trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0)

PATH = './cifar_net.pth'
cnn = CNN()
cnn.load_state_dict(torch.load(PATH))

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
def imshow(filename, img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.imsave(filename, np.transpose(npimg, (1, 2, 0)))
    plt.show()


cnn = CNN()


x = torch.zeros((1, 3, 32, 32), requires_grad=True)


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

for i in range(6000):
    # image, label = data
    # if label != 0:
    #     continue
    y = cnn(x)

    # _, predicted = torch.max(y.data, 1)
    # print(y.size())
    y[:1, 9:].backward()
    x.data += 0.15 * x.grad.data
    # print(x1.grad.data)
    x.grad.data.zero_()


imshow('truck.png', torchvision.utils.make_grid(x.detach()))

