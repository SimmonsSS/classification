from Lenet5 import LeNet
from Alexnet import AlexNet
from VGG import VGG16
from resnet import ResNet18
from resnet import ResBlock
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCH=10
pre_epoch=0
BATCH_SIZE=128
LR=0.01

"""choose model"""
# net=LeNet().to(device)
# net=AlexNet().to(device)
# net=VGG16().to(device)
net=ResNet18().to(device)

"""Lenet5训练测试数据集:MNIST"""
# transform_train=transforms.Compose([
#     transforms.ToTensor()
# ])
# train_set=torchvision.datasets.MNIST(root="./data",train=True,download=True,transform=transform_train)
# transform_test=transforms.Compose([
#     transforms.ToTensor()
# ])
# test_set=torchvision.datasets.MNIST(root="./data",train=False,download=True,transform=transform_test)

"""Alexnet,vgg16,resnet训练测试数据集:CIFAR10"""
transform_train=transforms.Compose([
    transforms.RandomCrop(32,padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))
])
train_set=torchvision.datasets.CIFAR10(root="./data",train=True,download=True,transform=transform_train)
transform_test=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))
])
test_set=torchvision.datasets.CIFAR10(root="./data",train=False,download=True,transform=transform_test)
#
trainloader=torch.utils.data.DataLoader(train_set,batch_size=BATCH_SIZE,shuffle=True,num_workers=2)
testloader=torch.utils.data.DataLoader(test_set,batch_size=BATCH_SIZE,shuffle=False,num_workers=2)

classes=('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(net.parameters(),lr=LR,momentum=0.9,weight_decay=5e-4)


for epoch in range(pre_epoch,EPOCH):
    print("\nEpoch: %d"%(epoch+1))
    net.train()
    sum_loss=0.0
    correct=0.0
    total=0.0
    for i,data in enumerate(trainloader,0):
        length=len(trainloader)
        inputs,labels=data
        inputs,labels=inputs.to(device),labels.to(device)
        optimizer.zero_grad()

        outputs=net(inputs)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        sum_loss+=loss.item()
        _,predicted=torch.max(outputs.data,1)
        total+=labels.size(0)
        correct+=predicted.eq(labels.data).cpu().sum()
        print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
              % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))

print('Train has finished, total epoch is %d' % EPOCH)


