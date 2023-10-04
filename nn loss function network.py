import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d, Conv2d, Flatten, Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset=torchvision.datasets.CIFAR10("../daTa",train=False,transform=torchvision.transforms.ToTensor(),download=True)

dataloader=DataLoader(dataset,batch_size=1)


class Qianshi(nn.Module):
    def __init__(self):
        super(Qianshi, self).__init__()
        self.model1=nn.Sequential(
            Conv2d(3,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32,64,5,padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024,64),
            Linear(64,10)
        )


    def forward(self,x):
        x=self.model1(x)
        return x

loss=nn.CrossEntropyLoss()

qianshi=Qianshi()
for data in dataloader:
    imgs,targets=data
    outputs=qianshi(imgs)
    result_loss=loss(outputs,targets)
    result_loss.backward()
    print("ok")

# grad就是梯度
# 根据梯度对其优化从而降低梯度