import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d, Conv2d, Flatten, Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class Qianshi(nn.Module):
    def __init__(self):
        super(Qianshi, self).__init__()
        self.conv1=Conv2d(3,32,5,padding=2)
#         这是第一个卷积部分
        self.maxpool1=MaxPool2d(2)
#         2是kernel size
        self.conv2=Conv2d(32,32,5,padding=2)
#         第二个卷积部分
        self.maxpool2=MaxPool2d(2)
        self.conv3=Conv2d(32,64,5,padding=2)
#         第三部分的卷积
        self.maxpool3=MaxPool2d(2)
        self.flatten=Flatten()
# 上面就是卷积池化卷积池化

#         先变成一个线性层
        self.linear1=Linear(1024,64)
        self.linear2=Linear(64,10)

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
        x=self.conv1(x)
        x=self.maxpool1(x)
        x=self.conv2(x)
        x=self.maxpool2(x)
        x=self.conv3(x)
        x=self.maxpool3(x)
        x=self.flatten(x)
        x=self.linear1(x)
        x=self.linear2(x)

        # x=self.model1(x)

        return x

qianshi=Qianshi()
print(qianshi)

# 去检验
input=torch.ones((64,3,32,32))
ouput=qianshi(input)
print(ouput.shape)

writer=SummaryWriter("logs_seq")
writer.add_graph(qianshi,input)
writer.close()

