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
# 优化器
optim=torch.optim.SGD(qianshi.parameters(),lr=0.01)
for epoch in range(20):
    running_loss=0.0
    for data in dataloader:
        imgs,targets=data
        outputs=qianshi(imgs)
        result_loss=loss(outputs,targets)
        optim.zero_grad()
        result_loss.backward()
        # 获得节点梯度
        optim.step()
    #     对每个参数调优
        running_loss=running_loss+result_loss
    print(running_loss)

# grad就是梯度
# 根据梯度对其优化从而降低梯度