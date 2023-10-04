import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset=torchvision.datasets.CIFAR10("../daTa",train=False,transform=torchvision.transforms.ToTensor(),download=True)

dataloader=DataLoader(dataset,batch_size=64)

class Qianshi(nn.Module):
    def __init__(self):
        super(Qianshi, self).__init__()
        self.linear=Linear(196608,10)
#         in_feature=196608 out_feature=10

    def forward(self,input):
        output=self.linear(input)
        return output

qianshi=Qianshi()

for data in dataloader:
    imgs,targets=data
    print(imgs.shape)
    # output=torch.reshape(imgs,(1,1,1,-1))
    output=torch.flatten(imgs)
    # 用flatten不用reshape 直接对feature动手
    print(output.shape)
    output=qianshi(output)
    print(output.shape)






