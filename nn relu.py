#非线性变化为了引入非线性特征

import torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input=torch.tensor([[1,-0.5],
                    [-1,3]])
input=torch.reshape(input,(-1,1,2,2))
# -1表示batch_size自己算
print(input.shape)

dataset=torchvision.datasets.CIFAR10("../daTa",train=False,download=True,
                                     transform=torchvision.transforms.ToTensor())
dataloader=DataLoader(dataset,batch_size=64)


class Qianshi(nn.Module):
    def __init__(self):
        super(Qianshi, self).__init__()
        self.relu1=ReLU()
        self.sigmoid1=Sigmoid()
#         inplace改变与否是直接导致input是否改变
    def forward(self,input):
        output=self.sigmoid1(input)
        return output

qianshi=Qianshi()
# output=qianshi(input)
# print(output)

writer=SummaryWriter("./logs_relu")
step=0
for data in dataloader:
    imgs,targets=data
    writer.add_images("input",imgs,global_step=step)
# global_step就是x坐标就是横坐标
    output=qianshi(imgs)
    writer.add_images("output",output,step)
    step=step+1
writer.close()


