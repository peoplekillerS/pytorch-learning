import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset=torchvision.datasets.CIFAR10("../daTa",train=False,download=True,transform=torchvision.transforms.ToTensor())

dataloader=DataLoader(dataset,batch_size=64)

input=torch.tensor([[1,2,0,3,1],
                   [0,1,2,3,1],
                   [1,2,1,0,0],
                   [5,2,3,1,1],
                   [2,1,0,1,1]],dtype=torch.float32)
# 说明数据类型全为浮点数
input =torch.reshape(input,(-1,1,5,5))
# -1就是表示让计算机自己去算出来
print(input.shape)

class Qianshi(nn.Module):
    def __init__(self):
        super(Qianshi, self).__init__()
        self.maxpool1=MaxPool2d(kernel_size=3,ceil_mode=True)


    def forward(self,input):
        output=self.maxpool1(input)
        return output
# qianshi=Qianshi(input)
# output=Qianshi(input)
# print(output)

qianshi=Qianshi()

writer=SummaryWriter("../logs_maxpool")
step=0
for data in dataloader:
    imgs,targets=data
    writer.add_image("input",imgs,step)
    output=Qianshi(imgs)
    writer.add_image("ouput",output,step)
    step=step+1
writer.close()