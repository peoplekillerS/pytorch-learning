import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset=torchvision.datasets.CIFAR10("../daTa",train=False,transform=torchvision.transforms.ToTensor(),
                                     download=True)
dataloader=DataLoader(dataset,batch_size=64)

class Qianshi(nn.Module):
    def __init__(self):
        super(Qianshi, self).__init__()
        self.conv1=Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)


    def forward(self,x):
        x=self.conv1(x)
        return x
qianshi=Qianshi()
print(qianshi)

writer=SummaryWriter("../Logs")
step=0
for data in dataloader:
    imgs,targets=data
    output=Qianshi(imgs)
    # print(output.shape)
    print(imgs.shape)
    print(output.shape)

    writer.add_image("input",imgs,step)

    output=torch.reshape(output,(-1,3,30,30))
    # -1表示不知道的话就填-1
    writer.add_image("input",output,step)
    step=step+1







