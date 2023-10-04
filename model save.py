import torch
import torchvision
from torch import nn

vgg16=torchvision.models.vgg16()
# 有两种方式保存
torch.save(vgg16,"vgg16_method1.pth")
# 不仅保存了网络模型还把参数保存了

# 保存方式2 保存字典 保存网络模型的参数 获得模型的状态
torch.save(vgg16.state_dict(),"vgg16_method2.pth")
# 这样子保存的内存较小

# 陷阱
class Qianshi(nn.Module):
    def __init__(self):
        super(Qianshi, self).__init__()
        self.conv1=nn.Conv2d(3,64,kernel_size=3)

    def forward(self,x):
        x=self.conv1(x)
        return x
qianshi=Qianshi()
torch.save(qianshi,"qianshi_method.pth")
