import torch
# 方式以 保存方式 加载模型
import torchvision

# from model save import *

model=torch.load("vgg16_method1.pth")
print(model)

model=torch.load("vgg16_method2.pth")
print(model)


# 方式2 加载模型
vgg16=torchvision.models.vgg16()
vgg16.load_state_dict("vgg_method2.pth")
# model=torch.load("vgg_method2.pth")
print(vgg16)

# 陷阱1 还要把这个类给加上来
class Qianshi(nn.Module):
    def __init__(self):
        super(Qianshi, self).__init__()
        self.conv1=nn.Conv2d(3,64,kernel_size=3)

    def forward(self,x):
        x=self.conv1(x)
        return x
#     qinashi=Qianshi()
model=torch.load("qianshi_method.pth")
print(model)

