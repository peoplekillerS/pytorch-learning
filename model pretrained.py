import torchvision

# train_data=torchvision.datasets.ImageNet("../data_image_net",split='train',download=True,
#                                          transform=torchvision.transforms.ToTensor())
from torch import nn

vgg16_true=torchvision.models.vgg16()
print(vgg16_true)
vgg16_false=torchvision.models.vgg16()
train_data=torchvision.datasets.CIFAR10("../daTa",train=True,download=True,
                                        transform=torchvision.transforms.ToTensor())

# 如何利用现有的网络来改进 比如再加一步 in 1000 out 10  因为这个数据集就是10类

vgg16_true.add_module('add_linear',nn.Linear(1000,10))
# 或者在里面添加
# vgg16_true.classifier.add_module('add_linear',nn.Linear(1000,10))

print(vgg16_true)

vgg16_false.classifier[6]=nn.Linear(4096,10)



