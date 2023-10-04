from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
# python 的用法 tensor数据类型
# 通过transforms.ToTensor去看两个问题
# 1、transforms谈如何使用python
# 2、为什么我们想要Tensor数据类型

img_path="data/train/ants_image/6240338_93729615ec.jpg"
img=Image.open(img_path)

writer=SummaryWriter("logs")

tensor_trans=transforms.ToTensor()
tensor_img=tensor_trans(img)

writer.add_image("Tensor_img",tensor_img)

writer.close()
print(tensor_img)