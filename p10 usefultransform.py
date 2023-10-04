from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer=SummaryWriter("logs")
img=Image.open("images/pytorch_2.jpg")
print(img)

# ToTensor使用
trans_totensor=transforms.ToTensor()
img_tensor=trans_totensor(img)
writer.add_image("ToTensor",img_tensor)



# Normalize
print(img_tensor[0][0][0])
trans_norm=transforms.Normalize([1,2,3],[1,2,3])
img_norm=trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("normalize",img_norm,2)

# Resize
print(img.size)
trans_resize=transforms.Resize((512,512))
# img PIL->resize->img_resize PIL(还是图片)
img_resize=trans_resize(img)
print(img_resize)
print(type(img_resize))

# img_resize PIL->totensor->img_resize tensor
img_resize=trans_totensor(img_resize)
writer.add_image("Resize",img_resize)
print(img_resize)

# Compose -resize -2
trans_resize_2=transforms.Resize(512)
# PIL -> PIL ->  tensor
trans_compose=transforms.Compose([trans_resize_2,trans_totensor])
img_resize_2=trans_compose(img)

writer.add_image("Resize",img_resize,1)

# Randomcrop
trans_random=transforms.RandomCrop(256,512)
trans_compose_2=transforms.Compose([trans_random,trans_totensor])
for i in range(10):
    img_crop=trans_compose_2(img)
    writer.add_image("RandomCrop",img_crop,i)



writer.close()



