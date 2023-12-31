# 准备数据集
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time

train_data=torchvision.datasets.CIFAR10("../daTa",train=True,download=True,
                                        transform=torchvision.transforms.ToTensor())
test_data=torchvision.datasets.CIFAR10(root="../daTa",train=False,download=True,transform=torchvision.transforms.ToTensor())

# length 长度
train_data_size=len(train_data)
test_data_size=len(test_data)
# 如果train_data_size=10,训练数据集的长度为：10
print("训练数据集的长度为:{}".format(train_data_size))
print("测试数据集的长度为:{}".format(test_data_size))

# 利用DataLoader加载数据集
train_dataloader=DataLoader(train_data,batch_size=64)
test_dataloader=DataLoader(test_data,batch_size=64)

# 创建网络模型

class Qianshi(nn.Module):
    def __init__(self):
        super(Qianshi, self).__init__()
        self.model=nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4,64),
            nn.Linear(64,10)
        )
    def forward(self,x):
        x=self.model(x)
        return x


qianshi=Qianshi()
# 利用gpu
if torch.cuda.is_available():
    qianshi=qianshi.cuda()

# 损失函数
loss_fn=nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn=loss_fn.cuda()

#定义优化器
# 1e-2=1*(10)^(-2)
learning_rate=0.01
optimizer=torch.optim.SGD(qianshi.parameters(),lr=learning_rate)


#设置训练网络的一些参数
# 记录训练的次数
total_train_step=0
# 记录测试的轮数
test_train_step=0
# 训练的轮数
epoch=30

# 添加TensorFlowboard

writer=SummaryWriter("./logs_train")
start_time=time.time()

for i in range(10):
    print("-------第{}轮训练开始-------".format(i+1))

    # 训练步骤开始
    qianshi.train()

    for data in train_dataloader:
        imgs,targets=data
        if torch.cuda.is_available():
            imgs=imgs.cuda()
            targets=targets.cuda()
        outputs=qianshi(imgs)
        loss=loss_fn(outputs,targets)

#         优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step+=1
        if total_train_step%100==0:
            end_time=time.time()
            print(end_time-start_time)
            print("训练次数：{},loss:{}".format(total_train_step,loss.item()))
            writer.add_scalar("train_loss",loss.item(),total_train_step)

    # 测试步骤开始 测试说明无梯度的训练好的
    total_test_loss=0
    qianshi.eval()
    # 整体正确的个数
    total_accuracy=0
    with torch.no_grad():
        for data in test_dataloader:
            imgs,targets=data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs=qianshi(imgs)
            loss=loss_fn(outputs,targets)
            total_test_loss=total_test_loss+loss.item()
            accuracy=(outputs.argmax(1)==targets).sum()
            total_accuracy+=accuracy
    print("整体测试集上的loss:{}".format(total_test_loss))
    print("整体测试集上的正确率:{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss",total_test_loss,test_train_step)
    writer.add_scalar("test_accuracy",total_accuracy/test_data_size,total_test_loss)
    test_train_step+=1

    torch.save(qianshi,"qianshi_{}_gpu.pth".format(i))
    # torch.save(qianshi.state_dict(),"qianshi_{}.pth".format(i))
    print("模型已保存")

writer.close()







