import torch
from torch.nn import L1Loss, MSELoss
from torch import nn
#注意点是那个shape 把大小搞对了就行

inputs=torch.tensor([1,2,3],dtype=torch.float32)
targets=torch.tensor([1,2,5],dtype=torch.float32)

inputs=torch.reshape(inputs,(1,1,1,3))
targets=torch.reshape(targets,(1,1,1,3))

loss=L1Loss(reduction='sum')
result=loss(inputs,targets)


loss_mse=MSELoss()
result_mse=loss_mse(inputs,targets)

print(result_mse)


print(result)

#crossloss那个主要作用
# 1、为计算实际输出和目标之间的差距
# 2、为我们更新输出提供部分一定的依据（反向传播），grad

x=torch.tensor([0.1,0.2,0.3])
y=torch.tensor([1])
x=torch.reshape(x,(1,3))
# 1是batch_size 3是类
loss_cross=nn.CrossEntropyLoss()
result_cross=loss_cross(x,y)
print(result_cross)

