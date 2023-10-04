import torch
outputs=torch.tensor([[0.1,0.2],
                     [0.3,0.4]])
print(outputs.argmax(1))
# 里面是0是1看的是方向 0纵向 1横向
preds=outputs.argmax(1)
targets=torch.tensor([0,1])
print((preds==targets).sum())
# 可以计算出相应位置相等的个数