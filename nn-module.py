import torch
from torch import nn

class Qianshi(nn.Module):
    
    def  __init__(self):
        super().__init__()


    def forward(self,input):
        output=input+1
        return output

qianshi=Qianshi()
x=torch.tensor(1.0)
output=qianshi(x)
print(output)
   
       














