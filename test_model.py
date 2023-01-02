import torch
import torch.nn as nn
import torch.nn.functional as F

# Dropout = nn.Dropout(p = 0.2)

# Dropout2d = nn.Dropout2d(p = 0.2)
# input_tensor = torch.randn(3,4,4)

# print(Dropout(input_tensor))
# print(Dropout2d(input_tensor))
#---------------------------------
#nn.Linear 像是向量相乘 128^20  *  20^30 = 128^30
# x = torch.randn(128,20)
# m = torch.nn.Linear(20,30)
# output = m(x)

# print(m.weight.shape)
# print(m.bias.shape)
# print(output.shape)

# ans = torch.mm(x,m.weight.t()) + m.bias
# print(ans.shape)

# print( torch.equal(ans,output))
#----------------------------------------
# x = torch.randn(1,1,3,4)
# print(x)
# print(x.view(x.size()[0],-1))
#----------------------------------------
# input = [3,4,6,5,
#          2,4,6,8,
#          1,6,7,8,
#          9,7,4,6]
# input = torch.Tensor(input).view(1,1,4,4)
# maxPooling_layer = torch.nn.MaxPool2d(kernel_size=2)

# output = maxPooling_layer(input)
# print(output)