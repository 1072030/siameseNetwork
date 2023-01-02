import torch
import sys 
print(sys.path[0])
print(torch.__version__)  #注意是雙下劃線
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name())
# print("Epoch:{0},  Current loss {1}\n".format("epoch","loss_contrastive.data[0]"))