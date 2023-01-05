import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
      #製作 CNN模型
      def __init__(self):
            super(SiameseNetwork, self).__init__()
            self.cnn1 = nn.Sequential(
                  #--------------------------------------------------------------------------------------------
                  nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1), # 64 + 2*0 -3 /1 +1 |62*62*16
                  nn.ReLU(),
                  nn.MaxPool2d(kernel_size=2, stride=2), #31*31*16
                  #block 2
                  nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1), # 31 + 2*0 -3 /1 +1 | 29*29 *32
                  nn.ReLU(),
                  nn.MaxPool2d(kernel_size=2, stride=2), # 14*14*32
                  # # block 3
                  nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1), # 14 + 2*0 -3 /1 +1 | 12*12 *64
                  nn.ReLU(),
                  nn.MaxPool2d(kernel_size=2, stride=2), # 6*6*64
                  )
            
            self.fc1 = nn.Sequential(
                  nn.Linear(6*6*64, 64), #線性變換
                  nn.ReLU(inplace=True),

                  nn.Linear(64, 16),
                  nn.ReLU(inplace=True),

                  nn.Linear(16, 4)#self.num_classes
                  )

      def forward_once(self, x):
            output = self.cnn1(x) #丟入卷積層
            output = output.view(output.size()[0], -1) #重新resize成同一個陣列
            output = self.fc1(output) #進行線性變換
            output = torch.nn.Sigmoid()(output)
            return output
      
      def forward(self, input1, input2):
            #被呼叫時即觸發
            output1 = self.forward_once(input1)
            output2 = self.forward_once(input2)
            return output1, output2
