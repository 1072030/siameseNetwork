import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
      #製作 CNN模型
      def __init__(self):
            super(SiameseNetwork, self).__init__()
            self.cnn1 = nn.Sequential(
                  #VGG16
                  #conv2d (in_channels,out_channels)
                  # nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                  # nn.ReLU(inplace=True),
                  # nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                  # nn.ReLU(inplace=True),
                  # nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                  # nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                  # nn.ReLU(inplace=True),
                  # nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                  # nn.ReLU(inplace=True),
                  # nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                  # nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                  # nn.ReLU(inplace=True),
                  # nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                  # nn.ReLU(inplace=True),
                  # nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                  # nn.ReLU(inplace=True),
                  # nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                  # nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                  # nn.ReLU(inplace=True),
                  # nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                  # nn.ReLU(inplace=True),
                  # nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                  # nn.ReLU(inplace=True),
                  # nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                  # nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                  # nn.ReLU(inplace=True),
                  # nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                  # nn.ReLU(inplace=True),
                  # nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                  # nn.ReLU(inplace=True),
                  # nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                  # nn.ReLU(inplace=True)
                  #--------------------------------------------------------------------------------------------
                          # block 1
                  nn.Conv2d(in_channels=3, out_channels=64, padding=1, kernel_size=3, stride=1),
                  nn.ReLU(),
                  nn.Conv2d(in_channels=64, out_channels=64, padding=1, kernel_size=3, stride=1),
                  nn.ReLU(),
                  nn.MaxPool2d(kernel_size=2, stride=2),

                  # block 2
                  nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                  nn.ReLU(),
                  nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                  nn.ReLU(),
                  nn.MaxPool2d(kernel_size=2, stride=2),

                  # block 3
                  nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),
                  nn.ReLU(),
                  nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
                  nn.ReLU(),
                  nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
                  nn.ReLU(),
                  nn.MaxPool2d(kernel_size=2, stride=2),

                  # block 4
                  nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1),
                  nn.ReLU(),
                  nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
                  nn.ReLU(),
                  nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
                  nn.ReLU(),
                  nn.MaxPool2d(kernel_size=2, stride=2),
                  # block 5
                  nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
                  nn.ReLU(),
                  nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
                  nn.ReLU(),
                  nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
                  nn.ReLU(),
                  nn.MaxPool2d(kernel_size=2, stride=2)
                  #--------------------------------------------------------------------------------------------
                  # nn.ReflectionPad2d(1), #利用输入邊界的反射来填充輸入張量。
                  # nn.Conv2d(1, 4, kernel_size=3), #製作卷積層
                  # nn.ReLU(inplace=True),#激活
                  # nn.BatchNorm2d(4), #數據的歸一化處理這使得數據在進行Relu之前不會因為數據過大而導致網絡性能的不穩定，一般輸入參數為batch_size*num_features*height*width，即為其中特徵的數量
                  # nn.Dropout2d(p=.2),#主要的作用是在神經網絡訓練過程中防止模型過擬合
                  
                  # nn.ReflectionPad2d(1),
                  # nn.Conv2d(4, 8, kernel_size=3),
                  # nn.ReLU(inplace=True),
                  # nn.BatchNorm2d(8),
                  # nn.Dropout2d(p=.2),
                      
                  # nn.ReflectionPad2d(1),
                  # nn.Conv2d(8, 8, kernel_size=3),
                  # nn.ReLU(inplace=True),
                  # nn.BatchNorm2d(8),
                  # nn.Dropout2d(p=.2),
                  )
            
            self.fc1 = nn.Sequential(
                  nn.Linear(512*7*7, 4096), #線性變換
                  nn.ReLU(inplace=True),
                  nn.Dropout(0.5),

                  nn.Linear(4096, 4096),
                  nn.ReLU(inplace=True),
                  nn.Dropout(0.5),

                  nn.Linear(4096, 3)#self.num_classes
                  )

      def forward_once(self, x):
            output = self.cnn1(x) #丟入卷積層
            output = output.view(output.size()[0], -1) #重新resize成同一個陣列
            output = self.fc1(output) #進行線性變換
            return output
      
      def forward(self, input1, input2):
            #被呼叫時即觸發
            output1 = self.forward_once(input1)
            output2 = self.forward_once(input2)
            return output1, output2
