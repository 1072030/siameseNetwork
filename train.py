
import torchvision.datasets as dset
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
from models import SiameseNetwork
from mydataset import MyDataset
from loss import ContrastiveLoss
import torch.optim as optim
import matplotlib.pyplot as plt
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
train_data = MyDataset(txt = sys.path[0]+'\data.txt',transform=transforms.Compose(
            [transforms.Resize((224,224)),transforms.ToTensor()]), should_invert=False)     #Resize到100,100

train_dataloader = DataLoader(dataset=train_data, shuffle=True, num_workers=0,batch_size=10)
net = SiameseNetwork()     # GPU加速
criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0005) #梯度校正 


counter = []
loss_history =[]
iteration_number =0
label_sum=0
correct_sum = 0
accuracy_sum={}
count=0
for epoch in range(0, 50): #每一輪的訓練次數
      print("--第{0}輪開始--\n".format(epoch))
      for i, data in enumerate(train_dataloader, 0):
            count+=1
            img0, img1, label = data
            img0, img1, label = Variable(img0), Variable(img1), Variable(label)  #使用GPU
            output1, output2= net(img0, img1)#開始訓練
            optimizer.zero_grad() #將所有參數的梯度緩衝區（buffer）歸零
            loss_contrastive = criterion(output1, output2, label) #loss計算
            a = torch.max(output1,1)[1]
            b = torch.max(output2,1)[1]
            correct = (a == b).sum().item()
            correct_sum+=correct
            label_sum+=label.size(0)
            print("correct:{0},label_size:{1}".format(correct,label.size(0)))
            loss_contrastive.backward()#方法開始進行反向傳播
            optimizer.step()#方法來更新權重
            # print(net)
            if (count == 3):
                  print("Current loss {0}".format(loss_contrastive.item())) #console顯示
                  print("Accuracy:{0}".format(correct_sum/label_sum))
                  accuracy_sum[epoch]=correct_sum/label_sum
                  print("sum:{0},correct:{1}\n".format(correct_sum,label_sum))
                  iteration_number += 10
                  counter.append(iteration_number)
                  loss_history.append(loss_contrastive.item())
                  label_sum=0
                  correct_sum = 0
                  count=0
save_dir="logs"
torch.save(net.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
print(accuracy_sum.items())
plt.plot(counter, loss_history)     # plot 損失函數曲線
plt.show()