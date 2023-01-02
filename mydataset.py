import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import linecache
import PIL
from PIL import Image
import numpy as np


class MyDataset(Dataset):    # 集成Dataset類
      def __init__(self, txt, transform=None, target_transform=None, should_invert=False):
            self.transform = transform
            self.target_transform = target_transform
            # self.should_invert = should_invert
            self.txt = txt       # 之前生成的train.txt
      def __getitem__(self, index):
            line = linecache.getline(self.txt, random.randint(1, self.__len__()))   # 隨機選擇
            line.strip('\n')
            img0_list= line.split()
            should_get_same_class = random.randint(0,1)     # 隨機數0或1，是否選擇同一個的，這里為了保證盡量使匹配和非匹配數據大致平衡（正負類樣本相當）
            if should_get_same_class:    # 執行的話就挑一張同一個作為匹配樣本對
                  while True:
                        img1_list = linecache.getline(self.txt, random.randint(1, self.__len__())).strip('\n').split()
                        if img0_list[1]==img1_list[1]:
                              break
            else:       # else就是隨意挑一個作為非匹配樣本對，當然也可能抽到同一個，概率較小而已
                  img1_list = linecache.getline(self.txt, random.randint(1,self.__len__())).strip('\n').split()
            
            img0 = Image.open(img0_list[0])    # img_list都是大小為2的列表，list[0]為圖像, list[1]為label
            img1 = Image.open(img1_list[0])
            # img0 = img0.convert("L")           # 轉為灰階
            # img1 = img1.convert("L")
        
            # if self.should_invert:             # 是否進行像素反轉操作，即0變1,1變0
            #     img0 = PIL.ImageOps.invert(img0)
            #     img1 = PIL.ImageOps.invert(img1)
  
            if self.transform is not None:     # 非常方便的transform操作，在實例化時可以進行任意定制
                img0 = self.transform(img0)
                img1 = self.transform(img1)
        
            return img0, img1 , torch.from_numpy(np.array([int(img1_list[1]!=img0_list[1])],dtype=np.float32))    # 注意一定要返回數據+標籤， 這裡返回一對圖像+label（應由numpy轉為tensor）
    
      def __len__(self):       # 總長
            fh = open(self.txt, 'r')
            num = len(fh.readlines()) #40
            fh.close()
            return num