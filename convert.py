import os
import sys

def convert(train=True):
    if(train):
        # print(sys.path[0])
        f=open(sys.path[0]+'/data.txt', 'w')
        dir_path = os.path.dirname(os.path.realpath(__file__))
        print(dir_path)
        data_path=dir_path+"/data"
        print(data_path)
        data_folder = os.listdir(data_path)
        for i in data_folder:
            family_json = os.listdir(data_path+"/"+i)
            for j in family_json:
                if i == "egregor_picture":
                    f.write(data_path+"/"+i+"/"+j+" 1"+"\n")
                elif i == "kingofhearts_picture":
                    f.write(data_path+"/"+i+"/"+j+" 2"+"\n")
                elif i == "valak_picture":
                    f.write(data_path+"/"+i+"/"+j+" 3"+"\n")
            # print(data_path)
        f.close()
convert()
