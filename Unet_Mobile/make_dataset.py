import os

#自行修改文件绝对路径
left = os.listdir(r"D:\Deep Learning\makedataset\leftImg8bit\train")
gtfine = os.listdir(r"D:\Deep Learning\makedataset\gtFine\train")

#自行修改txt文件相对路径
with open("D:/Deep Learning/makedataset/read_data/train_data.txt","w") as f:
    for name in left:
        png = name.replace("leftImg8bit","gtFine_labelTrainIds")
        # 判断left是否存在对应的gtfine
        if png in gtfine:
            f.write(name+";"+png+"\n")
