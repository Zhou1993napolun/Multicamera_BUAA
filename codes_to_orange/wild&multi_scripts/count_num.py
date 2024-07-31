import os

source = '/data01/zhangleiting/datasets/Wildtrack_dataset_full/Wildtrack_dataset/processed'

dirs = os.listdir(source)
l = []
t = 0
for dir in dirs:
    num = int(dir)
    l.append(num)
    t += 1
print(t)
print(l)
