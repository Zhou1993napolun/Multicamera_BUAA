import os
import shutil
import sys

source = '/data01/zhangleiting/datasets/Wildtrack_dataset_full/Wildtrack_dataset/split/full/'
target = '/data01/zhangleiting/datasets/Wildtrack_dataset_full/Wildtrack_dataset/time_ordered/'

names = os.listdir(source)
total = 0
dict1 = dict()  # 人物id
dict2 = dict()  # 时序信息
for name in names:
    id = name[:4]
    time = name[-8:-4]
    path = os.path.join(target, id+'_t'+time)
    if not os.path.exists(path):
        os.makedirs(path)
        print(id, '_t', time, 'is created')
    source_path = os.path.join(source, name)
    target_path = os.path.join(target, path, name)
    shutil.copy(source_path, target_path)
    total += 1
    print(total, 'th successfully transferred')
