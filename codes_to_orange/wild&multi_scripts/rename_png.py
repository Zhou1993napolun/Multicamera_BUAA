import os
import shutil
import sys

source = '/data01/zhangleiting/datasets/Wildtrack_dataset_full/Wildtrack_dataset/processed'
target = '/data01/zhangleiting/datasets/Wildtrack_dataset_full/Wildtrack_dataset/full'

indexes = os.listdir(source)
total = 0
for index in indexes:
    ppath = os.path.join(source, index)
    img_paths = os.listdir(ppath)
    name0 = index.zfill(4)
    count = 0
    for img_path in img_paths:
        # print(img_path)
        # sys.exit(0)
        count += 1
        total += 1
        camid = img_path[-5]
        tid = img_path[-10:-6]
        full_tail = name0+'_c'+camid+'_t'+tid+'.png'
        source_path = os.path.join(ppath, img_path)
        target_path = os.path.join(target, full_tail)
        # print(source_path)
        # print(target_path)
        # sys.exit()
        shutil.copy(source_path, target_path)
        print(index, 'person', count, 'th img transfered')
        print('totally', total, 'th img transfered')
