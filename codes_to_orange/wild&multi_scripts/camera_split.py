import os
import shutil

source = '/data01/zhangleiting/datasets/Wildtrack_dataset_full/Wildtrack_dataset/split/full'
target = '/data01/zhangleiting/datasets/Wildtrack_dataset_full/Wildtrack_dataset/camera_split'
imgs = os.listdir(source)
t = 0
for img in imgs:
    camid = img[6]
    # print(img)
    # print(camid)
    # break
    ori = os.path.join(source, img)
    tar = os.path.join(target, camid, img)
    shutil.copy(ori, tar)
    t += 1
    print(t, 'th finished')
