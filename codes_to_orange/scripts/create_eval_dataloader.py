import os
import shutil
import tqdm

'''
目前数据集划分存在问题：所有的train,test,val都基于c1下的数据进行分割，每一组图片中可能包含有c1，没有cx（x in [2:8]）
但不存在没有c1，有cx（x in [2:8]）的情况，导致eval存在偏差。考虑后续再补
'''
# source_path = '../datasets/Wildtrack_splited/'
source_path = '../datasets/Wildtrack_splited/'
# output_path = '../datasets/Wildtrack_eval/query'
output_path = '../datasets/Wildtrack_eval/train'
if not os.path.exists(output_path):
    os.makedirs(output_path)

id_time_set = set()
for name in tqdm.tqdm(os.listdir(os.path.join(source_path, '1', 'train'))):
    id = name[:4]
    t = name[7:13]
    if (id, t) not in id_time_set:
        id_time_set.add((id, t))
    dir_path = os.path.join(output_path, id + t)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    for num in range(1, 8):
        img_path = os.path.join(source_path, str(num), 'train')
        imgs = os.listdir(img_path)
        for img in imgs:
            # pass
            # print(img)
            if img.startswith(id) and img.endswith(t + '.png'):
                shutil.copyfile(os.path.join(img_path, img),
                                os.path.join(dir_path, img))
