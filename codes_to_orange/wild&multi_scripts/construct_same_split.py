import os
import os.path as osp
import shutil

dst = '/data01/zhangleiting/datasets/Wildtrack_dataset_full/Wildtrack_dataset/subset_split'
src = '/data01/zhangleiting/datasets/Wildtrack_dataset_full/Wildtrack_dataset/camera_split_preprocess'
# print(os.listdir(path1))

train_dict = {}
test_dict = {}
val_dict = {}

dicts = [train_dict, test_dict, val_dict]
splits = ['train', 'test', 'val']

path1 = osp.join(dst, '1')
for num, split in enumerate(splits):
    for name in os.listdir(osp.join(path1, split)):
        if name[0:4] + name[9:13] not in dicts[num]:
            dicts[num][name[0:4] + name[9:13]] = 1
        else:
            print('repeated')
print(train_dict)

for num in range(2, 8):
    train_num = 0
    test_num = 0
    val_num = 0
    src_path = osp.join(src, str(num))
    dst_path = osp.join(dst, str(num))
    if not osp.exists(dst_path):
        os.makedirs(dst_path)
    for name in os.listdir(src_path):
        # print(name)
        if name[0:4] + name[9:13] in train_dict:
            path = osp.join(dst_path, 'train')
            if not osp.exists(path):
                os.makedirs(path)
            shutil.copy(osp.join(src_path, name), osp.join(path, name))
            train_num += 1
        if name[0:4] + name[9:13] in test_dict:
            path = osp.join(dst_path, 'test')
            if not osp.exists(path):
                os.makedirs(path)
            shutil.copy(osp.join(src_path, name), osp.join(path, name))
            test_num += 1
        if name[0:4] + name[9:13] in val_dict:
            path = osp.join(dst_path, 'val')
            if not osp.exists(path):
                os.makedirs(path)
            shutil.copy(osp.join(src_path, name), osp.join(path, name))
            val_num += 1
    print('{}th camera: train: {}, test: {},val: {}'.format(
        str(num), train_num, test_num, val_num))
