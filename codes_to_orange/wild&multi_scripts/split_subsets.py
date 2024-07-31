'''
import os
import shutil
'''
import splitfolders
source = '/data01/zhangleiting/datasets/Wildtrack_dataset_full/Wildtrack_dataset/camera_split_preprocess/1/'
output = '/data01/zhangleiting/datasets/Wildtrack_dataset_full/Wildtrack_dataset/subset_split/1/output'
test = '/data01/zhangleiting/datasets/MultiviewX/bounding_box_test/'
query = '/data01/zhangleiting/datasets/MultiviewX/query/'


# train:validation:test=8:1:1
splitfolders.ratio(input=source, output=output,
                   seed=1121, ratio=(0.45, 0.1, 0.45))

# 应该先整合出来多个
# Q: 为什么我没有先组合再划分train, query, gallery