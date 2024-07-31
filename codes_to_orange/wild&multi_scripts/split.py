'''
import os
import shutil
'''
import splitfolders
source = '/data01/zhangleiting/datasets/Wildtrack_dataset_full/Wildtrack_dataset/split/'
output = '/data01/zhangleiting/datasets/Wildtrack_dataset_full/Wildtrack_dataset/split/output/'
test = '/data01/zhangleiting/datasets/MultiviewX/bounding_box_test/'
query = '/data01/zhangleiting/datasets/MultiviewX/query/'


# train:validation:test=8:1:1
splitfolders.ratio(input=source, output=output,
                   seed=1121, ratio=(0.45, 0.1, 0.45))
