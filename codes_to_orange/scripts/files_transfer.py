import os
from os.path import join
import shutil

root = '../datasets/Wildtrack_by_view_and_time/6/'
target = '../datasets/Orange_demo_wild/baseline/gallery0134_c6/'

# Ensure the target directory exists
# 如果路径不存在则创建它
if not os.path.exists(target):
    os.makedirs(target)

files = os.listdir(root)
for file in files:
    if 860 <= int(file) <= 945:
        source_path = join(root, file)
        target_path = join(target, file)
        if os.path.isdir(source_path):
            shutil.copytree(source_path, target_path)
        else:
            shutil.copy(source_path, target_path)

print("finished")