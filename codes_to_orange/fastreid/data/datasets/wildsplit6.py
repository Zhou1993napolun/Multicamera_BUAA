# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import os.path as osp
import re
import warnings

from .bases import ImageDataset
from . import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class WildSplit6(ImageDataset):
    dataset_dir = 'Wildtrack_splited_rerank/6'
    dataset_url = None
    dataset_name = "wildsplit6"

    def __init__(self, root='datasets', **kwargs):
        # self.root = osp.abspath(osp.expanduser(root))
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'val')
        self.gallery_dir = osp.join(self.dataset_dir, 'test')

        required_files = [
            self.dataset_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir, is_query=False)
        query = self.process_dir(self.query_dir, is_train=False)
        gallery = self.process_dir(
            self.gallery_dir, is_train=False, is_query=False)

        super(WildSplit6, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, is_train=True, is_query=True):
        # print('human is dead')
        img_paths = glob.glob(osp.join(dir_path, '*.png'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        data = []
        # print('mismatched')
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if is_query:
                camid += 1  # 训练要求query和gallery的camera id不一致
            assert 1 <= camid <= 7
            camid -= 1  # index starts from 0
            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
            data.append((img_path, pid, camid))

        return data
