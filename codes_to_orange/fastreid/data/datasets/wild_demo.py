# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import os.path as osp
import os
import re
import warnings

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class WildDemo(ImageDataset):
    """Market1501.

    Reference:
        Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.

    URL: `<http://www.liangzheng.org/Project/project_reid.html>`_

    Dataset statistics:
        - identities: 1501 (+1 for background).
        - images: 12936 (train) + 3368 (query) + 15913 (gallery).
    """
    dataset_dir = 'Orange_demo_wild/baseline'
    dataset_url = None
    dataset_name = "wild_demo"

    def __init__(self, root='datasets', **kwargs):
        # self.root = osp.abspath(osp.expanduser(root))
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query0134_c6')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery0134_c6')

        required_files = [
            self.dataset_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir)
        query = self.process_dir(self.query_dir, is_train=False)
        gallery = self.process_gallery(self.gallery_dir)

        super(WildDemo, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, is_train=True):
        # print('human is dead')
        img_paths = glob.glob(osp.join(dir_path, '*.png'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        data = []
        # print('mismatched')
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            assert 1 <= camid <= 7
            camid -= 1  # index starts from 0ll
            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
            data.append((img_path, pid, camid))

        return data

    def process_gallery(self, dir_path):
        img_files = sorted(os.listdir(dir_path))
        data = []
        for img_file in img_files:
            file_path = osp.join(dir_path, img_file)
            img_paths = glob.glob(osp.join(file_path, '*.png'))
            pattern = re.compile(r'([-\d]+)_c(\d)')
            for img_path in img_paths:
                pid, camid = map(int, pattern.search(img_path).groups())
                assert 1 <= camid <= 7
                camid -= 1  # index starts from 0
                data.append((img_path, pid, camid))

        # print(data)
        return data
