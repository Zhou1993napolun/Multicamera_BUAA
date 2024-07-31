from __future__ import print_function, absolute_import
import os.path as osp
from glob import glob
import re
import logging

from tabulate import tabulate
from termcolor import colored
# query: split by batch
# gallery: split by cam id

logger = logging.getLogger(__name__)


class WildEvalDemo(object):
    def __init__(self, root, subname):
        self.image_dir = osp.join(root)
        self.train_path = 'train'
        self.gallery_path = 'gallery' + subname
        self.gallery_mmoe_path = 'gallery' + subname + '_mmoe'
        self.query_path = 'query' + subname
        self.gallery, self.query, self.train, self.gallery_mmoe = {}, {}, {}, {}
        self.query_ids, self.train_ids, self.gallery_mmoe_ids = {}, {}, {}
        self.num_query_images, self.num_train_images, self.num_gallery_mmoe_images = 0, 0, 0
        self.gallery_ids, self.num_gallery_images = {}, {}
        self.load()
        # query数据输出格式：{dir_name:[[fname1, pid, cam1], [fname2, pid, cam2], ...[fname7, pid, cam7]]}
        # gallery数据输出格式：{'camid':[[fname1, pid, camid], ...], ...}

    def preprocess(self, subset='query', type='query', relabel=False):
        pattern = re.compile(r'([-\d]+)_c(\d)')
        all_pids = {}
        ret = {}
        cur_path = osp.join(self.image_dir, subset)
        sub_dirs = sorted(glob(osp.join(cur_path, '*')))
        # 子目录，例如[1,2,3,...] or [0000_t0000,...]
        for sub_dir in sub_dirs:
            fpaths = sorted(glob(osp.join(sub_dir, '*.png')))
            # 图片目录，提取pid和camid
            if type == 'gallery':
                sub_pids = {}
            tmp = []
            for fpath in fpaths:
                fname = osp.basename(fpath)
                pid, cam = map(int, pattern.search(fpath).groups())
                if type in ['query', 'train', 'gallery_mmoe']:
                    if relabel:
                        if pid not in all_pids:
                            all_pids[pid] = len(all_pids)
                    else:
                        if pid not in all_pids:
                            all_pids[pid] = pid
                    pid = all_pids[pid]
                else:
                    if relabel:
                        if pid not in sub_pids:
                            sub_pids[pid] = len(sub_pids)
                    else:
                        if pid not in sub_pids:
                            sub_pids[pid] = pid
                    pid = sub_pids[pid]
                # cam -= 1
                tmp.append([fname, pid, cam])
            if type in ['query', 'train', 'gallery_mmoe']:
                subdir_name = osp.basename(sub_dir)
                ret[subdir_name] = tmp
                self.num_query_images += len(fpaths)
            elif type == 'gallery':
                camid = osp.basename(sub_dir)
                ret[camid] = tmp
                self.num_gallery_images[camid] = len(fpaths)
                all_pids[camid] = sub_pids
            else:
                raise NameError('wrong subset name {}'.format(subset))
        return ret, all_pids

    def load(self):
        print(3333333333333333333333333333333333333333333333)
        print(self.query_path)
        print(4444444444444444444444444444444444444444444444)
        print(self.gallery_path)
        self.query, self.query_ids = self.preprocess(self.query_path, type='query')
        self.gallery_mmoe, self.gallery_mmoe_ids = self.preprocess(self.gallery_mmoe_path)
        self.gallery, self.gallery_ids = self.preprocess(self.gallery_path, type='gallery')
        # self.train, self.train_ids = self.preprocess(self.train_path)

    # def show_train(self):
    #     # 展示形式: subset, # id, # images, # batches
    #     # train, len(all_pid), self.num_query_images, len(ret)
    #     headers = ['subset', '# ids', '# images', '# batches']
    #     csv_results = [
    #         ['train', len(self.query_ids), self.num_query_images, len(self.query)]]
    #     table = tabulate(
    #         csv_results,
    #         tablefmt="pipe",
    #         headers=headers,
    #         numalign="left",
    #     )
    #     print(colored(table))
    #     logger.info(colored(table))

    # def show_query(self):
    #     # 展示形式: subset, # id, # images, # batches
    #     # query, len(all_pid), self.num_query_images, len(ret)
    #     headers = ['subset', '# ids', '# images', '# batches']
    #     csv_results = [
    #         ['query', len(self.query_ids), self.num_query_images, len(self.query)]]
    #     table = tabulate(
    #         csv_results,
    #         tablefmt="pipe",
    #         headers=headers,
    #         numalign="left",
    #     )
    #     print(colored(table))
    #     logger.info(colored(table))

    # def show_gallery_mmoe(self):
    #     # 展示形式: subset, # id, # images, # batches
    #     # query, len(all_pid), self.num_query_images, len(ret)
    #     headers = ['subset', '# ids', '# images', '# batches']
    #     csv_results = [
    #         ['gallery', len(self.gallery_mmoe_ids), self.num_gallery_mmoe_images, len(self.gallery)]]
    #     table = tabulate(
    #         csv_results,
    #         tablefmt="pipe",
    #         headers=headers,
    #         numalign="left",
    #     )
    #     print(colored(table))
    #     logger.info(colored(table))

    # def show_gallery(self):
    #     # 展示形式: subset, camid, # id, # images
    #     # gallery, 1-7, len(all_pids), self.num_gallery_images[i]
    #     headers = ['subset', 'camid', '# ids', '# images']
    #     csv_results = [['query', str(i), len(self.gallery_ids[str(
    #         i)]), self.num_gallery_images[str(i)]] for i in range(1, 8)]
    #     table = tabulate(
    #         csv_results,
    #         tablefmt="pipe",
    #         headers=headers,
    #         numalign="left",
    #     )
    #     print(colored(table))
    #     logger.info(colored(table))
