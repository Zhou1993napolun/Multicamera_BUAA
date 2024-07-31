from __future__ import absolute_import
import os.path as osp
from PIL import Image


class PreprocessTrain(object):
    def __init__(self, train, root=None, transform=None):
        super(PreprocessTrain, self).__init__()
        self.train = train
        self.root = root
        self.transform = transform
        self.index_dict = {}
        self.__create_index_dict__()

    def __len__(self):
        return len(self.train)

    def __create_index_dict__(self):
        i = 0
        for k, _ in self.train.items():
            if i not in self.index_dict:
                self.index_dict[i] = k
                i += 1
            else:
                raise NameError('repeated dir name')

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        subdir_name = self.index_dict[index]
        sub_imgs = self.train[subdir_name]
        res = []
        for sub_img in sub_imgs:
            fname, pid, camid = sub_img
            fpath = fname
            if self.root is not None:
                fpath = osp.join(self.root, subdir_name, fname)
            img = Image.open(fpath).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            res.append([img, fname, pid, camid])
            # 返回格式[[img,fname,pid,camid],[img.fname,pid,camid],...,[img,fname,pid,camid]]
        return res

#########################################################################
class PreprocessQuery(object):
    def __init__(self, query, root=None, transform=None):
        super(PreprocessQuery, self).__init__()
        self.query = query

        with open('orange_demo/mv_video/0119_t0575to1995/queries_paths.txt', 'a') as f:
            for fname, img_names in query.items():
                # print(11111111111111111111111)
                f.write(str(fname))
                for img_name in img_names:
                    f.write(str(img_name[2]))
                f.write('\n')
        print(query)

        self.root = root
        self.transform = transform
        self.index_dict = {}
        self.__create_index_dict__()

    def __len__(self):
        return len(self.query)

    def __create_index_dict__(self):
        i = 0
        for k, _ in self.query.items():
            if i not in self.index_dict:
                self.index_dict[i] = k
                i += 1
            else:
                raise NameError('repeated dir name')

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        subdir_name = self.index_dict[index]
        sub_imgs = self.query[subdir_name]
        res = []
        for sub_img in sub_imgs:
            fname, pid, camid = sub_img
            fpath = fname
            if self.root is not None:
                fpath = osp.join(self.root, subdir_name, fname)
            img = Image.open(fpath).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            res.append([img, fname, pid, camid])
            # 返回格式[[img,fname,pid,camid],[img.fname,pid,camid],...,[img,fname,pid,camid]]
        return res

##########################################################################################################
class PreprocessGallery(object):
    def __init__(self, gallery, camid, root=None, transform=None):
        super(PreprocessGallery, self).__init__()
        ################################
        ##########################################################################################################
        ##########################################################################################################
        ##########################################################################################################
        # print('==================================================================')
        # print(gallery)
        # print(555555555555555555555555555555)
        # print("camid:", camid)
        # print("gallery keys:", list(gallery.keys()))
        ##########################################################################################################
        ##########################################################################################################
        ##########################################################################################################
        ################################
        self.gallery = gallery[camid]
        with open('orange_demo/mv_video/0119_t0575to1995/galleries_paths.txt', 'a') as f:
            for fname, _, _ in self.gallery:
                f.write(str(fname) + '\n')

        self.root = root
        self.transform = transform
        self.camid = camid

    def __len__(self):
        return len(self.gallery)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.gallery[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, self.camid, fname)
        img = Image.open(fpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        # 返回格式
        return img, fname, pid, camid


class PreprocessGalleryAll(object):
    def __init__(self, gallery, camid, root=None, transform=None):
        super(PreprocessGalleryAll, self).__init__()
        self.gallery = gallery[camid]
        self.root = root
        self.transform = transform
        self.camid = camid

    def __len__(self):
        return len(self.gallery)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.gallery[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, self.camid, fname)
        img = Image.open(fpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        # 返回格式
        return img, fname, pid, camid
