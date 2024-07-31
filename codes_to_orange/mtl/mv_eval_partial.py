'''
单独跑一边测试
再写一遍自己的测试方式
尝试从数据集、模型层面调整
'''
import sys
sys.path.append('.')
if True:
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    import torch
    from mtl.utils.preprocessor import PreprocessTrain, PreprocessQuery, PreprocessGallery
    from mtl.utils.rank import evaluate_wild, calculate_score, calculate_score_by_group
    from mtl.datasets.wildeval import WildEval
    from mtl.datasets.wildevaldemo import WildEvalDemo
    from mtl.datasets.multieval import MultiEval
    from fastreid.data.transforms.build import build_transforms
    from fastreid.config import get_cfg
    # , default_setup, launch
    from fastreid.engine import DefaultTrainer, default_argument_parser
    from fastreid.utils.checkpoint import Checkpointer
    from fastreid.utils.logger import setup_logger
    from fastreid.utils.compute_dist import build_dist
    # from torchvision import transforms
    from torch.utils.data import DataLoader  # , Dataset
    from tqdm import tqdm
    # import logging

    # 是否禁止并行

    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def setup(config_file, opts):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # 比较和cfg默认参数不同的设置并进行替换
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    cfg.freeze()
    # default_setup(cfg, args)

    return cfg

# 构建由帧和id决定的行人图片组，用于分别由不同的模型进行测试
# 先构建一个非dataloader版本，用for循环处理（已完成wildtrack_eval）


def prepare_data(name='wild', root='../datasets', eval_mode=0, subname='0011'):
    if eval_mode:
        full_path = os.path.join(root, 'Orange_demo_wild', 'mv')
        return WildEvalDemo(full_path, subname)
    if name == 'wild':
        # full_path = os.path.join(root, 'Wildtrack_eval')
        full_path = os.path.join(root, 'Wildtrack_eval_rerank')
        return WildEval(full_path)
    elif name == 'multi':
        full_path = os.path.join(root, 'MultiviewX_eval')
        return MultiEval(full_path)


def create_train_dataloader(train, cfg, root='../datasets/Wildtrack_eval_rerank'):
    transforms = build_transforms(cfg, is_train=True)
    train_dataloader = DataLoader(
        PreprocessTrain(train=train, root=os.path.join(
            root, 'train'), transform=transforms), batch_size=1, num_workers=4, shuffle=False, pin_memory=False)
    return train_dataloader


def create_query_dataloader(query, cfg, root='../datasets/Wildtrack_eval_rerank', eval_mode=0, name='0011'):
    transforms = build_transforms(cfg, is_train=False)
    if not eval_mode:
        query_dataloader = DataLoader(
            PreprocessQuery(query=query, root=os.path.join(
                root, 'query'), transform=transforms), batch_size=1, num_workers=4, shuffle=False, pin_memory=False)
    else:
        query_dataloader = DataLoader(
            PreprocessQuery(query=query, root=os.path.join(
                root, 'query' + name), transform=transforms), batch_size=1, num_workers=4, shuffle=False, pin_memory=False)
    return query_dataloader


def create_gallery_dataloader(gallery, cfg, num_cam=7, root='../datasets/Wildtrack_eval_rerank', eval_mode=0, name='0011'):
    transforms = build_transforms(cfg, is_train=False)
    gallery_dataloader_dict = {}
    if not eval_mode:
        for i in range(1, num_cam + 1):
            subset = str(i)
            gallery_dataloader_dict[i] = DataLoader(
                PreprocessGallery(gallery=gallery, camid=subset, root=os.path.join(
                    root, 'gallery'), transform=transforms), batch_size=64, num_workers=4, shuffle=False, pin_memory=False)
    else:
        for i in range(1, num_cam + 1):
            subset = str(i)
            gallery_dataloader_dict[i] = DataLoader(
                PreprocessGallery(gallery=gallery, camid=subset, root=os.path.join(
                    root, 'gallery' + name), transform=transforms), batch_size=64, num_workers=4, shuffle=False, pin_memory=False)

    return gallery_dataloader_dict


def process_gallery(model_dict, gallery_dataloader_dict, num_cam):
    # assert len(model_dict) == len(gallery_dataloader_dict), 'mismatched'
    # num_cam = len(model_dict)
    features_dict = {}
    targets_dict = {}
    with torch.no_grad():
        for i in range(1, num_cam + 1):
            if i not in model_dict.keys():
                continue
            features_list, targets_list = [], []
            dataloader = gallery_dataloader_dict[i]
            model = model_dict[i]
            device = model.device
            for idx, datas in enumerate(tqdm(dataloader)):
                # 测试时启用
                # if idx > 1:
                #     break
                inputs, _, pids, _ = datas
                targets = pids.to(device)
                # shape[64,3,384,128]
                imgs = inputs.to(device)
                # shape[64,2048]
                features = model(imgs)
                features_list.append(features)
                targets_list.append(targets)
            features_dict[i] = torch.cat(features_list, dim=0)
            targets_dict[i] = torch.cat(targets_list, dim=0)
    return features_dict, targets_dict
    # features_dict: (key值1-7)，value shape: [3829,2048], ...
    # targets_dict: (key值1-7)，value shape: tensor[3829], ...


def predict(query_features, query_cameras, gallery_features_dict, cfg):
    # 计算query和对应camera id下gallery feature下的维度
    dist = []
    for batch, query_features_batch in enumerate(tqdm(query_features)):
        tmp_dist = []
        for idx, query_feature in enumerate(query_features_batch):
            camid = query_cameras[batch][idx]
            gallery_features = gallery_features_dict[camid]
            query_feature = query_feature.unsqueeze(0)
            cur_dist = build_dist(
                query_feature, gallery_features, cfg.TEST.METRIC)
            tmp_dist.append(cur_dist)
        dist.append(tmp_dist)
    return dist


def process_query(model_dict, query_dataloader):
    features_total = []
    target_total = []
    cam_total = []
    with torch.no_grad():
        for idx, datas in enumerate(tqdm(query_dataloader)):
            # 测试时启用
            # if idx > 3:
            #     break
            tmp_feature_list = []
            tmp_target_list = []
            tmp_cam_list = []
            for data in datas:
                input, _, pid, cam = data
                # 由于Dataloader一组会带来多个输入，所以无法在这里统一模型
                # 考虑到即使用了Dataloader中的batch方法一次导入多组输入，在使用模型的时候任然
                # 要筛选，最终输入的数据可能还是[1,x]的，并没有用到batch的优势，所以考虑在
                # Dataloader的时候就对query的batch设为1，在输入模型的时候对必要的维度
                # 进行squeeze操作（自动压缩大小为1的维度）
                if int(cam.squeeze()) not in model_dict.keys():
                    continue
                camid = int(cam.squeeze())
                model = model_dict[camid]
                device = model.device
                # shape[1,3,384,128]
                img = input.to(device)
                target = pid.to(device)
                feature = model(img)
                tmp_feature_list.append(feature)
                tmp_target_list.append(target)
                tmp_cam_list.append(camid)
            if not tmp_feature_list:
                continue
            features_total.append(torch.cat(tmp_feature_list, dim=0))
            target_total.append(torch.cat(tmp_target_list, dim=0))
            cam_total.append(tmp_cam_list)
    return features_total, target_total, cam_total
    # feature_total: 一个850长度的列表，每一个里面的维度为(6,2048)(取决于camera数量)
    # target_total: 一个850长度的列表，每一个里面的维度为tensor[6](取决于camera数量)
    # cam_total: 一个850长度的列表，每一个里面的维度为tensor[6](取决于camera数量)


# def predict(query_features, query_cameras, gallery_features_dict, cfg):
#     # 计算query和对应camera id下gallery feature下的维度
#     dist = []
#     for batch, query_features_batch in enumerate(tqdm(query_features)):
#         tmp_dist = []
#         for idx, query_feature in enumerate(query_features_batch):
#             camid = query_cameras[batch][idx]
#             gallery_features = gallery_features_dict[camid]
#             query_feature = query_feature.unsqueeze(0)
#             cur_dist = build_dist(
#                 query_feature, gallery_features, cfg.TEST.METRIC)
#             tmp_dist.append(cur_dist)
#         dist.append(tmp_dist)
#     return dist


def evaluate(dist, query_targets, gallery_targets, query_cameras, max_rank):
    # 由dist得到的每组距离，计算每个距离前n个排名最高的值对应的index。
    # 找到每个距离对应的camid，在gallery_targets[camid]下对应的index中
    # 得到预测的pid。与query_targets对应组下对应序号下的pid对比判断正否
    return evaluate_wild(dist=dist, q_targets=query_targets,
                         g_targets=gallery_targets, q_cameras=query_cameras, max_rank=max_rank)


def load_models(dataset, model_type='mgn_R50-ibn', is_origin=0, cam_list=[]):
    if dataset == 'wild':
        views = 7
        config_paths = {}
        for i in range(1, views + 1):
            if i not in cam_list:  # 只考虑视角1和视角6
                continue
            config_paths[i] = './configs/WildSplit' + \
                str(i) + '/' + model_type + '.yml'
        # 日志
        logger = setup_logger('./logs/wild/train_log.txt')
        # 构建多模型字典
        models = {}
        # 构建多配置字典
        cfgs = {}
        # num_classes_dict = {
        #     1: 286,
        #     2: 271,
        #     3: 258,
        #     4: 87,
        #     5: 204,
        #     6: 285,
        #     7: 98
        # }
        num_classes_dict = {
            1: 297,
            2: 297,
            3: 297,
            4: 297,
            5: 297,
            6: 297,
            7: 297
        }
        for i in range(1, views + 1):
            if i not in cam_list:
                continue
            if is_origin:
                opts = ['MODEL.WEIGHTS', './logs/wild/' +
                        model_type + '/model_best.pth',
                        'MODEL.DEVICE', '\"cuda:1\"',
                        'MODEL.HEADS.NUM_CLASSES', num_classes_dict[i],
                        ]
            else:
                opts = ['MODEL.WEIGHTS', './logs/wildsplit' +
                        str(i) + '/' + model_type + '/model_best.pth',
                        'MODEL.DEVICE', '\"cuda:1\"',
                        'MODEL.HEADS.NUM_CLASSES', num_classes_dict[i],
                        ]
            cfgs[i] = setup(config_paths[i], opts)
            cfgs[i].defrost()
            cfgs[i].MODEL.BACKBONE.PRETRAIN = False
            models[i] = DefaultTrainer.build_model(cfgs[i])
            models[i].eval()
            Checkpointer(models[i]).load(cfgs[i].MODEL.WEIGHTS)
    elif dataset == 'multi':
        views = 6
        config_paths = {}
        for i in range(1, views + 1):
            if i not in cam_list:
                continue
            config_paths[i] = './configs/Multisplit' + \
                str(i) + '/' + model_type + '.yml'
        # 日志
        logger = setup_logger('./logs/multi/train_log.txt')
        # 构建多模型字典
        models = {}
        # 构建多配置字典
        cfgs = {}
        num_classes_dict = {
            1: 350,
            2: 350,
            3: 350,
            4: 350,
            5: 350,
            6: 350,
            7: 350
        }
        for i in range(1, views + 1):
            if i not in cam_list:
                continue
            if is_origin:
                opts = ['MODEL.WEIGHTS', './logs/multiviewx2/' +
                        model_type + '/model_best.pth',
                        'MODEL.DEVICE', '\"cuda:1\"',
                        'MODEL.HEADS.NUM_CLASSES', num_classes_dict[i],
                        ]
            else:
                opts = ['MODEL.WEIGHTS', './logs/multisplit' +
                        str(i) + '/' + model_type + '/model_best.pth',
                        'MODEL.DEVICE', '\"cuda:1\"',
                        'MODEL.HEADS.NUM_CLASSES', num_classes_dict[i],
                        ]
            cfgs[i] = setup(config_paths[i], opts)
            cfgs[i].defrost()
            cfgs[i].MODEL.BACKBONE.PRETRAIN = False
            models[i] = DefaultTrainer.build_model(cfgs[i])
            models[i].eval()
            Checkpointer(models[i]).load(cfgs[i].MODEL.WEIGHTS)
    else:
        raise ValueError('未记录的数据集')

    return models, cfgs


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    dataset = 'multi'
    is_origin_model = 0
    eval_mode = 0  # 仅在orange_demo启用，设置为1
    name = '0236'  # 仅在orange_demo下有意义
    cam_num = 2
    if len(sys.argv) > 1:
        cam_num = int(sys.argv[1])    
    if dataset == 'wild':
        cam_all = [1, 6, 2, 3, 5, 7, 4]
        cam_list = cam_all[-cam_num:]
    elif dataset == 'multi':
        cam_all = [2, 6, 5, 4, 3, 1]
        cam_list = cam_all[-cam_num:]
    models, cfgs = load_models(dataset=dataset, is_origin=is_origin_model, cam_list=cam_list)
    if dataset == 'wild':
        root = 'datasets/Wildtrack_eval_rerank'
        n_cam = 7
        cfg = cfgs[cam_list[0]]
    elif dataset == 'multi':
        root = 'datasets/MultiviewX_eval'
        n_cam = 6
        cfg = cfgs[cam_list[0]]
    if eval_mode:
        root = 'datasets/Orange_demo_wild/mv'
    data = prepare_data(name=dataset, root='datasets', eval_mode=eval_mode, subname=name)
    query_dataloader = create_query_dataloader(
        data.query, cfg=cfg, root=root, eval_mode=eval_mode, name=name)
    gallery_dataloader_dict = create_gallery_dataloader(
        data.gallery, cfg=cfg, num_cam=n_cam, root=root, eval_mode=eval_mode, name=name)
    query_features, query_targets, query_cameras = process_query(
        models, query_dataloader)
    gallery_features, gallery_targets = process_gallery(
        models, gallery_dataloader_dict, n_cam)
    dist = predict(query_features, query_cameras, gallery_features, cfg)
    # array_save_path = 'orange_demo/mv_video/0236_t0575to1995/dist'
    # import numpy as np
    # # dist_flat = dist.flatten()
    # for idx, sub_dist in enumerate(dist[0]):
    #     np.save(array_save_path + str(idx) + '.npy', sub_dist)

    matches_by_distance, matches_by_vote, matches_ori = evaluate(
        dist, query_targets, gallery_targets, query_cameras, max_rank=50)
    save_path = 'mtl/logs_p/' + dataset + '/mv_eval/' + f'{str(cam_list)}_results.txt'
    original_output = sys.stdout
    with open(save_path, 'w') as f:
        sys.stdout = f
        print('by distance')
        calculate_score(matches_by_distance, query_cameras)
        print('by vote')
        calculate_score(matches_by_vote, query_cameras)
        print('ori')
        calculate_score(matches_ori, query_cameras)
        print('---------')
        print('starting evaluate by group')
        print('by distance')
        calculate_score_by_group(matches_by_distance)
        print('by vote')
        calculate_score_by_group(matches_by_vote)
        print('before end')
        # result format: Dict:{rank k, map, minp, metric}
        sys.stdout = original_output
