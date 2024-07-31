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
    from mtl.utils.preprocessor import PreprocessQuery
    from mtl.utils.rank import evaluate_wild_mmoe
    from mtl.datasets.wildeval import WildEval
    from mtl.datasets.multieval import MultiEval
    from mtl.modeling.mmoe import MMOE
    from mtl.modeling.fdn2 import FDN
    from fastreid.data.transforms.build import build_transforms
    from fastreid.config import get_cfg
    # , default_setup, launch
    from fastreid.engine import default_argument_parser
    from fastreid.utils.compute_dist import build_dist
    # from torchvision import transforms
    from torch.utils.data import DataLoader  # , Dataset
    from tqdm import tqdm
    from mv_eval_partial import load_models
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


def prepare_data(name='wild', root='../datasets'):
    if name == 'wild':
        full_path = os.path.join(root, 'Wildtrack_eval_rerank')
        return WildEval(full_path)
    elif name == 'multi':
        full_path = os.path.join(root, 'MultiviewX_eval')
        return MultiEval(full_path)


def create_mmoe_dataloader(query, gallery, cfg, root='../datasets/Wildtrack_eval_rerank'):
    transforms = build_transforms(cfg, is_train=False)
    query_dataloader = DataLoader(
        PreprocessQuery(query=query, root=os.path.join(
            root, 'query'), transform=transforms), batch_size=1, num_workers=4, shuffle=False, pin_memory=False)
    gallery_dataloader = DataLoader(
        PreprocessQuery(query=gallery, root=os.path.join(
            root, 'gallery_mmoe'), transform=transforms), batch_size=1, num_workers=4, shuffle=False, pin_memory=False)
    return query_dataloader, gallery_dataloader


def process(model_dict, num_cams, dataloader, mmoe, fdn, device):
    features_total = []
    target_total = []
    # cam_total = []
    with torch.no_grad():
        for idx, datas in enumerate(tqdm(dataloader)):
            # 测试一下后续功能
            # if idx > 10:
            #     break
            feas = {}  # 构建 key in [1:num_cam+1], shape为8*[1,256]的字典
            # camid_list = []
            for data in datas:
                input, _, pid, cam = data
                camid = int(cam.squeeze())
                if camid not in model_dict.keys():
                    continue
                model = model_dict[camid]
                # shape[1,3,384,128]
                img = input.to(device)
                target = pid.to(device)
                features = model.inference_mmoe(img)
                feas[camid] = features
            if not feas:
                continue
                # camid_list.append(camid)
            # cur_cams = len(feas)
            # 构建 key in [0:8], shape为[num_cam, hidden_size]的空字典，对应位置填充feas值
            feas_total = []
            for i in range(num_cams):
                feas_total.append(torch.zeros(8, 256).to(device))
            for i, ft in feas.items():
                feas_total[int(i) - 1] = torch.cat(ft, dim=0)
            # num_cam * [8, 256] -> [8, 256, num_cam]
            stacked_feas = torch.stack(feas_total, dim=-1)
            reshaped_feas = stacked_feas.view(8, -1)  # [8, num_cam*256]
            feas_out_list = mmoe(reshaped_feas)
            feas_out_list, _, _ = fdn(feas_out_list)
            pred_feat = torch.cat(feas_out_list, dim=1)
            features_total.append(pred_feat)
            target_total.append(target)
    return torch.cat(features_total, dim=0), target_total


def predict(query_features, gallery_features, cfg):
    return build_dist(query_features, gallery_features, cfg.TEST.METRIC)


def predict_all(num_cams, query_features, gallery_features_dict, cfg):
    # 计算query和对应camera id下gallery feature下的维度
    dist = []
    for camid in range(1, num_cams + 1):
        # camid = str(camid)
        gallery_features = gallery_features_dict[camid]
        # query_features = query_feature.unsqueeze(0)
        cur_dist = build_dist(
            query_features, gallery_features, cfg.TEST.METRIC)
        dist.append(cur_dist)
    return dist


def evaluate(dist, query_targets, gallery_targets, max_rank):
    # 由dist得到的每组距离，计算每个距离前n个排名最高的值对应的index。
    # 找到每个距离对应的camid，在gallery_targets[camid]下对应的index中
    # 得到预测的pid。与query_targets对应组下对应序号下的pid对比判断正否
    return evaluate_wild_mmoe(dist, query_targets,
                              gallery_targets, max_rank=max_rank)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    device = 'cuda:3'
    dataname = 'wild'
    mmoename = 'mmoe_reranked_seed24_epoch100_lr0.0001'
    fdnname = 'fdn2_e80_lr0.0001_alpha4000000_seed1121_optim1'
    # fdnname = 'fdn2_e80_lr0.0001_alpha0_seed1121_optim1'
    mmoe_epoch = 60
    fdn_epoch = 2
    cam_num = 2
    if len(sys.argv) > 1:
        cam_num = int(sys.argv[1])
    if dataname == 'wild':
        cam_all = [1, 6, 2, 3, 5, 7, 4]
        cam_list = cam_all[-cam_num:]
    elif dataname == 'multi':
        cam_all = [2, 6, 5, 4, 3, 1]
        cam_list = cam_all[-cam_num:]
    models, cfgs = load_models(dataset=dataname, cam_list=cam_list)
    print('models loaded')
    if dataname == 'wild':
        n_cam = 7
        mmoe_path = 'mtl/saves/wild/' + mmoename
        fdn_path = os.path.join(mmoe_path, fdnname)
        log_path = 'mtl/logs_p/wild/' + mmoename
        log_path = os.path.join(log_path, fdnname)
        root = 'datasets/Wildtrack_eval_rerank'
        cfg = cfgs[cam_list[0]]
    elif dataname == 'multi':
        n_cam = 6
        mmoe_path = 'mtl/saves/multi/' + mmoename
        fdn_path = os.path.join(mmoe_path, fdnname)
        log_path = 'mtl/logs_p/multi/' + mmoename
        log_path = os.path.join(log_path, fdnname)
        root = 'datasets/MultiviewX_eval'
        cfg = cfgs[cam_list[0]]
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    mmoe_model = MMOE(num_experts=8, input_dim=n_cam * 256, output_dim=32, num_tasks=8,
                      expert_hidden_dim=2048, gate_hidden_dim=2048).to(device)
    checkpoint_path = os.path.join(mmoe_path, f'mmoe_model_{mmoe_epoch-1}.pth')
    checkpoint = torch.load(checkpoint_path)
    mmoe_model.load_state_dict(checkpoint)
    mmoe_model.eval()
    print('mmoe loaded')
    fdn_model = FDN().to(device)
    fdn_checkpoint_path = os.path.join(fdn_path, f'fdn_model_{fdn_epoch-1}.pth')
    fdn_checkpoint = torch.load(fdn_checkpoint_path)  # 可能不对，训完再看
    fdn_model.load_state_dict(fdn_checkpoint)
    fdn_model.eval()
    print('fdn model loaded')

    for k, m in models.items():
        models[k] = m.to(device)
    torch.cuda.empty_cache()
    data = prepare_data(dataname, root='datasets')
    print('data prepared')
    # 组评估
    print('evaluation by group started')
    query_dataloader, gallery_dataloader = create_mmoe_dataloader(
        data.query, data.gallery_mmoe, cfg=cfg, root=root)
    print('dataloader created')
    query_features, query_targets = process(
        models, n_cam, query_dataloader, mmoe_model, fdn_model, device)
    print('query features calculated')
    gallery_features, gallery_targets = process(
        models, n_cam, gallery_dataloader, mmoe_model, fdn_model, device)
    print('gallety features calculated')
    dist = predict(query_features, gallery_features, cfg)
    print('distance calculated')
    mAP, mINP, ranks = evaluate(
        dist, query_targets, gallery_targets, max_rank=50)
    print('map:', mAP)
    print('minp', mINP)
    print('rank', ranks)
    # result format: Dict:{rank k, map, minp, metric}
    with open(os.path.join(log_path, f'{str(cam_list)}eval_group_results_epoch{fdn_epoch}.txt'), 'w') as f:
        f.write(f'mAP: {mAP}\n')
        f.write(f'mINP: {mINP}\n')
        f.write(f'ranks: {ranks}\n')
    # 单张评估
    # print('evaluation by single image started')
    # from mtl.mv_eval import create_gallery_dataloader, process_gallery, predict
    # from mtl.utils.rank import evaluate_wild_mmoe_all, calculate_score_mmoe_with_gallery_all
    # query_dataloader, _ = create_mmoe_dataloader(
    #     data.query, data.gallery_mmoe, cfg=cfgs[1], root=root)
    # gallery_dataloader_dict = create_gallery_dataloader(
    #     data.gallery, cfg=cfgs[1], num_cam=n_cam, root=root)
    # print('dataloader created')
    # gallery_features, gallery_targets = process_gallery(
    #     models, gallery_dataloader_dict)
    # print('gallety features calculated')
    # query_features, query_targets = process(
    #     models, n_cam, query_dataloader, mmoe_model, fdn_model, device)
    # print('query features calculated')
    # dist = predict_all(n_cam, query_features, gallery_features, cfgs[1])
    # matches_by_distance, matches_by_vote, matches_ori = evaluate_wild_mmoe_all(
    #     dist, query_targets, gallery_targets, max_rank=50)

    # original_output = sys.stdout
    # with open(os.path.join(log_path, f'eval_single_results_epoch{fdn_epoch}.txt'), 'w') as f:
    #     sys.stdout = f
    #     print('by distance')
    #     calculate_score_mmoe_with_gallery_all(matches_by_distance, n_cam)
    #     print('by vote')
    #     calculate_score_mmoe_with_gallery_all(matches_by_vote, n_cam)
    #     # print('ori')
    #     # calculate_score_mmoe_with_gallery_all(matches_ori)
    #     print('before end')
    #     # result format: Dict:{rank k, map, minp, metric}
    #     sys.stdout = original_output
