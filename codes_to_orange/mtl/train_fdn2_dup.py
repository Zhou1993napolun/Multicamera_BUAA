from mv_eval import load_models, prepare_data, create_train_dataloader
from fastreid.engine import default_argument_parser
from modeling.mmoe import MMOE
from modeling.fdn2 import FDN, FrobeniusLoss
import torch
# import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
# import pickle
# from torch.utils.tensorboard import SummaryWriter
import logging
import time
import datetime
import os
import sys
import random


def exception_handler(type, value, traceback):
    # 将异常信息记录到日志
    logging.error("Uncaught exception",
                  exc_info=(type, value, traceback))


def model_train(num_cam, epochs, model_dict, mmoe, fdn, alpha, train_dataloader, optimizer, scheduler, save_path, num_classes_dict, device, optim_type):
    for epoch in range(epochs):
        if optim_type == 0:
            scheduler.step()
        start_time = time.time()
        logging.info('Epoch {}'.format(epoch))
        total_epoch_loss_dl_b21 = 0
        total_epoch_loss_dl_b22 = 0
        total_epoch_loss_dl_b31 = 0
        total_epoch_loss_dl_b32 = 0
        total_epoch_loss_dl_b33 = 0
        total_epoch_loss = 0
        total_epoch_loss_b1 = 0
        total_epoch_loss_b2 = 0
        total_epoch_loss_b21 = 0
        total_epoch_loss_b22 = 0
        total_epoch_loss_b3 = 0
        total_epoch_loss_b31 = 0
        total_epoch_loss_b32 = 0
        total_epoch_loss_b33 = 0

        for i, datas in enumerate(tqdm(train_dataloader)):
            # 用以寻找错误数据
            # if i < 2203:
            #     continue
            feas = {}  # 构建 key in [1:num_cam+1], shape为8*[1,256]的字典
            target_dict = {}
            camid_list = []
            # 下面考虑新增一个字典，用于对输入网络的pid，按顺序重新设置id
            # 存在问题，如果batch的分布和原始模型训练不一致，那么可能会学习错误信息
            for data in datas:
                input, _, pid, cam = data
                camid = int(cam.squeeze())
                if pid > num_classes_dict[camid]:
                    logging.warning("for model{}, training data target {} surpasses its cls {}".format(
                        camid, pid, num_classes_dict[camid]))
                    # 考虑在这里对每一个相机视角下的pid重新排序，以验证pid是否会一直存在超过分类数的情况出现
                    continue
                model = model_dict[camid]
                # device = model.device
                # shape[1,3,384,128]
                img = input.to(device)
                target = pid.to(device)
                features, target = model.forward_mmoe(img, target)
                feas[camid] = features
                target_dict[camid] = target
                camid_list.append(camid)
            if not feas:
                logging.warning("data {} is not used".format(i))
                continue
            # cur_cams = len(feas)
            # 构建 key in [0:8], shape为[num_cam, hidden_size]的空字典，对应位置填充feas值
            feas_total = []
            for i in range(num_cam):
                feas_total.append(torch.zeros(8, 256).to(device))
            for i, ft in feas.items():
                feas_total[int(i) - 1] = torch.cat(ft, dim=0)
            # num_cam * [8, 256] -> [8, 256, num_cam]
            stacked_feas = torch.stack(feas_total, dim=-1)
            reshaped_feas = stacked_feas.view(8, -1)  # [8, num_cam*256]
            feas_out_list = mmoe(reshaped_feas)
            feas_out_list, features_b2, features_b3 = fdn(feas_out_list)
            losses_dict = {}
            dl_b21 = floss(features_b2[0], features_b2[2])
            dl_b22 = floss(features_b2[1], features_b2[3])
            dl_b31 = floss(features_b3[0], features_b3[3])
            dl_b32 = floss(features_b3[1], features_b3[4])
            dl_b33 = floss(features_b3[2], features_b3[5])
            # 这里只考虑了能够拍摄到目标的相机，没有对未拍摄到目标的相机计算损失
            for camid in camid_list:
                # 希望输入维度是8*dict(num_keys=3,value_shape[1,256])
                loss_dict = model_dict[camid].backward_mmoe(
                    feas_out_list, target_dict[camid])
                losses = sum(loss_dict.values())
                losses_dict[camid] = losses + alpha * (dl_b21 + dl_b22 + dl_b31 + dl_b32 + dl_b33)
                total_epoch_loss_dl_b21 += alpha * dl_b21
                total_epoch_loss_dl_b22 += alpha * dl_b22
                total_epoch_loss_dl_b31 += alpha * dl_b31
                total_epoch_loss_dl_b32 += alpha * dl_b32
                total_epoch_loss_dl_b33 += alpha * dl_b33
                total_epoch_loss_b1 += loss_dict['loss_cls_b1']
                total_epoch_loss_b2 += loss_dict['loss_cls_b2']
                total_epoch_loss_b21 += loss_dict['loss_cls_b21']
                total_epoch_loss_b22 += loss_dict['loss_cls_b22']
                total_epoch_loss_b3 += loss_dict['loss_cls_b3']
                total_epoch_loss_b31 += loss_dict['loss_cls_b31']
                total_epoch_loss_b32 += loss_dict['loss_cls_b32']
                total_epoch_loss_b33 += loss_dict['loss_cls_b33']
            total_loss = sum(losses_dict.values())
            total_epoch_loss += total_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        end_time = time.time()
        elapsed_time = end_time - start_time  # 计算耗时
        logging.info('time comsumed: {}'.format(elapsed_time))
        model_path = save_path + f"/fdn_model_{epoch}.pth"  # 每个模型保存到不同的文件
        torch.save(fdn.state_dict(), model_path)
        logging.info('total loss: {}'.format(total_epoch_loss))
        logging.info('loss dl b21: {}'.format(total_epoch_loss_dl_b21))
        logging.info('loss dl b22: {}'.format(total_epoch_loss_dl_b22))
        logging.info('loss dl b31: {}'.format(total_epoch_loss_dl_b31))
        logging.info('loss dl b32: {}'.format(total_epoch_loss_dl_b32))
        logging.info('loss dl b33: {}'.format(total_epoch_loss_dl_b33))
        logging.info('loss b1: {}'.format(total_epoch_loss_b1))
        logging.info('loss b2: {}'.format(total_epoch_loss_b2))
        logging.info('loss b21: {}'.format(total_epoch_loss_b21))
        logging.info('loss b22: {}'.format(total_epoch_loss_b22))
        logging.info('loss b3: {}'.format(total_epoch_loss_b3))
        logging.info('loss b31: {}'.format(total_epoch_loss_b31))
        logging.info('loss b32: {}'.format(total_epoch_loss_b32))
        logging.info('loss b33: {}'.format(total_epoch_loss_b33))
        logging.info('model is saved')
        logging.info('Epoch {} finished'.format(epoch))


if __name__ == "__main__":
    # 设置随机数种子
    seed = 1121  # 你可以选择任何种子值
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    # 设置异常处理器
    sys.excepthook = exception_handler
    # 获取当前时间
    cur_time = datetime.datetime.now()
    cur_time = cur_time.strftime("%Y-%m-%d %H:%M:%S")
    # 训练轮次
    epochs = 15
    # 优化器参数
    learning_rate = 0.0001
    # 记录日志
    dataname = 'multi'
    device = 'cuda:0'
    optim_type = 1
    # alpha = 4000000  # 1000000
    alpha = 4000000
    mmoe_name = 'mmoe_reranked_seed24_epoch100_lr0.0001/'
    mmoe_epoch = 40
    if dataname == 'wild':
        root = 'datasets/Wildtrack_eval_rerank'
        mmoe_path = 'mtl/saves/wild/' + mmoe_name
        n_cam = 7
    elif dataname == 'multi':
        root = 'datasets/MultiviewX_eval'
        mmoe_path = 'mtl/saves/multi/' + mmoe_name
        n_cam = 6
    filename = os.path.join('mtl/logs/', dataname, mmoe_name, f'fdn2_e{epochs}_lr{learning_rate}_alpha{alpha}_seed{seed}_optim{optim_type}')
    if not os.path.exists(filename):
        os.makedirs(filename)
    logging.basicConfig(filename=filename + '/train.log', level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')
    logging.info('training started at {}'.format(cur_time))
    logging.info('device: {}'.format(device))
    logging.info('correspond mmoe epoch: {}'.format(mmoe_epoch))
    # create dataloader
    args = default_argument_parser().parse_args()
    models, cfgs = load_models(dataname)
    for camid, m in models.items():
        models[camid] = m.to(device)
    torch.cuda.empty_cache()
    for _, model in models.items():
        for param in model.parameters():
            param.requires_grad = False
    mmoe_model = MMOE(num_experts=8, input_dim=n_cam * 256, output_dim=32, num_tasks=8,
                      expert_hidden_dim=2048, gate_hidden_dim=2048).to(device)
    checkpoint_path = os.path.join(mmoe_path, f'mmoe_model_{mmoe_epoch-1}.pth')
    checkpoint = torch.load(checkpoint_path)
    mmoe_model.load_state_dict(checkpoint)
    mmoe_model.eval()
    for param in mmoe_model.parameters():
        param.requires_grad = False
    data = prepare_data(name=dataname, root='datasets')
    train_dataloader = create_train_dataloader(
        data.train, cfg=cfgs[1], root=root)
    # dict_names = ['cls_outputs', 'pred_class_logits', 'features']

    #构建FDN模型
    fdn_model = FDN().to(device)

    # 构建优化器
    floss = FrobeniusLoss()
    scheduler = None
    if optim_type == 0:
        optimizer = optim.SGD(fdn_model.parameters(), lr=learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate / 10)
    elif optim_type == 1:
        optimizer = optim.Adam(fdn_model.parameters(), lr=learning_rate)

    save_path = os.path.join(mmoe_path, f'fdn2_e{epochs}_lr{learning_rate}_alpha{alpha}_seed{seed}_optim{optim_type}')
    # save_path = 'mtl/saves/' + 'reranked_seed' + str(seed) + '_epoch' + str(epochs) + '_hl' + str(nums_hl) + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if dataname == 'wild':
        num_classes_dict = {
            1: 297,
            2: 297,
            3: 297,
            4: 297,
            5: 297,
            6: 297,
            7: 297
        }
    elif dataname == 'multi':
        num_classes_dict = {
            1: 350,
            2: 350,
            3: 350,
            4: 350,
            5: 350,
            6: 350,
            7: 350
        }
    torch.cuda.empty_cache()
    model_train(n_cam, epochs, models, mmoe_model, fdn_model, alpha, train_dataloader,
                optimizer, scheduler, save_path, num_classes_dict, device, optim_type)
