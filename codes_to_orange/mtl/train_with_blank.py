from mv_eval import load_models, setup, prepare_data, create_train_dataloader
from fastreid.engine import DefaultTrainer, default_argument_parser
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import pickle
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


class FC(nn.Module):
    def __init__(self, num_cams, num_layers, hidden_dim) -> None:
        super().__init__()
        self.cams = num_cams
        self.nums = num_layers
        self.layer1 = nn.Linear(num_cams, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, 1)

    def forward(self, x, cur_cams):
        scale = cur_cams / self.cams
        x = F.tanh(self.layer1(x))
        for _ in range(self.nums):
            x = F.tanh(self.layer2(x))
        x = self.layer3(x)
        return x / scale

# 定义专家模型


class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Expert, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        return self.fc(x)

# 定义门控模型


class Gate(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(Gate, self).__init__()
        self.fc = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        return torch.softmax(self.fc(x), dim=1)

# 定义MMoE模型


class MMoE(nn.Module):
    def __init__(self, input_dim, num_tasks, num_experts, expert_hidden_dim):
        super(MMoE, self).__init__()
        self.experts = nn.ModuleList(
            [Expert(input_dim, expert_hidden_dim) for _ in range(num_experts)])
        self.gates = nn.ModuleList(
            [Gate(input_dim, num_experts) for _ in range(num_tasks)])
        self.shared_output_layer = nn.Linear(
            expert_hidden_dim * num_experts, num_tasks)

    def forward(self, x):
        expert_outputs = [expert(x) for expert in self.experts]
        gate_outputs = [gate(x) for gate in self.gates]

        # 通过门控网络加权组合专家输出
        task_outputs = [torch.sum(gate.unsqueeze(2) * expert_output, dim=1)
                        for gate, expert_output in zip(gate_outputs, expert_outputs)]

        # 将任务输出传递到共享输出层
        shared_output = torch.cat(task_outputs, dim=1)
        final_output = self.shared_output_layer(shared_output)

        return final_output


def model_train(epochs, model_dict, fcs, train_dataloader, optimizers, save_path, num_classes_dict, device):
    for epoch in range(epochs):
        start_time = time.time()
        logging.info('Epoch {}'.format(epoch))
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
                    logging.warning("for model{}, training data target {} surpasses its cls {}".format(camid, pid, num_classes_dict[camid]))
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
            cur_cams = len(feas)
            # 构建 key in [0:8], shape为[num_cam, hidden_size]的空字典，对应位置填充feas值
            feas_total = {}
            for i in range(8):
                feas_total[i] = torch.zeros(7, 256).to(device)
            for i, ft in feas_total.items():
                for j, fea in feas.items():
                    # 需要验证一下fea的格式
                    ft[int(j) - 1] = fea[i]  # 维度不匹配，update时特征维度是[num_classes]
            feas_out_list = []
            for k, f in feas_total.items():
                # 应为 8 * [1, hidden_size]
                # input_shape[7,256], output_shape[1,256]
                feas_out = fcs[k](f.T, cur_cams).T
                feas_out_list.append(feas_out)
            losses_dict = {}
            # 这里只考虑了能够拍摄到目标的相机，没有对未拍摄到目标的相机计算损失
            for camid in camid_list:
                # 希望输入维度是8*dict(num_keys=3,value_shape[1,256])
                loss_dict = model_dict[camid].backward_mmoe(
                    feas_out_list, target_dict[camid])
                losses = sum(loss_dict.values())
                losses_dict[camid] = losses
            total_loss = sum(losses_dict.values())
            for _, optimizer in optimizers.items():
                optimizer.zero_grad()
            total_loss.backward()
            for _, optimizer in optimizers.items():
                optimizer.step()

        end_time = time.time()
        elapsed_time = end_time - start_time  # 计算耗时
        logging.info('time comsumed: {}'.format(elapsed_time))
        for camid, model in fcs.items():
            model_path = save_path + f"model_{camid}.pth"  # 每个模型保存到不同的文件
            torch.save(model.state_dict(), model_path)
            logging.info('model{} is saved'.format(camid))
        logging.info('Epoch {} finished'.format(epoch))


def model_train_with_target(epochs, model_dict, fcs, train_dataloader, optimizer, dict_names):
    for epoch in range(epochs):
        for _, datas in enumerate(tqdm(train_dataloader)):
            feas = {}  # 构建 key in [1:num_cam+1], shape为8*[1,256]的字典
            for data in datas:
                input, _, pid, cam = data
                camid = int(cam.squeeze())
                model = model_dict[camid]
                device = model.device
                # shape[1,3,384,128]
                img = input.to(device)
                target = pid.to(device)
                features_dict, target = model.forward_mmoe(img, target)
                feas[camid] = features_dict
            cur_cams = len(feas)
            # 构建 key in [0:8], shape为[num_cam, hidden_size]的空字典，对应位置填充feas值
            feas_total = {}
            for i in range(8):
                feas_total[i] = {}
                for name in dict_names:
                    feas_total[i][name] = torch.zeros(7, 256).to(device)
            for i, ft in feas_total.items():
                for j, fea in feas.items():
                    # 需要验证一下fea的格式
                    for name in dict_names:
                        # 维度不匹配，update时特征维度是[num_classes]
                        ft[name][int(j) - 1] = fea[i][name]
            # 出了个问题，在计算损失的时候，训练时模型会回传三个参数，以字典形式保存着。这三个参数我是否都要用mmoe去优化，待考虑
            feas_out_list = []
            for k, f in feas_total.items():
                # 应为 8 * [1, hidden_size]
                tmp_feas_out_dict = {}
                for name in dict_names:
                    # input_shape[7,256], output_shape[1,256]
                    tmp_feas_out_dict[name] = fcs[k][name](
                        f[name].T, cur_cams).T
                feas_out_list.append(tmp_feas_out_dict)
            # 希望输入维度是8*dict(num_keys=3,value_shape[1,256])
            loss_dict = model.backward_mmoe(feas_out_list, cur_cams)
            losses = sum(loss_dict.values())
            losses.backward()
            optimizer.step()



# class FC(nn.Module):
#     def __init__(self,num_cam) -> None:
#         super().__init__()
#         self.


# class MMoE(nn.Module):
#     def __init__(self, hidden_size, mmoe_hidden_dim, n_expert, num_task) -> None:
#         super().__init__()
#         self.experts = torch.nn.Parameter(torch.rand(hidden_size, mmoe_hidden_dim, n_expert), requires_grad=True)
#         self.experts.data.normal_(0, 1)
#         self.experts_bias = torch.nn.Parameter(torch.rand(mmoe_hidden_dim, n_expert), requires_grad=True)
#         # gates
#         self.gates = [torch.nn.Parameter(torch.rand(hidden_size, n_expert), requires_grad=True) for _ in range(num_task)]
#         for gate in self.gates:
#             gate.data.normal_(0, 1)
#         self.gates_bias = [torch.nn.Parameter(torch.rand(n_expert), requires_grad=True) for _ in range(num_task)]

#     def forward(self):
#         pass
if __name__ == "__main__":
    # 设置随机数种子
    seed = 24  # 你可以选择任何种子值
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
    epochs = 10
    # 记录日志
    filename = 'mtl/logs/wild/' + 'reranked_seed' + str(seed) + '_epoch' + str(epochs)
    if not os.path.exists(filename):
        os.makedirs(filename)
    logging.basicConfig(filename=filename + '/train.log', level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')
    # 创建tensorboard
    # writer = SummaryWriter()
    # logging.info('epoch 30, correctly change cuda')
    logging.info('training started at {}'.format(cur_time))
    # create dataloader
    args = default_argument_parser().parse_args()
    models, cfgs = load_models('wild')
    device = 'cuda:1'
    for camid, m in models.items():
        models[camid] = m.to(device)
    torch.cuda.empty_cache()
    for _, model in models.items():
        for param in model.parameters():
            param.requires_grad = False
    # device = cfgs[1]['MODEL']['DEVICE']
    data = prepare_data(root='datasets')
    train_dataloader = create_train_dataloader(
        data.train, cfg=cfgs[1], root='datasets/Wildtrack_eval_rerank')
    dict_names = ['cls_outputs', 'pred_class_logits', 'features']
    fcs = {}
    # 考虑同时处理cls_outputs，pred_class_logits和features的情况
    # for i in range(8):
    #     fcs[i] = {}
    #     for name in dict_names:
    #         fcs[i][name] = FC(7, 3, 256).to(device)
    # paras = []
    # for _, nets in fcs.items():
    #     for _, net in nets.items():
    #         paras.extend(net.parameters())
    # 考虑只对前置256维feature进行处理
    for i in range(8):
        fcs[i] = FC(7, 3, 256).to(device)
    paras = {}
    for k, nets in fcs.items():
        paras[k] = nets.parameters()
    # optimizer = optim.Adam(fcs.parameters(), lr=0.001)
    optimizers = {}
    for i in range(8):
        optimizers[i] = optim.SGD(paras[i], lr=0.001)
    save_path = 'mtl/saves/' + 'reranked_seed' + str(seed) + '_epoch' + str(epochs) + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # num_classes_dict = {
    #     1: 286,
    #     2: 271,
    #     3: 258,
    #     4: 87,
    #     5: 204,
    #     6: 285,
    #     7: 98
    # }
    # rerank后共享相同的行人id集
    num_classes_dict = {
        1: 297,
        2: 297,
        3: 297,
        4: 297,
        5: 297,
        6: 297,
        7: 297
    }
    torch.cuda.empty_cache()
    model_train(epochs, models, fcs, train_dataloader, optimizers, save_path, num_classes_dict, device)
