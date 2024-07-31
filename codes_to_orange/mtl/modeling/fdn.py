# import sys
# sys.path.append('.')
import torch
import torch.nn as nn
import torch.nn.functional as F
from mtl.modeling.mmoe import Gate, Expert


class FrobeniusLoss(nn.Module):
    def __init__(self):
        super(FrobeniusLoss, self).__init__()

    def forward(self, f1, f2):
        f1_l2_norm = torch.norm(f1, p=2, dim=1, keepdim=True).detach()
        f1_l2 = f1.div(f1_l2_norm.expand_as(f1) + 1e-9)

        f2_l2_norm = torch.norm(f2, p=2, dim=1, keepdim=True).detach()
        f2_l2 = f2.div(f2_l2_norm.expand_as(f2) + 1e-9)

        loss = torch.mean((f1_l2.t().mm(f2_l2)).pow(2))
        return loss


class FDN(nn.Module):
    def __init__(self, num_shared_experts_b2=2, num_shared_experts_b3=3, num_subtasks=8, input_dim=256, output_dim=256, hidden_units=2048):
        super(FDN, self).__init__()
        self.num_shared_experts = num_shared_experts_b2 + num_shared_experts_b3
        self.num_shared_experts_b2 = num_shared_experts_b2
        self.num_shared_experts_b3 = num_shared_experts_b3
        self.num_subtasks = num_subtasks
        self.num_features = input_dim
        self.output_dim = output_dim
        self.hidden_units = hidden_units

        # 定义共享expert层
        self.shared_expert_layer_b2 = nn.ModuleList([Expert(
            self.num_shared_experts_b2 * self.num_features, self.hidden_units) for _ in range(self.num_shared_experts_b2)])
        self.shared_expert_layer_b3 = nn.ModuleList([Expert(
            self.num_shared_experts_b3 * self.num_features, self.hidden_units) for _ in range(self.num_shared_experts_b3)])

        # 定义共享gate层
        self.gate_shared_layers_b2 = nn.ModuleList(
            [Gate(self.num_shared_experts_b2 * self.num_features, num_shared_experts_b2) for _ in range(self.num_shared_experts_b2)])
        self.gate_shared_layers_b3 = nn.ModuleList(
            [Gate(self.num_shared_experts_b3 * self.num_features, num_shared_experts_b3) for _ in range(self.num_shared_experts_b3)])

        # 定义任务专属的expert层
        self.task_expert_layers = nn.ModuleList(
            [Expert(self.num_features, self.hidden_units) for _ in range(self.num_subtasks)])

        # 定义任务专属gate层
        self.gate_expert_layers = nn.ModuleList(
            [Gate(self.num_features, 1) for _ in range(self.num_subtasks)])

        # 定义task层
        self.task_layers = nn.ModuleList(
            [nn.Linear(self.hidden_units, self.output_dim) for _ in range(self.num_subtasks)])

    def forward(self, x):
        # x的大小为8*(1, num_features)
        # 特征顺序: b1, b2, b21, b22, b3, b31, b32, b33
        x_b2 = torch.cat(x[2:4], dim=1)
        x_b3 = torch.cat(x[5:8], dim=1)

        # 任务专属的expert层的输出
        task_expert_outputs = [expert_layer(sub_x) for sub_x, expert_layer in zip(x, self.task_expert_layers)]  # num_task_experts * (1, hidden_units)
        # 任务专属gate层的输出
        gate_expert_outputs = [self.gate_expert_layers[i](x[i]) for i in range(self.num_subtasks)]
        task_expert_outputs = [gate * sub_output for gate, sub_output in zip(
            gate_expert_outputs, task_expert_outputs)]

        # 共享expert层的输出
        shared_expert_outputs_b2 = [expert(x_b2) for expert in self.shared_expert_layer_b2]  # (hidden_units, num_shared_experts_b2)

        shared_expert_outputs_b3 = [expert(x_b3) for expert in self.shared_expert_layer_b3]  # (hidden_units, num_shared_experts_b3)

        # 任务共享gate层的输出
        gate_shared_outputs_b2 = [gate(x_b2) for gate in self.gate_shared_layers_b2]
        task_shared_outputs_b2 = []
        for task_idx in range(self.num_shared_experts_b2):
            weighted_shared_outputs_b2 = [
                gate_shared_outputs_b2[i][:, task_idx, None] *
                shared_expert_outputs_b2[i] for i in range(self.num_shared_experts_b2)]
            task_shared_output_b2 = sum(weighted_shared_outputs_b2)
            task_shared_outputs_b2.append(task_shared_output_b2)
            task_expert_outputs[2 + task_idx] += task_shared_output_b2

        gate_shared_outputs_b3 = [gate(x_b3) for gate in self.gate_shared_layers_b3]
        task_shared_outputs_b3 = []
        for task_idx in range(self.num_shared_experts_b3):
            weighted_shared_outputs_b3 = [
                gate_shared_outputs_b3[i][:, task_idx, None] *
                shared_expert_outputs_b3[i] for i in range(self.num_shared_experts_b3)]
            task_shared_output_b3 = sum(weighted_shared_outputs_b3)
            task_shared_outputs_b3.append(task_shared_output_b3)
            task_expert_outputs[5 + task_idx] += task_shared_output_b3

        task_outputs = [task_layer(task_expert_output) for task_layer, task_expert_output in zip(self.task_layers, task_expert_outputs)]

        return task_outputs, shared_expert_outputs_b2, shared_expert_outputs_b3


inputs = [torch.randn(1, 256) for _ in range(8)]
fdn = FDN()
fdn(inputs)