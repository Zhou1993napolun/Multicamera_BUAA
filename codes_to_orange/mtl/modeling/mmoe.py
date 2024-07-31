import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Expert, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return F.relu(self.fc(x))


class Gate(nn.Module):
    def __init__(self, input_dim, num_tasks):
        super(Gate, self).__init__()
        self.fc = nn.Linear(input_dim, num_tasks)

    def forward(self, x):
        return F.softmax(self.fc(x), dim=1)


class MMOE(nn.Module):
    def __init__(self, num_experts, input_dim, output_dim, num_tasks,
                 expert_hidden_dim, gate_hidden_dim):
        super(MMOE, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList(
            [Expert(input_dim, expert_hidden_dim) for _ in range(num_experts)])
        self.gates = nn.ModuleList(
            [Gate(input_dim, num_tasks) for _ in range(num_experts)])
        self.output_layers = nn.ModuleList(
            [nn.Linear(expert_hidden_dim, output_dim) for _ in range(num_tasks)])

    def forward(self, x):
        expert_outputs = [expert(x) for expert in self.experts]
        gate_outputs = [gate(x) for gate in self.gates]

        task_outputs = []
        for task_idx in range(len(self.output_layers)):
            weighted_expert_outputs = [
                gate_outputs[i][:, task_idx, None] *
                expert_outputs[i] for i in range(self.num_experts)]
            task_output = sum(weighted_expert_outputs)
            task_outputs.append(self.output_layers[task_idx](task_output))

        task_tensor = torch.stack(task_outputs, dim=0).permute(1, 0, 2)  # 8*[8, 32] -> [8, 8, 32]
        reshaped_tast_tensor = task_tensor.reshape(8, -1)  # [8, 256]
        task_tensor_outputs = [reshaped_tast_tensor[i:i + 1, :] for i in range(reshaped_tast_tensor.shape[0])]

        return task_tensor_outputs


# 示例用法
# 定义模型参数
num_experts = 8
input_dim = 7 * 256
output_dim = 32
num_tasks = 8
expert_hidden_dim = 2048
gate_hidden_dim = 2048

# 创建MMOE模型
model = MMOE(num_experts, input_dim, output_dim,
             num_tasks, expert_hidden_dim, gate_hidden_dim)

# 随机初始化输入数据
batch_size = 8
inputs = torch.randn(batch_size, 7 * 256)

# 模型前向传播
outputs = model(inputs)

# 打印模型输出
for task_idx, output in enumerate(outputs):
    print(f"Task {task_idx + 1} output shape: {output.shape}")
