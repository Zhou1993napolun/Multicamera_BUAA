import torch
import torch.nn as nn
import torch.optim as optim

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


# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 用于分类任务的示例损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
for epoch in range(num_epochs):
    for batch in data_loader:
        inputs, targets = batch

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()

# 在测试集上进行评估或推理
model.eval()
with torch.no_grad():
    for batch in test_loader:
        inputs, targets = batch
        outputs = model(inputs)
        # 在这里执行评估或推理操作
