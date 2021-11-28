# -*- File Info -*-
# @File      : train.py
# @Date&Time : 2021-11-06, 22:18:23
# @Project   : aiBenchmark
# @Author    : yoc
# @Email     : iyoc@foxmail.com
# @Software  : PyCharm - Razer Blade

import time
from tqdm import tqdm

import torch
import torchvision
import torchvision.transforms as T
import torch.utils.data as data

import torch.nn as nn
import torch.optim as optimizer

import model

print(" --------------- ResNet Benchmark --------------- ")

# 数据集路径
root_data = './dataset/cifar10'

# DataSet
train_set = torchvision.datasets.CIFAR10(root=root_data,
                                         train=True,
                                         transform=T.ToTensor(),
                                         download=False)
test_set = torchvision.datasets.CIFAR10(root=root_data,
                                        train=False,
                                        transform=T.ToTensor(),
                                        download=False)

# 数据集信息
train_data_size = len(train_set)
test_data_size = len(test_set)
print(f"数据集 {root_data[-7:]} | 训练集图片 {train_data_size} 张 | 测试集 {test_data_size} 张")

# DataLoader
train_loader = data.DataLoader(train_set, batch_size=128, shuffle=False)
test_loader = data.DataLoader(test_set, batch_size=128, shuffle=False)

"""计算设备"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(f"设备号 {device} | 设备名 {torch.cuda.get_device_name(device)}\n")
print(f"设备 -> {device}\n")

"""模型"""
model_run = model.resnet_152p
model_run = model_run.to(device)
# print("模型\n", list(model_run.named_parameters()))

"""创建损失函数"""
loss_func = nn.CrossEntropyLoss().to(device)

"""优化器"""
learning_rate = 1e-2  # 0.01 = 1 * (10)^(-2)
optimizer = optimizer.SGD(model_run.parameters(), lr=learning_rate)

"""训练"""
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 轮数
epoch = 3
# 计时时间
time_start = time.time()

for i in range(epoch):

    # print(f"EPOCH {i + 1} / {epoch}: ")

    # 训练
    model_run.train()
    # 计时
    time_epoch = 0
    # 进度条
    loop = tqdm(enumerate(train_loader), total=len(train_loader), unit='img')
    # for batch in train_loader:
    for index, (batch, targets) in loop:
        # 计时
        time_batch_start = time.time()

        # 获取数据和标签
        imgs = batch.to(device)
        targets = targets.to(device)

        # 得到推理结果
        outputs = model_run(imgs)
        # 计算batch层面的损失
        batch_loss = loss_func(outputs, targets)

        # 优化器优化
        optimizer.zero_grad()  # 优化器梯度清零
        batch_loss.backward()  # 反向传播求梯度
        optimizer.step()  # 优化模型参数

        # 计时
        time_batch_end = time.time()
        time_batch = time_batch_end - time_batch_start

        # 记录、打印
        total_train_step += 1  # 训练次数加一
        # if total_train_step % 100 == 0:
        #     print(f"训练了第 {total_train_step} 个batch，batch loss：{batch_loss.item()}")
        # print(f"训练了第 {total_train_step} 个batch，batch loss：{batch_loss.item()}，batch用时：{time_batch}")

        time_epoch += time_batch
        # 更新信息
        loop.set_description(f'Epoch [{i + 1}/{epoch}]')
        # loop.set_postfix(loss=batch_loss.item(), acc=running_train_acc)
        loop.set_postfix(loss=batch_loss.item(), time=time_batch)

    print("EPOCH: {}/{} 用时：{:.2f}分钟!({:.4f}s)".format(i + 1,
                                                      epoch,
                                                      time_epoch // 60 + (time_epoch % 60) / 60,
                                                      time_epoch))

    # 测试
    model_run.eval()  # nn.Module.eval()只对一些层如Dropout、BatchNorm等有作用，详见官方文档
    epoch_test_loss = 0
    epoch_accuracy = 0

    with torch.no_grad():  # 保证模型不会被调优

        for data in test_loader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)

            test_outputs = model_run(imgs)
            test_batch_loss = loss_func(test_outputs, targets)
            epoch_test_loss += test_batch_loss.item()  # .item()控制数据类型相同
            batch_accuracy = ((test_outputs.argmax(1) == targets).sum()).item()  # 具体详见草稿文件
            epoch_accuracy += batch_accuracy  # 累积该次epoch在测试集上的整体的预测正确个数

    print(f"epoch 为 {i + 1} 时模型在测试集上整体的 loss 为：{epoch_test_loss}")
    print(f"epoch 为 {i + 1} 时模型在测试集上整体的 accuracy 为：{epoch_accuracy / test_data_size}\n")

    total_test_step += 1

    # 保存每次 epoch 后的模型状态
    # torch.save(CifarModel.state_dict(), f"./18_saved_state_dict/CifarModel{i + 1}.pth")
    # print(f"第 {i+1} 轮模型已保存！")
