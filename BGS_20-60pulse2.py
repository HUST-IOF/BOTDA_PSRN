import numpy as np
import matplotlib.pyplot as plt
# from BGSfunction import BGSfunction
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import savemat
import torch
import math
import h5py

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# del tensor  # 释放张量
torch.cuda.empty_cache()

# 仿真参数设置
point = 36
print(f'扫频步长：{int((10.92e3 - 10.78e3) / (point - 1))}MHz')
deltaZ = 0.1  # 计算空间精度
deltaT = deltaZ / (2 * 10**8)  # 对应的时间精度
pulse_values = [60, 50, 40, 30, 20]  # 定义脉宽的循环范围
num = 1  # 光纤段数量
x = 10000  # 每一整段光纤长度
BGS3 = torch.zeros(num*len(pulse_values), point, x)  # 初始化大矩阵，形状 (num, 71, 1000)
BFS3 = torch.zeros(num*len(pulse_values), x)  # 存储 BFS 值
SW3 = torch.zeros(num*len(pulse_values), x)  # 存储 SW 值
Intensity3 = torch.zeros(num*len(pulse_values), x)  # 存储 Intensity 值
for pulse_idx, pulse in enumerate(pulse_values):
    np.random.seed(42)
    print(f'当前脉宽: {pulse}ns')
    pulseWidth = pulse*1e-9  # 脉冲宽度

    # 生成光纤段长度
    sectionLength = np.random.randint(5, 51, 800000)  # 每段光纤长度
    fiberLength = np.cumsum(sectionLength)  # 光纤总长度

    # 随机参数
    BFS = (10.89 - 10.81) * np.random.rand(len(sectionLength)) + 10.81  # BFS范围
    BFS *= 1e9  # 转换为 Hz
    SW = (35 - 25) * np.random.rand(len(sectionLength)) + 25  # 谱宽范围
    SW *= 1e6  # 转换为 Hz
    Intensity = (1 - 0.8) * np.random.rand(len(sectionLength)) + 0.8  # 归一化强度范围

    # 初始化数组
    BFS1 = np.zeros(fiberLength[-1])
    SW1 = np.zeros(fiberLength[-1])
    Intensity1 = np.zeros(fiberLength[-1])

    # 设置光纤段参数
    BFS1[:fiberLength[0]] = BFS[0]
    SW1[:fiberLength[0]] = SW[0]
    Intensity1[:fiberLength[0]] = Intensity[0]

    for i in range(1, len(sectionLength)):
        BFS1[fiberLength[i-1]:fiberLength[i]] = BFS[i]
        SW1[fiberLength[i-1]:fiberLength[i]] = SW[i]
        Intensity1[fiberLength[i-1]:fiberLength[i]] = Intensity[i]


    for i in range(num):
        print(i, end=' ')
        BFS2 = torch.tensor(BFS1[i * x:(i + 1) * x]).to(device)
        # print(BFS2.shape)
        SW2 = torch.tensor(SW1[i * x:(i + 1) * x]).to(device)
        Intensity2 = torch.tensor(Intensity1[i * x:(i + 1) * x]).to(device)
        # 创建时间数组 T
        T = torch.arange(pulseWidth, 0, -2 * deltaT, device=device)
        fiberLength = x
        step = int((10920 - 10780) / (point - 1)) * 1e6  # 总带宽 140e6，由 point 决定步长
        sweepFreq = torch.arange(10.78e9, 10.92e9 + 1, step, device=device)
        sweepFreq = sweepFreq.view(-1, 1)  # 现在形状为 (num_freq, 1)，准备进行广播
        # tau_min = BFS_expan.unsqueeze(1) ** 2 - sweepFreq ** 2 - 1j * sweepFreq * SW_expan.unsqueeze(1)
        tau = 1j * torch.pi * (
                BFS2 ** 2 - sweepFreq ** 2 - 1j * sweepFreq * SW2) / sweepFreq
        tau_conj = tau.conj()  # 共轭的 tau,形状将为 (batch_size, num_freq, width)
        # 逐元素相乘（广播机制）
        tau_outer = T.view(1, -1, 1) * tau_conj.unsqueeze(1)  # 形状 (f, n, w)
        Gain = Intensity2 * torch.real((1 - torch.exp(-tau_outer)) / tau_conj.unsqueeze(1))
        offsets = torch.arange(0, fiberLength - pulse, device=device)  # 计算所有偏移量 (n - len(T)) 对应的对角线
        # diags = [torch.sum(torch.diagonal(Gain, offset=o, dim1=1, dim2=2), dim=1) for o in offsets]
        # BGS = torch.stack(diags, dim=-1)
        rows = torch.arange(pulse, device=device).unsqueeze(1)  # 形状 (20, 1)
        cols = offsets.unsqueeze(0) + rows  # 广播至 (20, 780)
        # 提取所有对角线元素，形状变为 (64, 20, 780)
        selected = Gain[:, rows, cols]
        # 沿脉冲维度求和，得到形状 (64, 780)
        gainSignal = selected.sum(dim=1)
        # print(gainSignal.shape)(71,10000)(71,9800)
        # 计算在大矩阵中的索引
        idx = pulse_idx * num + i  # 当前 BGS 在大矩阵中的位置
        BGS3[idx, :, 100:-100] = gainSignal[:,100:-100+pulse]
        # noise_scale = (torch.rand(1) * 4.5 + 0.5) / 1000
        # noise = torch.randn(71, 540) * noise_scale    # 生成噪声
        # BGS3[i] += noise
        BFS3[idx, 100:-100] = BFS2[100:-100] # 记录 BFS 值
        SW3[idx, 100:-100] = SW2[100:-100]  # 记录 BFS 值
        Intensity3[idx,100:-100] = Intensity2[100:-100]  # 记录 BFS 值

        # 删除不再需要的变量
        del BFS2, SW2, Intensity2, tau, tau_conj, tau_outer, Gain, offsets,cols, selected, gainSignal
        torch.cuda.empty_cache()  # 清空 GPU 缓存
    print(f'BGS3形状:{BGS3.shape}')
    print(f'BFS3形状：{BFS3.shape}')


# np.save(f'./data/fangzhen/20250222_20-60ns_BGS.npy',BGS3)
# np.save(f'./data/fangzhen/20250222_20-60ns_ns_BFS.npy',BFS3)
# np.save(f'./data/fangzhen/20250222_20-60ns_ns_SW.npy',SW3)
# np.save(f'./data/fangzhen/20250222_20-60ns_ns_Intensity.npy',Intensity3)
data_dict = {
    'BFS': BFS3,
    'SW': SW3,
    'Intensity': Intensity3,
    'BGS': BGS3
}
# savemat(f'./matlabs/fangzhen/20250222_20-60ns_data_generation.mat', data_dict)
torch.save(data_dict, f'./matlabs/fangzhen/BGS_20-60ns_5zu_{point}point.pt')  # 保存为 PyTorch 文件
