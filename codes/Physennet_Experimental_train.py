import numpy as np
import os
import torch
import math
from torch.utils import data
import time
from torch import nn,optim
from scipy.io import savemat
from torchinfo import summary
# from model2 import resnet34
from PhysenNet import resnet34

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())


class ShiyanDataset(data.Dataset):
    def __init__(self,fea_list, transform=None):
        self.transform = transform
        self.fea_list = fea_list.to(device)
        self.fea_len = fea_list.size(1)
    def __getitem__(self,idx):
        feature = self.fea_list[:,self.fea_len-j-800:self.fea_len-j]
        feature = feature/torch.amax(feature[:,:-pulse])
        feature = feature.float()
        return feature

    def __len__(self):
        dataset_len = 1
        return dataset_len

# BGS reconstruction from predicted BFS, SW, and Intensity
def BGS_Reconstruction(data,pulse):
    BFS=data[:,:,0,:]
    SW=data[:,:,1,:]
    Intensity=data[:,:,2,:]
    BFS = (torch.squeeze(BFS) * (10.89 - 10.81) + 10.81) * 1e9
    SW = (torch.squeeze(SW) * (35 - 25) + 25) * 1e6
    Intensity = torch.squeeze(Intensity) * (1 - 0.8) + 0.8
    # 仿真参数设置
    deltaZ = 0.1  # 空间精度
    deltaT = deltaZ / (2 * 10 ** 8)  # 时间精度
    pulseWidth = pulse * 1e-9
    step = int((10920 - 10780) / (frame - 1)) *1e6 # 总带宽 140e6，由 point 决定步长
    sweepFreq = torch.arange(10.78e9, 10.92e9 + 1, step, device=device)
    T = torch.arange(pulseWidth, 0, -2 * deltaT, device=device)
    fiberLength = 800
    sweepFreq = sweepFreq.view(-1, 1)  # 现在形状为 (num_freq, 1)，准备进行广播
    tau = 1j * torch.pi * (
            BFS ** 2 - sweepFreq ** 2 - 1j * sweepFreq * SW) / sweepFreq
    tau_conj = tau.conj()  # 共轭的 tau,形状将为 (batch_size, num_freq, width)
    # 逐元素相乘（广播机制）
    tau_outer = T.view(1, -1, 1) * tau_conj.unsqueeze(1)  # 形状 (f, n, w)
    Gain = Intensity * torch.real((1 - torch.exp(-tau_outer)) / tau_conj.unsqueeze(1))
    offsets = torch.arange(0, fiberLength-pulse, device=device)  # 计算所有偏移量 (n - len(T)) 对应的对角线
    rows = torch.arange(pulse, device=device).unsqueeze(1)
    cols = offsets.unsqueeze(0) + rows  #
    # 提取所有对角线元素
    selected = Gain[:, rows, cols]
    # 沿脉冲维度求和，
    BGS = selected.sum(dim=1)
    return BGS/torch.amax(BGS[:,:])

# Define total variation regularization loss
class TVLoss(nn.Module):
    def __init__(self, lambda_tv):
        super().__init__()
        self.lambda_tv = lambda_tv  # 正则化强度系数

    def forward(self, y_pred):
        y_pred = torch.squeeze(y_pred)
        # 计算相邻元素的绝对差之和
        tv_loss = torch.sum(torch.abs(y_pred[:, 1:] - y_pred[:, :-1]))
        return self.lambda_tv * tv_loss

frame = 71 # 71:2MHz;36:4MHz;18:8MHz
pulse = 40 #pulse_values = [60, 50, 40, 30, 20]
m = 1
tv_loss = TVLoss(lambda_tv=5e-6)
BGS = torch.tensor(np.load(f'../data/Experimental_BGS{pulse}ns{m}_{frame}p.npy', allow_pickle=True))
if pulse == 40:#The 40-ns BGS experimental data is 10 meters longer than the others.
    j = 100
else:j=0
isstride = True
if frame < 36:
    isstride = False
train_dataset = ShiyanDataset(BGS)
GT = []

print(f'脉宽:{pulse}ns,BGS形状{BGS.shape}')
print(f'训练数据集的大小:{len(train_dataset)},扫频步长:{int((10920 - 10780) / (frame - 1)) * 1e6},训练数据大小：{BGS.shape}')

# 创建数据loader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
model = resnet34(isstride=isstride).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()  # 首先创建 MSELoss 对象

save_dir = f'./'
# Initialize last layer parameters
with torch.no_grad():
    nn.init.constant_(model.conv_last.bias, 0.1)  # 初始化偏置
    nn.init.zeros_(model.conv_last.weight)  # 初始化权重


# run
start_time = time.perf_counter()
# Training loop
train_loss_list = []
train_mseloss_list = []
BGS_recon_list = []
BGS_pred_list = []
BGS_phy_list = []
BFS_list=[]
SW_list=[]
Intensity_list=[]
n_epoch = 2000
save_interval = 10
total_training = 0.0
# 记录初始学习率
prev_lr = optimizer.param_groups[0]['lr']
print(f"初始学习率: {prev_lr}")
train_start = time.time()
for epoch in range(n_epoch):
    # 训练
    model.train()

    running_loss = 0
    train_loss = 0.0
    train_mseloss = 0.0
    # print(f'周期:{epoch+1}')
    for input_data in train_loader:
        input_data = input_data.to(device)
        BGS_pred = model(input_data.unsqueeze(1))  # 通过一次网络的输出
        BGS_recon = BGS_Reconstruction(BGS_pred, pulse=pulse)
        input_data = torch.squeeze(input_data)
        loss_mse = loss_fn(BGS_recon[:,:], input_data[:,:-pulse])
        loss_tv = tv_loss(BGS_pred)
        loss = loss_mse + loss_tv
        # 反向传播和优化
        optimizer.zero_grad()  # 参数梯度清零，因为会累加
        loss.backward()  # loss反向传播
        optimizer.step()
        # 累计损失
        train_loss += loss.item()
        train_loss_list.append(train_loss)
        train_mseloss += loss_mse.item()
        train_mseloss_list.append(train_mseloss)
        BGS_pred_num = BGS_pred.detach().cpu().numpy()
        BGS_pred_list.append(BGS_pred_num)
        input_data = input_data.detach().cpu().numpy()
        BGS_recon_num = BGS_recon.detach().cpu().numpy()
        BGS_recon_list.append(BGS_recon_num)
        print(f'Epoch [{epoch + 1}/{n_epoch}] Loss: {loss:.8f} loss_mse: {loss_mse:.8f} loss_tv: {loss_tv:.8f}')
        # ================= 保存与日志 =================
        # 打印周期损失（避免每个step都打印）
        # print(f'Epoch [{epoch + 1}/{n_epoch}] Loss: {train_loss:.6f}')
        # 间隔保存模型
        if (epoch + 1) % 1000 == 0:
            save_path = os.path.join(save_dir, f"model_Experimental_{pulse}ns_{frame}p_{epoch + 1}epoch.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss_list,
            }, save_path)
            print(f"Checkpoint saved at {save_path}")
            data_dict = {
                'BGS_pre': BGS_pred_list,
                'BGS_raw':input_data,
                'BGS_recon': BGS_recon_list,
                'BGS_GT': GT,
                'TrainLoss': np.column_stack((train_loss_list, train_mseloss_list)),
            }
            import h5py
            with h5py.File(f'./matlabs/PSRN_Experimental_{pulse}ns{m}_{frame}p_BGS.h5', 'w') as f:
                for key, value in data_dict.items():
                    f.create_dataset(key, data=value)
            # savemat(f'./matlabs/shiyan/shiyan_{pulse}ns_BGS_predicted_{m}_{frame}p.mat', data_dict)
print(f"Total training time: {time.time() - train_start:.2f}s")
