# Import required libraries
import numpy as np
import os
import torch
import math
from torch.utils import data
import time
from torch import nn, optim
from scipy.io import savemat
from torchinfo import summary
from PhysenNet import resnet34

# Set the computation device (GPU if available)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

# Define custom dataset with optional noise addition
class MyDataset(data.Dataset):
    def __init__(self, BGS, noise_scale, transform=None):
        self.transform = transform
        self.BGS = BGS.to(device)
        self.noise_scale = noise_scale

    def __getitem__(self, idx):
        # Normalize input, add Gaussian noise
        feature = self.BGS
        feature = feature / torch.amax(feature[:, :-pulse])
        noise = torch.randn(feature.shape) * self.noise_scale
        feature += noise.to(device)
        return feature.float()

    def __len__(self):
        return 1

# BGS reconstruction from predicted BFS, SW, and Intensity
def BGS_Reconstruction(data, pulse):
    # De-normalize predicted physical quantities
    BFS = data[:, :, 0, :]
    SW = data[:, :, 1, :]
    Intensity = data[:, :, 2, :]
    BFS = (torch.squeeze(BFS) * (10.89 - 10.81) + 10.81) * 1e9
    SW = (torch.squeeze(SW) * (35 - 25) + 25) * 1e6
    Intensity = torch.squeeze(Intensity) * (1 - 0.8) + 0.8

    deltaZ = 0.1
    deltaT = deltaZ / (2 * 1e8)
    pulseWidth = pulse * 1e-9
    step = int((10920 - 10780) / (point - 1)) * 1e6
    sweepFreq = torch.arange(10.78e9, 10.92e9 + 1, step, device=device)

    T = torch.arange(pulseWidth, 0, -2 * deltaT, device=device)
    fiberLength = int(BFS.shape[0])
    sweepFreq = sweepFreq.view(-1, 1)
    tau = 1j * torch.pi * (BFS ** 2 - sweepFreq ** 2 - 1j * sweepFreq * SW) / sweepFreq
    tau_conj = tau.conj()

    # Compute Brillouin gain
    tau_outer = T.view(1, -1, 1) * tau_conj.unsqueeze(1)
    Gain = Intensity * torch.real((1 - torch.exp(-tau_outer)) / tau_conj.unsqueeze(1))

    # Extract and sum diagonal lines (simulate spatial response)
    offsets = torch.arange(0, fiberLength - pulse, device=device)
    rows = torch.arange(pulse, device=device).unsqueeze(1)
    cols = offsets.unsqueeze(0) + rows
    selected = Gain[:, rows, cols]
    BGS = selected.sum(dim=1)
    return BGS / torch.amax(BGS)

# Define total variation regularization loss
class TVLoss(nn.Module):
    def __init__(self, lambda_tv):
        super().__init__()
        self.lambda_tv = lambda_tv

    def forward(self, y_pred):
        y_pred = torch.squeeze(y_pred)
        tv_loss = torch.sum(torch.abs(y_pred[:, 1:] - y_pred[:, :-1]))
        return self.lambda_tv * tv_loss

# ----------- Training Configuration -----------

# Pulse width and BGS parameters
pulse_values = [60, 50, 40, 30, 20]
pulse = 40
point = 71 # 71:2MHz;36:4MHz;18:8MHz
lambda_tv = 1e-6
noise_scale = 0.005
isstride = True
if point <36:
    isstride = False

# Load simulation dataset
data_dict = torch.load(f'./data/Simulation_BGS_20-60ns_5zu_{point}point.pt', weights_only=True)
BFS = data_dict['BFS']
SW = data_dict['SW']
Intensity = data_dict['Intensity']
BGS = data_dict['BGS']
save_dir = f'./'
# Normalize inputs
BFS = (BFS[:, 100:-100] / 1e9 - 10.81) / (10.89 - 10.81)
SW = (SW[:, 100:-100] / 1e6 - 25) / (35 - 25)
Intensity = (Intensity[:, 100:-100] - 0.8) / (1 - 0.8)
BGS = BGS[:, :, 100:-100]

idx = pulse_values.index(pulse)
start = 5000
end = 5600
train_BGS = BGS[idx, :, start:end]
GT = np.stack((BFS[idx, :], SW[idx, :], Intensity[idx, :]), axis=0)

# Create dataset and dataloader
train_dataset = MyDataset(train_BGS, noise_scale)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)

# Initialize model and optimizer
model = resnet34(isstride=isstride).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()
tv_loss = TVLoss(lambda_tv)

# Initialize last layer parameters
with torch.no_grad():
    nn.init.constant_(model.conv_last.bias, 0.1)
    nn.init.zeros_(model.conv_last.weight)

# ----------- Training Loop -----------

n_epoch = 2000
train_loss_list = []
train_mseloss_list = []
BGS_recon_list = []
BGS_pred_list = []
BFS_list = []
SW_list = []
Intensity_list = []

print(f'Pulse width: {pulse} ns')
print('Training dataset size:', len(train_dataset))
print(f"Initial learning rate: {optimizer.param_groups[0]['lr']}")

# Start training
train_start = time.time()
for epoch in range(n_epoch):
    model.train()
    train_loss = 0.0
    train_mseloss = 0.0

    for input_data in train_loader:
        input_data = input_data.to(device)
        BGS_pred = model(input_data.unsqueeze(1))  # Predict physical quantities
        BGS_recon = BGS_Reconstruction(BGS_pred, pulse)
        loss_mse = loss_fn(BGS_recon, input_data.squeeze()[:, :-pulse])
        loss_tv = tv_loss(BGS_pred)
        loss = loss_mse + loss_tv

        # Backpropagation and parameter update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record losses and results
        train_loss += loss.item()
        train_loss_list.append(train_loss)
        train_mseloss += loss_mse.item()
        train_mseloss_list.append(train_mseloss)
        BGS_pred_list.append(BGS_pred.detach().cpu().numpy())
        BGS_recon_list.append(BGS_recon.detach().cpu().numpy())

        print(f'Epoch [{epoch + 1}/{n_epoch}] Loss: {loss:.8f} loss_mse: {loss_mse:.8f} loss_tv: {loss_tv:.8f}')

        # Save model and all results every 1000 epochs
        if (epoch + 1) % 1000 == 0:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"model_Simulation_{pulse}ns_{point}p_{epoch + 1}epoch.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss_list,
            }, save_path)
            print(f"Checkpoint saved at {save_path}")

            # Save reconstruction results to HDF5
            import h5py
            data_all = {
                'BGS_clean': train_BGS.numpy(),
                'BGS_pre': BGS_pred_list,
                'BGS_raw': input_data.detach().cpu().numpy(),
                'BGS_recon': BGS_recon_list,
                'BGS_GT': GT,
                'TrainLoss': np.column_stack((train_loss_list, train_mseloss_list)),
            }
            with h5py.File(f'./matlabs/PSRN_Simulation_{pulse}ns_{point}p_BGS.h5', 'w') as f:
                for key, value in data_all.items():
                    f.create_dataset(key, data=value)

print(f"Total training time: {time.time() - train_start:.2f}s")
