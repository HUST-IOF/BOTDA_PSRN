import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat  # For MATLAB .mat file export
import torch  # Main computation library (GPU accelerated)
import math
import h5py  # For HDF5 file format support

# Set computation device (GPU if available)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Clear GPU cache to free memory
torch.cuda.empty_cache()

# =====================
# SIMULATION PARAMETERS
# =====================
point = 71  # Number of frequency sampling points. 71:2MHz;36:4MHz;18:8MHz
print(f'Sweep step: {int((10.92e3 - 10.78e3) / (point - 1))}MHz')  # Calculate frequency resolution
deltaZ = 0.1  # space resolution (meters)
deltaT = deltaZ / (2 * 10 ** 8)  # Time resolution derived from spatial resolution (speed of light in fiber)
pulse_values = [60, 50, 40, 30, 20]  # Pulse widths to simulate (ns)
num = 1  # Number of fiber segments to process per pulse width
x = 10000  # Length of each fiber segment (points)

# Initialize result tensors
BGS3 = torch.zeros(num * len(pulse_values), point, x)  # Brillouin Gain Spectra (3D tensor: [config, freq, position])
BFS3 = torch.zeros(num * len(pulse_values), x)  # Brillouin Frequency Shift
SW3 = torch.zeros(num * len(pulse_values), x)  # Spectral Width
Intensity3 = torch.zeros(num * len(pulse_values), x)  # Scattering Intensity

# Main simulation loop over pulse widths
for pulse_idx, pulse in enumerate(pulse_values):
    np.random.seed(42)  # Fixed seed for reproducible randomness
    print(f'Current pulse width: {pulse}ns')
    pulseWidth = pulse * 1e-9

    # ==============================
    # FIBER PROPERTY GENERATION
    # ==============================
    # Create random fiber segments (800,000 segments)
    sectionLength = np.random.randint(5, 51, 800000)  # Each segment 0.5-5m long
    fiberLength = np.cumsum(sectionLength)  # Cumulative length array

    # Generate random optical properties for each segment:
    BFS = (10.89 - 10.81) * np.random.rand(len(sectionLength)) + 10.81  # BFS in GHz (10.81-10.89 range)
    BFS *= 1e9  # Convert to Hz
    SW = (35 - 25) * np.random.rand(len(sectionLength)) + 25  # Spectral width in MHz (25-35 range)
    SW *= 1e6  # Convert to Hz
    Intensity = (1 - 0.8) * np.random.rand(len(sectionLength)) + 0.8  # Intensity (0.8-1.0 normalized)

    # Map segment properties to high-resolution fiber array
    BFS1 = np.zeros(fiberLength[-1])
    SW1 = np.zeros(fiberLength[-1])
    Intensity1 = np.zeros(fiberLength[-1])

    # Assign properties to first segment
    BFS1[:fiberLength[0]] = BFS[0]
    SW1[:fiberLength[0]] = SW[0]
    Intensity1[:fiberLength[0]] = Intensity[0]

    # Assign properties to subsequent segments
    for i in range(1, len(sectionLength)):
        BFS1[fiberLength[i - 1]:fiberLength[i]] = BFS[i]
        SW1[fiberLength[i - 1]:fiberLength[i]] = SW[i]
        Intensity1[fiberLength[i - 1]:fiberLength[i]] = Intensity[i]

    # ==================================
    # BGS CALCULATION FOR EACH SEGMENT
    # ==================================
    for i in range(num):
        print(i, end=' ')  # Progress indicator
        # Extract current fiber segment properties
        BFS2 = torch.tensor(BFS1[i * x:(i + 1) * x]).to(device)
        SW2 = torch.tensor(SW1[i * x:(i + 1) * x]).to(device)
        Intensity2 = torch.tensor(Intensity1[i * x:(i + 1) * x]).to(device)

        # Create time array for pulse (reverse order)
        T = torch.arange(pulseWidth, 0, -2 * deltaT, device=device)
        fiberLength = x  # Current segment length

        # Frequency sweep parameters (10.78-10.92 GHz range)
        step = int((10920 - 10780) / (point - 1)) * 1e6  # Frequency step in Hz
        sweepFreq = torch.arange(10.78e9, 10.92e9 + 1, step, device=device)
        sweepFreq = sweepFreq.view(-1, 1)  # Reshape for broadcasting [freq, 1]

        # ========================================
        # CORE BGS COMPUTATION (GPU ACCELERATED)
        # ========================================
        # Calculate complex tau parameter (Brillouin interaction term)
        tau = 1j * torch.pi * (
                BFS2 ** 2 - sweepFreq ** 2 - 1j * sweepFreq * SW2) / sweepFreq
        tau_conj = tau.conj()  # Complex conjugate

        # Compute outer product of time and tau_conj
        tau_outer = T.view(1, -1, 1) * tau_conj.unsqueeze(1)  # [freq, time, position]

        # Calculate gain
        Gain = Intensity2 * torch.real((1 - torch.exp(-tau_outer)) / tau_conj.unsqueeze(1))

        # ========================================
        # SIGNAL INTEGRATION (PULSE PROPAGATION)
        # ========================================
        offsets = torch.arange(0, fiberLength - pulse, device=device)  # Valid diagonal offsets

        # Prepare indices for diagonal summation (simulating pulse integration)
        rows = torch.arange(pulse, device=device).unsqueeze(1)  # Pulse duration indices
        cols = offsets.unsqueeze(0) + rows  # Position indices

        # Extract and sum diagonals (equivalent to moving pulse integration)
        selected = Gain[:, rows, cols]  # [freq, pulse_duration, valid_positions]
        gainSignal = selected.sum(dim=1)  # Integrate over pulse duration [freq, position]

        # ========================================
        # STORE RESULTS (WITH EDGE TRIMMING)
        # ========================================
        idx = pulse_idx * num + i  # Calculate storage index
        # Store central portion (trim 100 points from each edge)
        BGS3[idx, :, 100:-100] = gainSignal[:, 100:-100 + pulse]
        BFS3[idx, 100:-100] = BFS2[100:-100]  # Store BFS
        SW3[idx, 100:-100] = SW2[100:-100]  # Store spectral width
        Intensity3[idx, 100:-100] = Intensity2[100:-100]  # Store intensity

        # ========================================
        # MEMORY MANAGEMENT (CRITICAL FOR GPU)
        # ========================================
        del BFS2, SW2, Intensity2, tau, tau_conj, tau_outer, Gain, offsets, cols, selected, gainSignal
        torch.cuda.empty_cache()  # Clear GPU memory

    # Progress reporting
    print(f'\nBGS3 shape: {BGS3.shape}')
    print(f'BFS3 shape: {BFS3.shape}')

# ======================
# DATA SAVING OPTIONS
# ======================
# NumPy save options:
# np.save(f'./data/fangzhen/20250222_20-60ns_BGS.npy', BGS3)
# np.save(f'./data/fangzhen/20250222_20-60ns_BFS.npy', BFS3)
# np.save(f'./data/fangzhen/20250222_20-60ns_SW.npy', SW3)
# np.save(f'./data/fangzhen/20250222_20-60ns_Intensity.npy', Intensity3)

# MATLAB export option:
# savemat(f'./matlabs/fangzhen/20250222_20-60ns_data_generation.mat', data_dict)

# PyTorch save (recommended for tensor data):
data_dict = {
    'BFS': BFS3,
    'SW': SW3,
    'Intensity': Intensity3,
    'BGS': BGS3
}
torch.save(data_dict, f'../data/Simulation_BGS_20-60ns_5sample_{point}point.pt')