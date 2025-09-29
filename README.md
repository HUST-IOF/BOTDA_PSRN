# BOTDA_PSRN
Unsupervised super-spatial-resolution Brillouin frequency shift extraction based on physical enhanced spatial resolution neural network

## Key Features
- **BGS convolution model-based physical constraint and self-supervised learning:** PSRN integrates a physical convolution model of the BGS with deep neural networks, enabling self-supervised optimization guided by physical priors to enhance robustness and interpretability without requiring labeled training data. 
- **Physics-guided regularization:** PSRN incorporates total variation (TV) regularization as a key constraint to alleviate the multi-solution problem arising from unsupervised training. 
- **Plug-and-play generalization:** PSRN achieves plug-and-play sub-meter SR reconstruction across diverse BOTDA configurations (varying pulse widths/sweep steps) without retraining, significantly outperforming supervised deep learning and conventional methods (e.g., DPP).


## Contact:
- For inquiries or collaborations, reach out to [d202280977@hust.edu.cn].

## Repository Contents

This repository includes essential codes for evaluating and simulating our proposed method, featuring:

- **codes**:Code for PSRN model, simulation/experimental training, and creating simulation BGS.

- **data**: The simulated BGS and experimental BGS used in the article.

- **matlabs**: Save the generated code results and the script for plot.

