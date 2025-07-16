# BOTDA_PSRN
Unsupervised super-spatial-resolution Brillouin frequency shift extraction based on physical enhanced spatial resolution neural network

## Abstract
 Spatial resolution (SR), a core parameter of Brillouin optical time-domain analysis (BOTDA) sensors, determines the minimum fiber length over which physical perturbations can be accurately detected. However, the phonon lifetime in the fiber imposes an inherent limit on the SR, making sub-meter-level SR challenging in high-SR monitoring scenarios. Conventional SR enhancement approaches, constrained by hardware limitations, often involve complex systems, or increased measurement times. Although traditional deconvolution methods can mitigate hardware constraints, they suffer from distortion due to the nonlinear nature of the BOTDA response. Supervised deep learning approaches have recently emerged as an alternative, offering faster and more accurate post-processing through data-driven models. However, the need for extensive labeled data and the lack of physical priors lead to high computational costs and limited generalization. To overcome these challenges, we propose an unsupervised deep learning deconvolution framework, Physics-enhanced SR deep neural network (PSRN) guided by an approximate convolution model of the Brillouin gain spectrum (BGS). We validate PSRN on both simulated and experimental data. The results demonstrate that PSRN can reconstruct sub-meter SR BGS and accurately retrieve the corresponding Brillouin frequency shift (BFS) from any low-resolution BGS input in a plug-and-play fashion, leveraging the interplay between neural network inference and embedded physical priors. In the case of a 0.5 m hot-spot, the BFS retrieved by PSRN is highly consistent with that obtained from a supervised trained neural network (STNN). Unlike the STNN, our unsupervised method does not require labeled data or training process. Furthermore, our framework can solve the inverse problem with more general applicability, enabling high-SR BGS reconstruction and BFS retrieval across varying pulse widths, fiber lengths, and frequency sweep steps. This plug and play post-processing technique paves the way to enable novel high-SR BOTDA sensors, representing a significant advancement for Brillouin sensing applications 
## Contact:
- For inquiries or collaborations, reach out to [d202280977@hust.edu.cn].

## Repository Contents

This repository includes essential codes for evaluating and simulating our proposed method, featuring:

- **codes**:Code for PSRN model, simulation/experimental training, and creating simulation BGS.

- **data**: The simulated BGS and experimental BGS used in the article.

- **matlabs**: Save the generated code results and the script for plot.
