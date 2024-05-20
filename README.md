## CNN-MLP and CNN autoencoder with supplemental scalar values

This repository contains source codes utilized in a part of "Convolutional neural networks for fluid flow analysis: toward effective metamodeling and low-dimensionalization," [Theor. Comput. Fluid Dyn. 35 (5), 633-658 (2021)](https://doi.org/10.1007/s00162-021-00580-0)

In this repository, sample codes for two types of neural networks are provided;
1. a CNN-MLP type neural network that is trained to estimate drag and lift force coefficient of flow over a flat plate with its supplemental scalar values, i.e., a Re number and an angle of attack
2. a CNN autoencoder which is trained to low-dimensionalize a two-dimensional isotropic homogeneous decaying turbulence with its supplemental scalar values, i.e., an initial Re number

Although the provided code corresponds to 'Case 1' stated in the paper (which concatenates the scalar value at the earliest layer of the network), it can easily be modified for the other cases as demonstrated in our paper.
Detailed information for the construction of the networks are stated in tables 1--3 in the paper.

CNN-MLP             |  CNN autoencoder
:-------------------------:|:-------------------------:
![alt text](https://github.com//Masaki-Morimoto/CNN-MLP_and_CNN-AE-network-structure-with-supplemental-scalar-inputs/blob/images/fig3_CNN-MLP.png?raw=true)  |  ![alt text](https://github.com//Masaki-Morimoto/CNN-MLP_and_CNN-AE-network-structure-with-supplemental-scalar-inputs/blob/images/fig4_CNN-AE.png?raw=true)

<div style="text-align: center;">A CNN-MLP and CNN autoencoder with supplemental scalar values utilized in this study. Copyright ©️ 2021 by the Springer.</div>

## Information

Author: Masaki Morimoto ([Keio University](https://kflab.jp/ja/)) and Kai Fukami ([UCLA](https://sites.google.com/view/kai-fukami/home?authuser=0))

This repository contains

- CNN-MLP_model_with_scalar-input.py
- CNN-AE_model_with_scalar_input.py


For citations, please use the reference below:

Masaki Morimoto, Kai Fukami, Kai Zhang, Aditya, G. Nair, and Koji Fukagata "Convolutional neural networks for fluid flow analysis: toward effective metamodeling and low-dimensionalization," Theor. Comput. Fluid Dyn. 35 (5), 633-658 (2021)

Authors provide no guarantees for this code.
Use as-is and for academic research use only; no commercial use allowed without permission.
The code is written for educational clarity and not for speed.

## Requirements
- Python 3.X
- keras
- tensorflow
- numpy
- pandas
- cv2
