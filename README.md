## CNN-MLP and CNN autoencoder with supplemental scalar values

This repository contains a simple source code utilized in a part of "Convolutional neural networks for fluid flow analysis: toward effective metamodeling and low-dimensionalization," arXiv:2101.02535 [physics.flu-dyn.].

In this repostiry, two types of the network are provided;
1. a CNN-MLP type neural network which is trained to estimate drag and lift force coefficient of flow over a flat plate with its supplemental scalar values, i.e., a Re number and an angle of attack
2. a CNN autoeoncder which is trained to low-dimensionalize a two-dimensional isotropic homogeneous decaying turbulence with its supplemental scalar values, i.e., an initial Re number

Although the provided code corresponds to 'Case 1' stated in the paper (which concatenates the scalar value at the earliest layer of the network), it can easily be modified for the other cases as demonstrated in our [arXiv](https://arxiv.org/abs/2101.02535).
A detailed information for the construction of the networks are stated in tabels 1--3 in the paper.

Vorticity field             |  Grad-CAM map
:-------------------------:|:-------------------------:
![alt text](https://github.com//Masaki-Morimoto/Grad-CAM_for_fluid-flows/blob/images/vorticity.png?raw=true)  |  ![alt text](https://github.com//Masaki-Morimoto/Grad-CAM_for_fluid-flows/blob/images/grad-cam.png?raw=true)

<div style="text-align: center;">Vorticity field of a cylinder wake (input data) and its Grad-CAM map of the force coefficient estimation.</div>

## Information

Author: Masaki Morimoto ([Keio University](https://kflab.jp/ja/))

This repository contains

Grad-CAM_for_CD-prediction_cylinder.ipynb

For citations, please use the reference below:

Masaki Morimoto, Kai Fukami, Kai Zhang, and Koji Fukagata "Generalization techniques of neural networks for fluid flow estimation," arXiv:2011.11911 (2020).

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
