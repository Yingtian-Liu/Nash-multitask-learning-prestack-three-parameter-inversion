# Nash-MTL-STCN: Nash Equilibrium Multitask Learning for Prestack Three-Parameter Inversion
[Yingtian Liu](yingtianliu06@outlook.com), Yong Li, Huating Li, Junheng Peng, Zhangquan Liao, Wen Feng, and Mingwei Wang

Codes and data for the manuscript: "Nash-multitask learning-semisupervised temporal convolutional network method for prestack three-parameter inversion".

This repository contains the implementation of the paper:

Yingtian Liu et al., "**Nash-multitask learning-semisupervised temporal convolutional network method for prestack three-parameter inversion**," in *Geophysics*, 2025, 90/4. [[Paper PDF]](https://pubs.geoscienceworld.org/seg/geophysics/article-abstract/90/4/R175/654779/Nash-multitask-learning-semisupervised-temporal)

## Abstract
Deep-learning techniques have been widely used in prestack three-parameter inversions to address ill-posed problems. Among these techniques, multitask learning (MTL) methods can simultaneously train multiple tasks, enhancing model generalization and predictive performance. However, existing MTL methods typically adopt heuristic or nonheuristic approaches to jointly update the gradient of each task, which often leads to gradient conflicts between different tasks, reducing inversion accuracy. To address this issue, we develop a semisupervised temporal convolutional network (STCN) method based on Nash equilibrium, referred to as the Nash-MTL-STCN method. First, temporal convolutional networks with noncausal convolution and convolutional neural networks (CNNs) are used as multitask layers to extract shared features from partial angle stack seismic data, with CNNs serving as the single-task layer. Subsequently, a feature mechanism is used to extract shared features in the multitask layer through hierarchical processing, and the gradient combination of these shared features is treated as a Nash game for the optimization of strategy and joint updates. This approach maximizes the overall utility of the three-parameter inversion while alleviating gradient conflicts. In addition, to enhance the generalization and stability of the network, we incorporate geophysical forward modeling and low-frequency constraints into the network. Experimental results demonstrate that our method resolves the gradient conflict issue associated with conventional MTL methods with constant weights and achieves higher precision than four widely used nonheuristic MTL methods. Further experiments using field data also validate the effectiveness of our method.


## Hierarchical processing structure of the proposed feature mechanism.
![](https://github.com/Yingtian-Liu/Nash-multitask-learning-prestack-three-parameter-inversion/blob/main/Image/Process%20visualization%20of%20the%20Nash-MTL-STCN%20method.png)


## Process visualization of the Nash-MTL-STCN method
![](https://github.com/Yingtian-Liu/Nash-multitask-learning-prestack-three-parameter-inversion/blob/main/Image/Process%20visualization%20of%20the%20Nash-MTL-STCN%20method.png)

## Quantitative Evaluation Metrics
| Parameter       | Pearson Correlation |  R² Score  | SSIM |
|-----------------|----------|---------------------|------|
| P-wave Velocity | 0.9907   | 0.9689              | 0.9097 |
| S-wave Velocity | 0.9906   | 0.9673              | 0.8917 |
| Density         | 0.9792   | 0.9447              | 0.9110 |

## Data
### Data Description
The data used in this code include:
- Synthetic seismic data generated from classic geophysical models
- Low-frequency constraint models for inversion regularization

### Data Download
The data file should be downloaded automatically when the code is run.

Alternatively, you can download the data file manually at this [link](https://example.com/data.npy) (replace with actual link) and place it in the same folder as train.py file.

All seismic and well-log data are saved in `.npy` format for efficient loading and processing.

## Citation:
If you have found our code and data useful, we kindly ask you to cite our work
```
@article{liu2025nash,
  title={Nash-multitask learning-semisupervised temporal convolutional network method for prestack three-parameter inversion},
  author={Liu, Yingtian and Li, Yong and Li, Huating and Peng, Junheng and Liao, Zhangquan and Feng, Wen and Wang, Mingwei},
  journal={Geophysics},
  volume={90},
  number={4},
  pages={R175--R193},
  year={2025},
  publisher={Society of Exploration Geophysicists}
}
```
