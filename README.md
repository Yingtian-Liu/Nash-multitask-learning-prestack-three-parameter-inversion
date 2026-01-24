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


## Sample Results



### Visual Comparison of Inversion Results
| Parameter Inversion | Estimated Result | True Value | Absolute Difference |
|:-----------:|:-----------:|:-----------:|:-------------------:|
| P-wave Velocity | ![](https://github.com/Yingtian-Liu/Nash-multitask-learning-prestack-three-parameter-inversion/blob/main/Image/Process%20visualization%20of%20the%20Nash-MTL-STCN%20method.png) | ![](https://p5-flow-imagex-sign.byteimg.com/tos-cn-i-a9rns2rl98/f829581538154fba8c898a6465bc00e0.png~tplv-a9rns2rl98-image.png?lk3s=8e244e95&rcl=20260124204216087CA240721D660BADE8&rrcfp=dafada99&x-expires=2085482536&x-signature=KWXQ47QaXAr%2F7im49PLz7OnFbrU%3D) | ![](https://p5-flow-imagex-sign.byteimg.com/tos-cn-i-a9rns2rl98/dc5484af26844650a2292618a5d0df87.png~tplv-a9rns2rl98-image.png?lk3s=8e244e95&rcl=20260124204216087CA240721D660BADE8&rrcfp=dafada99&x-expires=2085482536&x-signature=wk%2F7g%2FHWIMAn4spGPfJyBzs%2FJlo%3D) |

### Quantitative Evaluation Metrics
| Parameter       | R² Score | Pearson Correlation | RMSE |
|-----------------|----------|---------------------|------|
| P-wave Velocity | 0.9689   | 0.9913              | 0.05 |
| S-wave Velocity | 0.9673   | 0.9911              | 0.06 |
| Density         | 0.9447   | 0.9828              | 0.04 |

## Data
### Data Description
The data used in this code include:
- Synthetic seismic data generated from classic geophysical models
- Field prestack seismic data and corresponding well-log data
- Low-frequency constraint models for inversion regularization

### Data Download
The data file should be downloaded automatically when the code is run.

Alternatively, you can download the data file manually at this [link](https://example.com/data.npy) (replace with actual link) and place it in the same folder as train.py file.

All seismic and well-log data are saved in `.npy` format for efficient loading and processing.

## Repository Structure
