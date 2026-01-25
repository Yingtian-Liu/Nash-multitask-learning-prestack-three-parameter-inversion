# 🎯 Nash-MTL-STCN: Nash Equilibrium Multitask Learning for Prestack Three-Parameter Inversion  

**Authors:** [Yingtian Liu](mailto:yingtianliu06@outlook.com), Yong Li, Huating Li, Junheng Peng, Zhangquan Liao, Wen Feng, and Mingwei Wang  

> Code and data for the manuscript: **"Nash-multitask learning-semisupervised temporal convolutional network method for prestack three-parameter inversion"**  
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.1-red.svg)](https://pytorch.org/)
[![Python 3.7+](https://img.shields.io/badge/Python-3.7%2B-green.svg)](https://www.python.org/)
---

## 📖 Overview  
This repository contains the implementation of the paper:  

**Yingtian Liu et al.**, *"Nash-multitask learning-semisupervised temporal convolutional network method for prestack three-parameter inversion,"* in **Geophysics**, 2025, 90(4).  
[[📄 Paper PDF]](https://pubs.geoscienceworld.org/seg/geophysics/article-abstract/90/4/R175/654779/Nash-multitask-learning-semisupervised-temporal)  

### ✨ Abstract  
Deep-learning techniques have been widely used in prestack three-parameter inversions to address ill-posed problems. Among these techniques, multitask learning (MTL) methods can simultaneously train multiple tasks, enhancing model generalization and predictive performance. However, existing MTL methods typically adopt heuristic or nonheuristic approaches to jointly update the gradient of each task, which often leads to gradient conflicts between different tasks, reducing inversion accuracy.  

To address this issue, we develop a **semisupervised temporal convolutional network (STCN) method based on Nash equilibrium**, referred to as the **Nash-MTL-STCN** method. First, temporal convolutional networks with noncausal convolution and convolutional neural networks (CNNs) are used as multitask layers to extract shared features from partial angle stack seismic data, with CNNs serving as the single-task layer. Subsequently, a feature mechanism is used to extract shared features in the multitask layer through hierarchical processing, and the gradient combination of these shared features is treated as a **Nash game** for the optimization of strategy and joint updates. This approach maximizes the overall utility of the three-parameter inversion while alleviating gradient conflicts.  

Additionally, to enhance the generalization and stability of the network, we incorporate **geophysical forward modeling** and **low-frequency constraints** into the network. Experimental results demonstrate that our method resolves the gradient conflict issue associated with conventional MTL methods with constant weights and achieves higher precision than four widely used nonheuristic MTL methods. Further experiments using field data also validate the effectiveness of our method.  

---

## 🎨 Method Visualization  

### 🔬 Exploration of Utility Function Parameter Space  
![Utility Function Exploration](https://github.com/Yingtian-Liu/Nash-multitask-learning-prestack-three-parameter-inversion/blob/main/Image/Exploration%20of%20utility%20function%20parameter%20space.png)  

### 📊 Process Visualization of the Nash-MTL-STCN Method  
![Process Visualization](https://github.com/Yingtian-Liu/Nash-multitask-learning-prestack-three-parameter-inversion/blob/main/Image/Process%20visualization%20of%20the%20Nash-MTL-STCN%20method.png)  

---

## 📈 Quantitative Evaluation Metrics  

| Parameter         | Pearson Correlation | R² Score  | SSIM     |
|-------------------|---------------------|-----------|----------|
| **P-wave Velocity** | 0.9907              | 0.9689    | 0.9097   |
| **S-wave Velocity** | 0.9906              | 0.9673    | 0.8917   |
| **Density**         | 0.9792              | 0.9447    | 0.9110   |

---

## 🚀 Training & Testing
To train the model using the default parameters (as reported in the paper) and test it on the full Marmousi 2 model, run:
```
python train.py
```

## 🏗️ Setup Environment  
Create a conda environment and install dependencies:  
```
bruges==0.5.4
matplotlib==3.8.0
numpy==1.24.0
python-dateutil==2.8.2
pytorch==2.1.1
tqdm==4.67.1
wget==3.2
```
## 📁 Data Description
The data used in this code include:
- Synthetic seismic data generated from classic geophysical models
- Low-frequency constraint models for inversion regularization

## 📚 Citation
If you find our work useful, please cite:
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
To reference this open-source implementation, please use:
```
@misc{nashmtlstcn2024,
  author       = {Yingtian Liu},
  title        = {{Nash-MTL-STCN}: Open-source implementation for prestack three-parameter inversion},
  year         = {2024},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/Yingtian-Liu/Nash-multitask-learning-prestack-three-parameter-inversion}},
  note         = {Accessed: 2026-01-25}
}
```


