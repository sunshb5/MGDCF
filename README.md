<p align="center">
<!-- <img src="MGDCF_LOGO.png" width="400"/> -->
<!-- </p> -->

# Introduction

本项目主要是针对论文"[MGDCF: Distance Learning via Markov Graph Diffusion for Neural Collaborative Filtering](https://arxiv.org/abs/2204.02338)"的复现与改进。并在原论文的基础上做出了以下三点创新：
+ 增加了LightGCN-InfoBPR，DropEdge-InfoBPR等模型实验对照组，以验证论文中提出的排序损失InfoBPR函数的泛化性；
+ 更改了计算InfoBPR时采取的负样本数量，以验证不同数据、模型上InfoBPR何时更有效，调整后在异构MGDN、同构MGDN上均有一定的提升；
+ 提出了如下结论：最优排序损失函数Loss_Rank应根据不同数据集、不同模型的特征进行选择，采用了更多负样本数量的InfoBPR在某些情况下并不一定优于BPR的效果；
    
本仓库实现了MGDCF的 TensorFlow 版本。

<p align="center">
<img src=".\architecture.png" height = "330" alt="" align=center />
<br><br>
<b>Figure 1.</b> Overall Framework of MGDCF.
</p>



# Paper Links

+ Paper Access:
    - **IEEE Xplore**: [https://ieeexplore.ieee.org/document/10384729](https://ieeexplore.ieee.org/document/10384729)
    - **ArXiv**: [https://arxiv.org/abs/2204.02338](https://arxiv.org/abs/2204.02338)



 
# Requirements

+ Linux
+ Python 3.10.12
+ tensorflow == 2.15.0
+ tf_geometric == 0.1.6
+ tf_sparse == 0.0.17
+ grecx == 0.0.8
+ tqdm=4.66.4
 


 
# Directory Structure

    ├── ReadMe.md            // 帮助文档
    
    ├── requirement.txt      // 环境依赖文件

    ├── cf_task_handle.py    // 主函数文件，调用不同模型处理CF任务
    
    ├── mgdcf                // MGDCF框架，包含同构MGDN、异构MGDN、部分GNN模型的实现以及用到的tools
    
    │   ├── layers      // 包含处理CF任务的各种模型
    
    │       └── __init__.py
    
    │       └── hetero_mgdn.py    // 同构MGDN模型
    
    │       └── homo_mgdn.py      // 异构MGDN模型
    
    │       └── specific_gnn.py   // 其他部分GNN模型

    │   ├── utils     // 包含模型用到的tools
    
    │       └── __init__.py
    
    │       └── homo_adjacency.py      // 同构图邻接矩阵建立
    
    │       └── normalized_factor.py   // 计算MGDN模型归一化分母
    
    │   └── __init__.py

    │   └── cache.py        // 构建缓存的tool，加速任务
    
    ├── scripts       // python运行脚本
    
    │   ├── amazon-book
    
    │       └── ......

    │   ├── gowalla
    
    │       └── ......

    │   ├── yelp
    
    │       └── ......

 
# Run

您可以运行如下命令来执行MGDCF:
```shell
cd scripts
sh ${DATASET}/$SCRIPT_NAME
```
例如:
```shell
cd scripts
sh amazon-book/HeteroMGDCF_yelp.sh
```
您也可以直接在命令行执行：
```shell
python -u cf_task_handle.py ${DATASET} ${Other parameters needed}
```
您也可以在colab平台的jupyter notebook上执行(其T4 GPU运行时满足本项目的大部分环境)。
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1SAbCp3spIATpNhGD3pTJuFhm7L2EdKqR#scrollTo=N-De8-JcaSb7)




# Cite

```
@ARTICLE{10384729,
  author={Jun Hu and Bryan Hooi and Shengsheng Qian and Quan Fang and Changsheng Xu},
  journal={IEEE Transactions on Knowledge and Data Engineering}, 
  title={MGDCF: Distance Learning via Markov Graph Diffusion for Neural Collaborative Filtering}, 
  year={2024},
  volume={},
  number={},
  pages={1-16},
  doi={10.1109/TKDE.2023.3348537}
}
```
 

 
