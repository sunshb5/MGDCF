# Introduction

本项目主要是针对论文"[MGDCF: Distance Learning via Markov Graph Diffusion for Neural Collaborative Filtering](https://arxiv.org/abs/2204.02338)"的复现。并在原论文的基础上做出了以下两点创新：
+ 增加了LightGCN-InfoBPR, ... , 等模型实验对照组，以验证论文中提出的排序损失InfoBPR函数的泛化性；
+ 更改了计算InfoBPR函数时负样本的数量，以验证不同数据、模型上InfoBPR何时更有效，调整后在异构MGDN、...上均有一定的提升。
    
本仓库实现了MGDCF的 TensorFlow 版本。




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
 

 
