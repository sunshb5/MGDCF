<p align="center">
<img src="MGDCF_LOGO.png" width="400"/>
</p>


# 项目介绍

本项目主要是针对论文"[MGDCF: Distance Learning via Markov Graph Diffusion for Neural Collaborative Filtering](https://arxiv.org/abs/2204.02338)"的复现。并在原论文的基础上做出了以下两点创新：
+ 增加了LightGCN-InfoBPR, ... , 等模型实验对照组，以验证论文中提出的排序损失InfoBPR函数的泛化性；
+ 调整了InfoBPR
    
本仓库包含该论文的 TensorFlow 实现。

# 论文链接

+ Paper Access:
    - **IEEE Xplore**: [https://ieeexplore.ieee.org/document/10384729](https://ieeexplore.ieee.org/document/10384729)
    - **ArXiv**: [https://arxiv.org/abs/2204.02338](https://arxiv.org/abs/2204.02338)
 
# 环境依赖

+ Linux
+ Python 3.7
+ tensorflow == 2.7.0
+ tf_geometric == 0.1.5
+ tf_sparse == 0.0.17
+ grecx >= 0.0.6
+ tqdm=4.51.0
 
 
# 目录结构描述
    ├── ReadMe.md           // 帮助文档
    
    ├── AutoCreateDDS.py    // 合成DDS的 python脚本文件
    
    ├── DDScore             // DDS核心文件库，包含各版本的include、src、lib文件夹，方便合并
    
    │   ├── include_src     // 包含各版本的include、src文件夹
    
    │       ├── V1.0
    
    │           ├── include
    
    │           └── src
    
    │       └── V......
    
    │   └── lib             // 包含各版本的lib文件夹
    
    │       ├── arm64       // 支持arm64系统版本的lib文件夹
    
    │           ├── V1.0
    
    │           └── V......
    
    │       └── x86         // 支持x86系统版本的lib文件夹
    
    │           ├── V1.0
    
    │           └── V......
    
    ├── target              // 合成结果存放的文件夹
    
    └── temp                // 存放待合并的服务的服务文件夹

 
# 使用说明

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
 

 
