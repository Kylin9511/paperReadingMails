# Headline

大家好：

 <b><span style="color:green">这是2018年度第4篇Arxiv Weekly。</span> </b>

本文是 __模型压缩、NAS__ 方向的文章。

# Highlight

> __Song Han组新工作，利用网络自动搜索技术进行网络压缩，并取得超过手动压缩的性能。__

# Information

### Title
_AMC: AutoML for Model Compression and Acceleration on Mobile Devices_

### Link
https://arxiv.org/pdf/1802.03494.pdf

### Source

- 麻省理工大学（Massachusetts Institute of Technology）
- 西安交大（Xian Jiaotong University）
- 谷歌（Google）

# Introduction

模型压缩已经成为神经网络移动化技术的核心之一。在有限的计算资源和能量条件下如何实现模型的部署，是神经网络落地的关键问题。

CNN压缩技术的出现给这个问题提供了一个有力的解决方案。然而传统的模型压缩是纯手动的，基本是基于一些启发式的规则（heuristic rule-based compression）探索模型储存、效率和性能的最佳平衡。 手动的启发式平衡点，显然是次优化的，而且这样的压缩十分耗时，需要有经验的工程师和大量的时间和算力。

本文中提出了自动化模型压缩策略（<span style="color:brown">AutoML for Model Compression, AMC</span>）, 基于强化学习来进行模型压缩。实验结果表明，相比于启发式规则指导的模型压缩，AMC能够获得更高的压缩率的同时保留更多的性能。

以FLOPs为指标，用AMC指导VGG的4倍压缩相比于手动压缩涨点2.7%。在MobileNet上利用AMC进一步压缩网络，在安卓手机平台和Titan XP GPU上分别实现了ImageNet Inference中1.81$\times$ 和1.43$\times$的加速，top1 acc仅仅降低了0.1%。

# Keys

### 1. 先验pruning技巧

一般来说，启发式的压缩会考虑如下的因素：
 - 尽量降低第一个layer的pruning rate，因为第一个layer往往参数量不大，而且其抽象的底层特征是否准确会影响后续的所有layer的性能。
 - 尽量多在FC layer做压缩，因为FC layer一般非常臃肿，占据了极大的储存和计算资源，并且高层的特征重复度高。
 - 尽量找准每个layer的敏感点，pruning rate不能超过这个阈值。

 但是显然，<span style="color:red">这三条原则从最优性上不能得证，从可操作性上需要大量的实验去调整各个layer的阈值，最糟糕的是只要模型改变所有的努力都需要重来一次，先验成果推广困难。</span>

### 2. AMC的pipeline

文章提出的AMC结构如下图所示：

<center><img src="./003_01.png?raw=true" width = "55%" /></center>

可以看到

# Results

May the force be with you

# Insights

May the force be with you

-  <span style="color:red">May the force be with you</span> <span style="color:grey">[May the force be with you]</span>

<center><img src="./001_01.png?raw=true" width = "80%" /></center>