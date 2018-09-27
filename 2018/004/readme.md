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

### 2. 问题分类

我们知道，传统的pruning研究分成两个大类：
- _Fine-grained pruning_ 旨在独立地考察和剪枝weight tensor中的每个elements。由于操作空间大，可以几乎无损地实现较大比例的压缩，但是需要配合[EIE](https://arxiv.org/pdf/1602.01528.pdf)一类的针对性硬件结构来保证inference的效率。
- _Coarse-grained pruning_ 粗粒度pruning操作针对weight tensor中的某一块操作，例如一行、一列、一个channel。由于<span style="color:blue">结构整饬，虽然压缩比例较小，但是inference的效率不受什么影响，能够在通用硬件上实现较好的性能</span>。

本文中研究的是Coarse-grained pruning中channel层面的压缩。

### 3. AMC的pipeline

文章提出的AMC结构如下图所示：

<center><img src="./004_01.png?raw=true" width = "55%" /></center>

可以看到整体上是标准的RL结构，而且能大体看出是[Actor-Critic](http://rail.eecs.berkeley.edu/deeprlcourse-fa17/f17docs/lecture_5_actor_critic_pdf.pdf)结构，具体的大家可以参考莫烦大佬的[讲解](https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/6-1-actor-critic/)和[代码](https://github.com/princewen/tensorflow_practice/tree/master/Basic-Actor-Critic)。实际上本文是使用了Actor-Critic结构的最新变体DDPG（Deep Deterministic Policy Gradient）结构，在后面会进一步描述。

### 4. 状态空间定义

对于每个layer，考虑优化如下的状态空间

$$s_t = \left( t,n,c,h,w,stride,k,FLOPs[t],reduced,rest,a_{t-1} \right)$$

其中，$t$为layer的编号，$FLOPs[t]$为layer的FLOPs值，$a_{t-1}$为上一层的压缩率（此处定义为压缩至百分之多少，也即稀疏率）

而比较有趣的是$reduced$和$rest$，前者是指本层前的所有layer剪枝减掉的所有FLOPs；后者是指本层后的所有layer原始的所有FLOPs。这两个参数的存在是方便agent区分不同的layer，以便给出不同的剪枝策略。

### 5. 强化学习agent选择

传统的NAS中，一般Action space是高度离散的，例如$\{64,128,256,512\}$就足够覆盖一个优秀结构所需要用到的channels数量。额外增加一个65对搜索结构的优化没有太大帮助。

然而在网络压缩问题当中，网络结构对压缩率的敏感程度很高。可能0.19的压缩率网络就灾难性崩坏了，但是0.2的压缩率网络acc会基本不降。为了适应不用backbone的压缩需求，显然AMC当中的agent需要拥有连续的action space，输出一个具体的压缩率而不是从压缩率列表中选一个。这就指向了Policy Gradients结构而不是最早的DQN系列结构。

至于为什么选择Actor-Critic，莫烦有过[比较精彩的论述](https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/6-1-A-AC/)
> 我们有了像 Q-learning这么伟大的算法, 为什么还要瞎折腾出一个 Actor-Critic? 原来 Actor-Critic 的 Actor 的前生是 Policy Gradients, 这能让它毫不费力地在连续动作中选取合适的动作, 而 Q-learning 做这件事会瘫痪. 那为什么不直接用 Policy Gradients 呢? 原来 Actor Critic 中的 Critic 的前生是 Q-learning 或者其他的 以值为基础的学习法 , 能进行单步更新, 而传统的 Policy Gradients 则是回合更新, 这降低了学习效率.



本文实际选用的DDPG和Actor-Critic原理上是一码事，DDPG是为了修正Actor-Critic存在如下的问题衍生出来的工作。
> Actor-Critic 涉及到了两个神经网络, 而且每次都是在连续状态中更新参数, 每次参数更新前后都存在相关性, 导致神经网络只能片面的看待问题, 甚至导致神经网络学不到东西

# Results

May the force be with you

# Insights

May the force be with you

-  <span style="color:red">May the force be with you</span> <span style="color:grey">[May the force be with you]</span>

<center><img src="./001_01.png?raw=true" width = "80%" /></center>