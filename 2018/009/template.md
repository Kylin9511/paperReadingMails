# Headline

大家好：

 <b><span style="color:green">这是2018年度第8篇Arxiv Weekly。</span> </b>

本文是 __模型压缩__ 方向的文章。

# Highlight

> __对网络压缩技术的探究性工作，指出了几个旧有的错误印象，得出了新的指导性结论，能够辅助优化网络压缩过程__

# Information

### Title
_Rethinking the Value of Network Pruning_

### Link
https://arxiv.org/pdf/1810.05270.pdf

### Source

- 加州伯克利大学（University of California, Berkeley）
- 清华大学（Tsinghua University）

### Publication

Under review as a conference paper at International Conference on Machine Learning (ICLR) 2019

# Introduction

神经网络压缩作为实现神经网络小型化的主流技术之一，引发了广泛的关注。传统的神经网络压缩算法架构主要是三段式的： __训练大型网络->剪枝->finetune__。其核心是在剪枝过程中设计某种策略或者评价标准，使得不重要的参数被剪掉；而重要的参数被保留。从而实验压缩模型、acc基本不降低。

本文中通过实验发现了“反常识”的现象。首先我们通过对6个state-of-the-art的pruning算法的实验，发现<span style="color:blue">对训练好的大网络进行剪枝-finetune，其性能不如对剪枝后的网络进行重训（此时用random initialization）来得好</span>。故而如果剪枝算法中假设了目标网络的结构，那么之前复杂的pipeline没有太大的意义。

通过对多个剪枝算法、多个数据集、多个任务的交叉组合验证，我们发现了如下的现象/结论：
- 获取高效的小网络 __不需要__ 训练一个大规模的、参数/描述能力过冗余的网络
- 经过剪枝的小网络参数 __不需要__ 参考充分训练获得的原网络的对应参数。
- 一个经过剪枝的小网络是否高效，核心不在于保留了大网络中的哪些参数，而是剪枝后是什么结构。传统的剪枝pipeline之所以work，不是因为挑选和保留了大网络中足够重要的参数，而是这个挑选的过程耦合了一个网络结构搜索过程。 __换言之，pruning其实是一种NAS技术__。

# Keys

### 问题和背景

在传统上，我们认为神经网络的“over parameterization”是显然的和需要解决的。而pruning则是解决这个问题的最直接手段之一。

而在之前的观念里，以及之前pruning相关的文章中，训练出一个准确的，强大的大网络对剪枝后的网络很重要。<span style="color:red">因为小网络的参数基本是使用大网络剪枝后获得的参数进行finetune得到的，而这么做是因为主流认为大网络经过充分训练能够得到强有力的参数，值得继承</span>。

本文认为不然。

本文的核心想法是，pruning后小网络效果好，不依赖于大网络的参数，而依赖于小网络的结构本身。大网络参数在完成了“挑选小网络结构特性”的使命后就成为了累赘，没有太大保留价值。

为了证明这个观点，本文参考了六种最新的pruning算法/架构作为实验的背景。现列举如下：

1. [$L_1$-norm based channel pruning](https://www.researchgate.net/profile/Igor_Durdanovic/publication/307536925_Pruning_Filters_for_Efficient_ConvNets/links/585189cf08ae95fd8e168343/Pruning-Filters-for-Efficient-ConvNets.pdf)。
这篇文章是ICLR 2017的会议文章，也是最早提出做channel pruning的文章。<span style="color:gray">[关于weight/channel/layer pruning，可参考之前的[Arxiv Weekly 2018/004](https://github.com/luzhilin19951120/paperReadingMails/tree/master/2018/004)中的论述]</span>
本文作为channel pruning的开山之作，自然使用的是很简洁的方式。具体来说就是使得每一层当中$L_1$范数最小的某个比例 $p$ 的channel被剪枝。

2. [ThiNet](http://openaccess.thecvf.com/content_ICCV_2017/papers/Luo_ThiNet_A_Filter_ICCV_2017_paper.pdf)
这篇文章是ICCV 2017的会议文章，思路更进一步。同样按比例剪枝channel，但是criterion改为优先选择下一层
# Results

May the force be with you

# Insights

May the force be with you

-  <span style="color:red">May the force be with you</span> <span style="color:grey">[May the force be with you]</span>

<center><img src="./001_01.png?raw=true" width = "80%" /></center>