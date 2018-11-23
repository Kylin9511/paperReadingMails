[TOC]

# Headline

大家好：

 <b><span style="color:green">这是2018年度第8篇Arxiv Weekly。</span> </b>

本文是 __压缩综述__ 方向的文章。

# Highlight

> __对网络压缩技术的探究性工作，指出了几个旧有的错误印象，得出了新的指导性结论，能够辅助优化网络压缩过程__

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

神经网络压缩作为实现神经网络小型化的主流技术之一，引发了广泛的关注。传统的神经网络压缩算法架构主要是三段式的： __训练大型网络->剪枝->finetune__。其核心是在剪枝过程中设计某种策略或者评价标准，使得不重要的参数被剪掉；而重要的参数被保留。从而实验压缩模型、acc基本不降低。

本文中通过实验发现了“反常识”的现象。首先我们通过对6个state-of-the-art的pruning算法的实验，发现<span style="color:blue">对训练好的大网络进行剪枝-finetune，其性能不如对剪枝后的网络进行重训（此时用random initialization）来得好</span>。故而如果剪枝算法中假设了目标网络的结构，那么之前复杂的pipeline没有太大的意义。

通过对多个剪枝算法、多个数据集、多个任务的交叉组合验证，我们发现了如下的现象/结论：
- 获取高效的小网络 __不需要__ 训练一个大规模的、参数/描述能力过冗余的网络
- 经过剪枝的小网络参数 __不需要__ 参考充分训练获得的原网络的对应参数。
- 一个经过剪枝的小网络是否高效，核心不在于保留了大网络中的哪些参数，而是剪枝后是什么结构。传统的剪枝pipeline之所以work，不是因为挑选和保留了大网络中足够重要的参数，而是这个挑选的过程耦合了一个网络结构搜索过程。 __换言之，pruning其实是一种NAS技术__。

# Keys

### 1.问题

在传统上，我们认为神经网络的“over parameterization”是显然的和需要解决的。而pruning则是解决这个问题的最直接手段之一。

而在之前的观念里，以及之前pruning相关的文章中，训练出一个准确的，强大的大网络对剪枝后的网络很重要。<span style="color:red">因为小网络的参数基本是使用大网络剪枝后获得的参数进行finetune得到的，而这么做是因为主流认为大网络经过充分训练能够得到强有力的参数，值得继承</span>。

本文认为不然。

本文的核心想法是，pruning后小网络效果好，不依赖于大网络的参数，而依赖于小网络的结构本身。大网络参数在完成了“挑选小网络结构特性”的使命后就成为了累赘，没有太大保留价值。

### 2.背景

为了证明这个观点，本文参考了六种最新的pruning算法/架构作为实验的背景。现列举如下：

##### [1. $L_1$-norm based channel pruning](https://www.researchgate.net/profile/Igor_Durdanovic/publication/307536925_Pruning_Filters_for_Efficient_ConvNets/links/585189cf08ae95fd8e168343/Pruning-Filters-for-Efficient-ConvNets.pdf)。
这篇文章是ICLR 2017的会议文章，也是最早提出做channel pruning的文章。<span style="color:gray">[关于weight/channel/layer pruning，可参考之前的[Arxiv Weekly 2018/003](https://github.com/luzhilin19951120/paperReadingMails/tree/master/2018/003)中的论述]</span>

本文作为channel pruning的开山之作，自然使用的是很简洁的方式。具体来说就是使得每一层当中$L_1$范数最小的某个比例 $p$ 的channel被剪枝。

<center>
<img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/008/008_append_01.png?raw=true" width = 85% />
</center>

可以看到，channel pruing对计算量的节省是直接有效的，对第i个kernel剪枝一个通道将节省$r_i$次计算。

$$r_i= n_ik^2h_{i+1}w_{i+1} + n_{i+2}k^2h_{i+2}w_{i+2}$$

在具体操作的时候，channel pruning还会一道一些问题。

首先是选择channel to prune。本文采用了所有pruing文章几乎通用的一个思v路，就是prune掉不敏感（sensitive）的那部分channel。敏感性的衡量方式有很多种，其中最consuming的就是分别剪枝训练后之间看掉点的数目。而本文在各个layer之间的L1 sum分布和敏感性之间建立了对应关系，故而不用训练就能知道个大概。

<center>
<img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/008/008_append_02.png?raw=true" width = 85% />
</center>

另一个问题是channel pruning的一个细节。也即计算后续pruning的过程中是否考虑前面pruning带来的影响。

<center>
<img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/008/008_append_03.png?raw=true" width = 85% />
</center>

最后值得一提的是，在channel pruning当中，由于pruning操作会影响下一轮feature map的大小，故在resnet等存在分枝的结构中，需要注意对应关系。本文认为Identity path相比Residual path更加重要，故而在冲突的时候以Identity path的选择为准。

<center>
<img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/008/008_append_04.png?raw=true" width = 85% />
</center>

##### [2. ThiNet](http://openaccess.thecvf.com/content_ICCV_2017/papers/Luo_ThiNet_A_Filter_ICCV_2017_paper.pdf)
这篇文章是ICCV 2017的会议文章，思路更进一步。同样按比例剪枝channel，但是criterion改为优先选择下一层的feature map中参数最小的channel对应的卷积核进行剪枝。从直觉上，这样会更加靠近我们pruning同时不影响acc性能的意图。
<center>
<img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/008/008_01.png?raw=true" width = 55% />
</center>

##### [3. regression based Feature Reconstruction](http://openaccess.thecvf.com/content_ICCV_2017/papers/He_Channel_Pruning_for_ICCV_2017_paper.pdf)
这篇文章是ICCV 2017的会议文章，思路更更进一步。同样按比例剪枝channel，但是criterion改为尽量使得剪枝后下下层的feature map受影响最小。显然这又进一步试图保留原网络的性能。
<center><img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/008/008_02.png?raw=true" width = 55% /></center>

##### [4. Network Slimming](http://openaccess.thecvf.com/content_ICCV_2017/papers/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.pdf)
这篇文章是ICCV 2017的会议文章，<span style="color:blue">与前面文章不同，本文是在训练过程中自动产生每个层的pruning rate，所以结构不经过训练无法得到</span>。

<center>
<img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/008/008_append_05.png?raw=true" width = 85% />
</center>

本文采用的方式是引用BN层的scaler $\alpha$作为scaling factor，然后将这个factor动态乘到每一个channel上，有点类似SE的“元操作”。为了保证最后有足够的channel scaling factor是near-zero的，文章引入了一个乘子稀疏化的loss如下式。其中g是稀疏化函数。

$$L = \sum_{(x,y)}l\left(f(x,W),y\right) + \lambda\sum_{\gamma \in \Theta}g(\gamma)$$

在完成稀疏化乘子网络训练后，即可动态剪枝去除其中不重要的channel。

<center><img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/008/008_03.png?raw=true" width = 85% /></center>

##### [5. Sparse Structure Seletction](https://arxiv.org/pdf/1707.01213.pdf)
这篇文章是ECCV 2018会议文章。文章是Network Slimming的变体，仍然是生成了scaling factor指导pruning，只不过不是channel pruning而是block pruning。有些像[BlockDrop: Dynamic Inference Paths in Residual Networks](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wu_BlockDrop_Dynamic_Inference_CVPR_2018_paper.pdf)一文，当然，本文没有引入Reinforcement Learning。
<center><img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/008/008_04.png?raw=true" width = 85% /></center>

##### [6. Non-structured Weight Pruning](https://papers.nips.cc/paper/5784-learning-both-weights-and-connections-for-efficient-neural-network.pdf)
这篇文章为NIPS 2015会议文章，韩松代表作之一，引用已经800+。核心方法是“啥也不说，就是硬干”，直接进行element-wise pruning，然后finetune。后续还有所谓的DSD等方法，其实就是反复硬干。至于pruning rate就是手工大量实验堆叠出一个最佳值即可。
<center><img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/008/008_05.png?raw=true" width = 85% /></center>

# Experiments

### 1.实验分类
本文属于探究性质的论文，所以实验的作用是探索和证明文章的观点。

文章把6个baseline网络分为两类，一类是所谓的Pre-defined Target Architecture（前三个实验）；另一类是所谓的Automatically Discovered Target Architecture（后三个实验）。其区别就是不训练能不能画出来大概的结构，如下图所示：
<center><img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/008/008_06.png?raw=true" width = 45% /></center>

### 2.训练细节
在训练参数上，文章采用了CIFAR10/100和ImageNet的经典参数，和VGG、ResNet、DenseNet的经典backbone，pruning method就是前文所述的6个。

需要特别注意的一点是，文章对 __traning budget__ 重点进行了测试。所谓的training budget也即当模型压缩后，进行重新训练的时候，是按照相同的epochs数训还是按照相同的FLOPs训。 <span style="color:blue">文章认为至少要允许进行同FLOPs的训练才有公平的比较可能。故文章中所有的重训网络都有两组对照，也即Scratch-E和Scratch-B，前者保证训练epochs数相同；后者保证训练FLOPs数相同</span>。
<span style="color:gray">[关于这一点，李翔在Arxiv Insights中推送了他们的网络压缩经验，佐证了小网络确实需要更多的epochs进行迭代，保持FLOPs基本不变，才能收敛到最好的结果。]</span>

另外值得一提的是，文章复现了6个benchmark，对于文章中没有提到训练参数的情况都重新跑了。并且<span style="color:red">diss了一些文章作者，表示复现的大网络性能明显好于文章中提到的性能，也即怀疑原文为了凸显pruning的性能压低了大网络的性能。</span>具体没跑，不得而知。

### 3.实验结果
1. [$L_1$-norm based channel pruning](https://www.researchgate.net/profile/Igor_Durdanovic/publication/307536925_Pruning_Filters_for_Efficient_ConvNets/links/585189cf08ae95fd8e168343/Pruning-Filters-for-Efficient-ConvNets.pdf)。
<center><img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/008/008_07.png?raw=true" width = 85% /></center>

2. [ThiNet](http://openaccess.thecvf.com/content_ICCV_2017/papers/Luo_ThiNet_A_Filter_ICCV_2017_paper.pdf)

<center><img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/008/008_08.png?raw=true" width = 85% /></center>

3. [regression based Feature Reconstruction](http://openaccess.thecvf.com/content_ICCV_2017/papers/He_Channel_Pruning_for_ICCV_2017_paper.pdf)

<center><img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/008/008_09.png?raw=true" width = 85% /></center>

4. [Network Slimming](http://openaccess.thecvf.com/content_ICCV_2017/papers/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.pdf)

<center><img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/008/008_10.png?raw=true" width = 85% /></center>

5. [Sparse Structure Seletction](https://arxiv.org/pdf/1707.01213.pdf)

<center><img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/008/008_11.png?raw=true" width = 85% /></center>

6. [Non-structured Weight Pruning](https://papers.nips.cc/paper/5784-learning-both-weights-and-connections-for-efficient-neural-network.pdf)

<center><img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/008/008_12.png?raw=true" width = 85% /></center>

其结论基本是统一的，就是或者Scratch-E，或者Scratch-B，能够持平或者超出利用原模型参数finetune的结果。

<span style="color:red">除了在ImageNet上进行实验6时，SongHan的原始方案高于Scratch-B/E</span>。

# Further Exploration

### 1.对结果的猜想
文章对实验验证进行了进一步的分析探究。也即解释既然传统的pruning没有能够真正选出“有意义”的最佳参数，为什么还是非常work。

主要的解释是，pruning操作天然的是非常高效和合理的网络结构搜索方案，能够得出更优化的小网络结构，因此做到了“模型压缩，精度不降”。

### 2.猜想的观察性验证
为了验证这个想法，作者测试了“胡乱压缩”和用验证有效的压缩策略压缩，带来的“网络参数效率”的不同，如下图所示：
<center><img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/008/008_13.png?raw=true" width = 85% /></center>

另外，作者发现经过优秀的压缩pipeline得到的小网络结构，其实是倾向于“有规律可循”的，也就是说其实压缩类似于一个天然的小网络设计过程，设计出来的东西符合一定的规律，而不是乱七八糟无章可循。
<center><img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/008/008_14.png?raw=true" width = 85% /></center>

值得注意的是，文章中给出了上图所示的element-wise pruning最终的结果。<span style="color:blue">可以看到随着迭代次数的增多和层数的加深，$3\times3$的卷积核渐渐退化成了对称的“十字架”的模样。</span>启发我们可以设计类似模样的“异形卷积核”或者“渐变卷积核”来减少网络的计算量同时保持网络性能。

### 3.猜想的设计性验证
为了进一步证明上述说法是成立的，作者利用上面发现的规律，基于VGG19和CIFRA100，设计了一些小网络进行测试。具体来说：
- 设计Guided Pruning系列网络，方式为基于Network Slimming方法先得到一个pruning好的网络，然后统计每个“layer stage”<span style="color:gray">[也即下图中的同颜色layer]</span>中的平均channel剪枝数。利用这个剪枝比例直接对VGG19进行重新剪枝得到的网络即为Guided Pruning Network
<center><img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/008/008_15.png?raw=true" width = 85% /></center>

- 设计Guided Sparsify系列网络，方式为基于Non-structured Weight Pruning方法观察pruning结束后卷积核的平均强度<span style="color:gray">[如上上图中Figure4所示]</span>。然后根据这个强度拟合一个大概规律直接搭建稀疏化的网络结构，称为Guided Sparsify Network。

- 设计transfered Guided系列对比网络。这是基于一个有趣的观察，<span style="color:blue">同一个系列网络经过pruning后会得到相似的规律，这样的规律有可迁移性。</span>故设计实验利用VGG16+CIFAR10进行训练+pruning+获得Guiding超参数，然后在VGG19上根据超参数进行网络搭建，并在CIFRA100上测试。

其结果如下图所示：
<center><img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/008/008_16.png?raw=true" width = 85% /></center>

可以看到，不但Guided系列的网络能够达到和pruning pipeline获得的最优网络相似的性能，连Transfered系列的网络也几部不分上下。

<span style="color:blue">这进一步有力佐证了pruning其实是一种提取网络结构信息，给定条件在大网络上搜索最佳小网络的方法。而不是给定大网络学习到的的参数信息，从其中搜索关键信息剔除无关信息的类似蒸馏的方法。</span>

### 4.本文结果的局限性
当然，作者也列举了<span style="color:red">本文提出方法的不可用场景</span>（也即传统pruning pipeline的不可替代性）

- 当已经给出了完备的pre-trained network，不用白不用，而且训练新网络的计算资源受限的时候。
- 当有必要把一个网络结构进行多个不同pruning rate的压缩的时候，用一个训好的大网络进行不同pruing rate的压缩操作最高效。

<span style="color:gray">[实际上，本文也不以提出解决方案为核心，而是以认知的更新为核心，所以这样的局限性并无关紧要。</span>

# Insights

### 1.文章的现实价值
本文的结论对现实训练还是有不少直接价值的：
1. 进行网络压缩的时候没必要一定训练一个完美的大网络。而且在经过剪枝得到小网络后，至少要重训一下网络再做出性能的论断。不能把finetune的结果就当做小网络性能的最终结果。
2. 可以通过pruning操作的结果观察网络结构当中起作用的主要部分，从而指导下一步的结构优化设计。例如十字架形状的卷积核、漏斗状的block等。
3. 对于得到的小网络，如果重训，需要设置对照组，保证和大网络的FLOPs几乎持平，也即需要用更多的epochs进行一组训练。
4. 其实channel pruning的作用和SE十分类似，只不过SE是以涨点为目的，pruning是以小型化为目的，但是细细思考手段和思路其实是十分相似的，也许可以相互借鉴。<span style="color:blue">例如利用下一个layer的权重甚至是下下个layer的改变量判断这层卷积核的重要性就是很值得尝试的命题</span>。

### 2.关于Pruning和NAS方向融合趋势的一点分析
不难看到，pruning和NAS在渐渐向着互相结合彼此参考的方向发展，这方面也可能会给我们后续的研究工作一些参考和突破的可能。

众所周知传统的NAS一般基于强化学习（Reinforcement Learning, RL）或者进化算法（Evolutionary Algorithm, EA）。在复杂度上显然超过pruning算法的几个数量级。

而近来：
- [CVPR2018：MorphNet: Fast & Simple Resource-Constrained Structure Learning of DeepNetworks](https://arxiv.org/pdf/1711.06798.pdf)一文当中引入了类似Network Slimming的结构进行网络自动化设计；
- [ICML2018：Efficient neural architecture search via parameter sharing](https://arxiv.org/pdf/1802.03268.pdf)等文章也利用大网络参数继承的方式提高NAS的效率。这显然是传统pruning思路应用到NAS方向的体现。

另一方面，我们之前也提到过，利用RL来进行pruning就在18年9月份左右大大火了一把，谷歌、腾讯、MIT（韩松组）等机构都推出了基于RL的网络自动剪枝算法/平台。我们还在[Arxiv Weekly 2018/004](https://github.com/luzhilin19951120/paperReadingMails/tree/master/2018/004)中推送过韩松组的相关工作，大家可以回去翻翻回忆一下。