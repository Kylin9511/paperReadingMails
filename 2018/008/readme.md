# Headline

大家好：

 <b><span style="color:green">这是2018年度第X篇Arxiv Weekly。</span> </b>

本文是 __模型压缩__ 方向的文章。

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

神经网络压缩作为实现神经网络小型化的主流技术之一，引发了广泛的关注。传统的神经网络压缩算法架构主要是三段式的： __训练大型网络->剪枝->finetune__。其核心是在剪枝过程中设计某种策略或者评价标准，使得不重要的参数被剪掉；而重要的参数被保留。从而实验压缩模型、acc基本不降低。

本文中通过实验发现了“反常识”的现象。首先我们通过对6个state-of-the-art的pruning算法的实验，发现<span style="color:blue">对训练好的大网络进行剪枝-finetune，其性能不如对剪枝后的网络进行重训（此时用random initialization）来得好</span>。故而如果剪枝算法中假设了目标网络的结构，那么之前复杂的pipeline没有太大的意义。

通过对多个剪枝算法、多个数据集、多个任务的交叉组合验证，我们发现了如下的现象/结论：
- 获取高效的小网络 __不需要__ 训练一个大规模的、参数/描述能力过冗余的网络
- 经过剪枝的小网络参数 __不需要__ 参考充分训练获得的原网络的对应参数。
- 一个经过剪枝的小网络是否高效，核心不在于保留了大网络中的哪些参数，而是剪枝后是什么结构。传统的剪枝pipeline之所以work，不是因为挑选和保留了大网络中足够重要的参数，而是这个挑选的过程耦合了一个网络结构搜索过程。 __换言之，pruning其实是一种NAS技术__。

# Keys

### 问题和背景

在传统上，我们认为神经网络的“over parameterization”是显然的和需要解决的。而pruning则是解决这个问题的最直接手段之一。

而在之前的观念里，以及之前pruning相关的文章中，训练出一个准确的，强大的大网络对剪枝后的网络很重要。<span style="color:red">因为小网络的参数基本是使用大网络剪枝后获得的参数进行finetune得到的，而这么做是因为主流认为大网络经过充分训练能够得到强有力的参数，值得继承</span>。

本文认为不然。

本文的核心想法是，pruning后小网络效果好，不依赖于大网络的参数，而依赖于小网络的结构本身。大网络参数在完成了“挑选小网络结构特性”的使命后就成为了累赘，没有太大保留价值。

为了证明这个观点，本文参考了六种最新的pruning算法/架构作为实验的背景。现列举如下：

1. [$L_1$-norm based channel pruning](https://www.researchgate.net/profile/Igor_Durdanovic/publication/307536925_Pruning_Filters_for_Efficient_ConvNets/links/585189cf08ae95fd8e168343/Pruning-Filters-for-Efficient-ConvNets.pdf)。
这篇文章是ICLR 2017的会议文章，也是最早提出做channel pruning的文章。<span style="color:gray">[关于weight/channel/layer pruning，可参考之前的[Arxiv Weekly 2018/004](https://github.com/luzhilin19951120/paperReadingMails/tree/master/2018/004)中的论述]</span>
本文作为channel pruning的开山之作，自然使用的是很简洁的方式。具体来说就是使得每一层当中$L_1$范数最小的某个比例 $p$ 的channel被剪枝。

2. [ThiNet](http://openaccess.thecvf.com/content_ICCV_2017/papers/Luo_ThiNet_A_Filter_ICCV_2017_paper.pdf)
这篇文章是ICCV 2017的会议文章，思路更进一步。同样按比例剪枝channel，但是criterion改为优先选择下一层的feature map中参数最小的channel对应的卷积核进行剪枝。从直觉上，这样会更加靠近我们pruning同时不影响acc性能的意图。
<center><img src="./008_01.png?raw=true" width = 55%" /></center>

3. [regression based Feature Reconstruction](http://openaccess.thecvf.com/content_ICCV_2017/papers/He_Channel_Pruning_for_ICCV_2017_paper.pdf)
这篇文章是ICCV 2017的会议文章，思路更更进一步。同样按比例剪枝channel，但是criterion改为尽量使得剪枝后下下层的feature map受影响最小。显然这又进一步试图保留原网络的性能。
<center><img src="./008_02.png?raw=true" width = 55%" /></center>

4. [Network Slimming](http://openaccess.thecvf.com/content_ICCV_2017/papers/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.pdf)
这篇文章是ICCV 2017的会议文章，<span style="color:blue">与前面文章不同，本文是在训练过程中自动产生每个层的pruning rate，所以结构不经过训练无法得到</span>。
实际上，本文采用的方式是引用BN层的scaler$\alpha$作为scaling factor，然后动态剪枝去除其中不重要的channel。因此是一种动态稀疏化网络的操作。
<center><img src="./008_03.png?raw=true" width = 85%" /></center>

5. [Sparse Structure Seletction](https://arxiv.org/pdf/1707.01213.pdf)
这篇文章是ECCV 2018会议文章。文章是Network Slimming的变体，仍然是生成了scaling factor指导pruning，只不过不是channel pruning而是block pruning。有些像[BlockDrop: Dynamic Inference Paths in Residual Networks](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wu_BlockDrop_Dynamic_Inference_CVPR_2018_paper.pdf)一文。
<center><img src="./008_04.png?raw=true" width = 85%" /></center>

6. [Non-structured Weight Pruning](https://papers.nips.cc/paper/5784-learning-both-weights-and-connections-for-efficient-neural-network.pdf)
这篇文章为NIPS 2015会议文章，韩松代表作之一，引用已经800+。核心方法是“啥也不说，就是硬干”，直接进行element-wise pruning，然后finetune。后续还有所谓的DSD等方法，其实就是反复硬干。至于pruning rate就是手工大量实验堆叠出一个最佳值即可。
<center><img src="./008_05.png?raw=true" width = 85%" /></center>

# Results

本文属于探究性质的论文，所以results的作用是探索和证明文章的观点。

文章把6个baseline网络分为两类，一类是所谓的Pre-defined Target Architecture；另一类是所谓的Automatically Discovered Target Architecture。其区别就是不训练能不能画出来大概的结构，如下图所示：
<center><img src="./008_06.png?raw=true" width = 45%" /></center>

在训练参数上，文章采用了CIFAR10/100和ImageNet的经典参数，和VGG、ResNet、DenseNet的经典backbone，pruning method就是前文所述的6个。

需要特别注意的一点是，文章对traning budget重点进行了测试。所谓的training budget也即当模型压缩后，进行重新训练的时候，是按照相同的epochs数训还是按照相同的FLOPs训。 <span style="color:blue">文章认为至少要允许进行同FLOPs的训练才有公平的比较可能。故文章中所有的重训网络都有两组对照，也即Scratch-E和Scratch-B，前者保证训练epochs数相同；后者保证训练FLOPs数相同</span>。

另外值得一提的是，文章复现了6个benchmark，对于文章中没有提到训练参数的情况都重新跑了。并且<span style="color:red">diss了一些文章作者，表示复现的大网络性能明显好于文章中提到的性能，也即怀疑原文为了凸显pruning的性能压低了大网络的性能。</span>具体没跑，不得而知。

1. [$L_1$-norm based channel pruning](https://www.researchgate.net/profile/Igor_Durdanovic/publication/307536925_Pruning_Filters_for_Efficient_ConvNets/links/585189cf08ae95fd8e168343/Pruning-Filters-for-Efficient-ConvNets.pdf)。
<center><img src="./008_07.png?raw=true" width = 85%" /></center>

2. [ThiNet](http://openaccess.thecvf.com/content_ICCV_2017/papers/Luo_ThiNet_A_Filter_ICCV_2017_paper.pdf)

<center><img src="./008_08.png?raw=true" width = 85%" /></center>

3. [regression based Feature Reconstruction](http://openaccess.thecvf.com/content_ICCV_2017/papers/He_Channel_Pruning_for_ICCV_2017_paper.pdf)

<center><img src="./008_09.png?raw=true" width = 85%" /></center>

4. [Network Slimming](http://openaccess.thecvf.com/content_ICCV_2017/papers/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.pdf)

<center><img src="./008_10.png?raw=true" width = 85%" /></center>

5. [Sparse Structure Seletction](https://arxiv.org/pdf/1707.01213.pdf)

<center><img src="./008_11.png?raw=true" width = 85%" /></center>

6. [Non-structured Weight Pruning](https://papers.nips.cc/paper/5784-learning-both-weights-and-connections-for-efficient-neural-network.pdf)

<center><img src="./008_12.png?raw=true" width = 85%" /></center>

其结论基本是统一的，就是或者Scratch-E，或者Scratch-B，能够持平或者超出利用原模型参数finetune的结果。

# Further Exploration

文章对实验验证进行了进一步的分析探究。也即解释既然传统的pruning没有能够真正选出“有意义”的最佳参数，为什么还是非常work。

主要的解释是，pruning操作天然的是非常高效和合理的网络结构搜索方案，能够得出更优化的小网络结构，因此做到了“模型压缩，精度不降”。

为了验证这个想法，作者测试了“胡乱压缩”和用验证有效的压缩策略压缩，带来的“网络参数效率”的不同，如下图所示：
<center><img src="./008_13.png?raw=true" width = 85%" /></center>

另外，作者发现经过优秀的压缩pipeline得到的小网络结构，其实是倾向于“有规律可循”的，也就是说其实压缩类似于一个天然的小网络设计过程，设计出来的东西符合一定的规律，而不是乱七八糟无章可循。
<center><img src="./008_14.png?raw=true" width = 85%" /></center>

# Insights

May the force be with you

-  <span style="color:red">May the force be with you</span> <span style="color:grey">[May the force be with you]</span>

<center><img src="./001_01.png?raw=true" width = "80%" /></center>
