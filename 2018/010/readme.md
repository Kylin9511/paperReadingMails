# Headline

大家好：

 <b><span style="color:green">这是2018年度第10篇Arxiv Weekly。</span> </b>

本文是 __模型架构__ 方向的文章。

# Highlight

> __提出AmoebaNet，最近的ImageNet Backbone，达到83.9%/96.6%的t1/t5 acc__

# Information

### Title
_Regularized Evolution for Image Classifier Architecture Search_

### Link
https://arxiv.org/pdf/1802.01548.pdf

### Source

- 谷歌（Google Brain）<span style="color:gray">[Quoc V.Le 组]</span>

# Introduction

在手工设计图像分类神经网络的过程中，得到了许多成果。这些成果中具有指导性的部分启发了NAS技术的诞生。之前的主流NAS是通过RL进行搜索空间内的网络结构遍历实现的。

另一方面，通过启发式算法中的进化算法进行分类网络生成的努力从未停止，但是生成的网络结构都不能和手工设计的优秀网络相比。

本文通过进化算法给出了up-to-date的分类网络AmoebaNet。 __首次，evolution aided NAS outperform了传统手工设计网络__。在ImageNet上，大规模的AmoebaNet-A（469M）实现了83.9%/96.6%的t1/t5 acc；而<span style="color:blue">comparable规模的AmoebaNet-A（87M）也达到了82.8%/96.1%的t1/t5 acc</span>。

通过控制变量实验，我们证明了本文提出的进化算法相比于RL算法复杂度更低，而且能够更早给出可靠的网络结构。故而，本文提出的进化算法是更简单高效的new NAS technology。

# Keys

### 1. 本文的亮点

经典的进化学习方法（ _neuro-evolution of topologies_）并不能生成足够优秀的网络结构。为了改变这样的情况，本文提出了两个改进点，使得算法搜索的结果得到了革新：
- Adding ages。文章对之前最好的 _tournament selection evolutionary_ 算法进行了改变，__使得tournament的结果更偏向younger genotypes__。可以看到这样的策略改变带来了非常关键的性能提升。
- Using NASNet search space。文章采用了最简化的变异（mutation）方法，使得整个进化过程能够被约束在NASNet的搜索空间中，等于更好地利用先验网络知识。

<center>
<img src="./010_01.png?raw=true" width = "50%" />
</center>

文章的进化算法如上图所示。

### 2. 对NASNet搜索空间的解释

<center>
<img src="./010_02.png?raw=true" width = "50%" />
</center>

上图左侧可以看到NASNet搜索空间的共性结构，本文中的AmoebaNet在这个空间中生成，自然沿用了同样的结构。

其中nomal cell全部使用skip connection的方式连接；reduction cell和而normal cell只有两个不同：其一是训练独立故具体样式不同；其二是reduction cell的末尾会做一次stride=2的downsample，使得每个feature map大小变为原本的1/4。

启发式的网络生成采用“ __五步断肠__”的原则，也即有且只有5次可选择的操作，每次操作合并两个结点。5次操作完成后，所有叶子节点被强行concatenate形成输出。

<span style="color:blue">搜索结束后会留下两个超参待定。分别是每一层中normal cell的数量N和初始输入的feature map的大小F。这两个参数给scale up整个网络留下了接口。

最后解释一下search space提供的所有候选操作：
- None
- 3 * 3, 5 * 5, 7 * 7 sep（separable conv，即point wise+element wise的组合）
- 3 * 3 avg/max pooling
- 3 * 3 dilated（空洞） sep conv
- 1 * 7 + 7 * 1 conv

# Results

### 1. 搜索出的结果

<center>
<img src="./010_03.png?raw=true" width = "80%" />
</center>

上图为本文中用于性能测试的AmoebaNet-A

<center>
<img src="./010_04.png?raw=true" width = "80%" />
</center>

上图为另外三种可行的结构B、C、D，在后续的工作中会用到

### 2. 网络的性能表现

<center>
<img src="./010_05.png?raw=true" width = "80%" />
</center>

如图，确实达到了当时ImageNet上up-to-date的水平。

# Insights

基于启发式学习的网络搜索新工作。

本推送中重点介绍了网络搜索的搜索空间和搜索结果。

但是文章的实际价值集中在更迅速、更高效地新型NAS技术。利用启发式方法想来是高效率的代名词，正如利用RL是低效率烧卡的代名词一样。如果对文章中启发式算法设计细节感兴趣，可以进一步参照原文。