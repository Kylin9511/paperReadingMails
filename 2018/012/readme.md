[TOC]

# Headline

大家好：

 <b><span style="color:green">这是2018年度第12篇Arxiv Weekly。</span> </b>

本文是 __训练技巧__ 方向的文章。

# Highlight

> __本文综述了一系列利用正则化和增强，在cifar/imagenet上对抗overfitting的技术，其中state-of-the-art的部分对一些backbone的涨点较有实用价值__

# Information

## title

- A serial of works fight against overfitting

## links

- paper list provided later

## source

- Momenta Rocker

# Preview of Overfitting

## 成因
由于模型空间远远大于数据空间。也即参数过度冗余，训练数据不足，独立方程数目远少于未知数的数目，因此求出来的未知数可以有很多组解。

## 解决手段
- __enlarge training dataset__
原理是显然的，但是有烧钱的问题。而且随着dataset的增长，计算等资源也会同步加快消耗。不能作为解决问题的唯一手段。
- __cross validation & early stop__
不能涨点，只能准确预测模型性能，选出最好模型并加快迭代。
- __regularization related__
正则化是防止overfitting的核心技术，通过对训练引入某种先验信息约束防止拟合走向错误的极端。
    - _weight decay_：在正常的loss后面附加参数的范数项，通过假定模型参数具有高斯/拉布拉斯分布，使得参数空间极大压缩，许多奇异的overfit点就被抹去了。
    - _normalization_：一种如今看来十分令人无奈的技术，Hinton的心理阴影。对网络的训练有至关重要而难以完美解释的作用。实际上normalization也存在一系列的技术，BN；LN；GN；IBN；SN等，可以专门分享一个小时。
- __data related__
许多技术对数据进行操作，期望获得更合理的分布或者变相增加数据集。
    - _data augmentation_：变相增大数据集。 
    - _data shuffle_：常规操作，通过数据集的shuffle方式输入高度有序的训练样本后，模型对输入数据顺序进行拟合（一种特殊的过拟合）
    - _label smoothing_：为了防止label的over confidence，将gt中true的confidence改为0.9，false的confidence改为0.1。

# Regularization

## 1.Dropout & DropConnect

### paperlink

- __dropout:__
[G. E. Hinton, N. Srivastava, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov. Improving neural networks by preventing co-adaptation of feature detectors. CoRR, abs/1207.0580, 2012.](https://arxiv.org/pdf/1207.0580.pdf?utm_content=buffer3e047&utm_medium=social&utm_source=twitter.com&utm_campaign=buffer)

- __dropconnect:__
[Li Wan, Matthew Zeiler, Sixin Zhang, Yann L Cun, and Rob Fergus. Regularization of neural networks using dropconnect. ICML, 2013.](http://proceedings.mlr.press/v28/wan13.pdf)

### keys

dropout是这个领域最早的工作，核心思想是根据给定的超参数dropout rate，在训练时将FC层的输出进行elementwise的随机置零。

<span style="color:red">dropout压制了参数的过冗余，并使网络能够学习到更鲁棒、更高级的信息，而不是对无意义信息过拟合。</span>

也有人认为，<span style="color:blue">dropout本质上是一种ensemble技术，每个模型都基于前面模型的基础train一次，最终的模型是由一系列共享参数的模型演化生成。</span>

$$ y = mask \odot Activate(Wx) $$

而dropconnect是对dropout的自然扩展，直接对FC层的参数W进行elementwise的随机置零，等价于对FC层前后神经元的connection进行dropping操作，也因此得名。

$$ y = Activate((mask \odot W) x) $$

### results

在Alexnet等早期网络上，dropout有奇效，而dropconnect能够取得比dropout优化一些的点数。

<center>
<img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/012/012_01.png?raw=true" width = "55%" />
</center>

## 2.Maxout

### paperlink

- [Goodfellow I J, Warde-Farley D, Mirza M, et al. Maxout networks[J]. arXiv preprint arXiv:1302.4389, 2013.](https://arxiv.org/pdf/1302.4389.pdf)

### keys

maxout是13年Goodfellow在小数据集上state-of-the-art的工作。

maxout的操作十分朴素，对原本的FC层进行K倍增操作，然后从倍增后得到的K个输出中选择一个最大的作为final output。
<center>
<img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/012/012_08.png?raw=true" width = "55%" />
</center>

<span style="color:red">实际上maxout是一种可学习的激活函数</span>，如果W的规模足够大，K=2的maxout就能够拟合任意的连续函数。
<center>
<img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/012/012_09.png?raw=true" width = "55%" />
</center>
<center>
<img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/012/012_10.png?raw=true" width = "55%" />
</center>

可能是由于这样的激活操作过于consuming，而且后期网络规模越来越大，对每个部件的效率要求越来越高，目前似乎maxout操作已经没什么人用

## 3. Spatial Dropout

### paperlink

- [Jonathan Tompson, Ross Goroshin, Arjun Jain, Yann LeCun, and Christoph Bregler. Efficient object localization using convolutional networks. In CVPR, 2015.](https://arxiv.org/pdf/1411.4280.pdf)

### keys

显然dropout机制本身无法直接很好地作用于卷积层。其中主要有两个原因。
1. 卷积层的冗余度没有FC层那么大，尤其对于前期卷积层暴力添加dropout有导致欠拟合掉点的风险，超参数不好控制。
2. 卷积层的输出是高度耦合相关的，一个信息可能包含在一个区域当中。<span style="color:red">这样的耦合使得单纯添加传统的dropout不能实现去除网络冗余表达能力的意图</span>。(这样的相关性主要存在于同一个feature map的相邻pixel间)
<center>
<img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/012/012_05.png?raw=true" width = "55%" />
</center>

因此一个很容易想到的方法是，对于后期的已经高度抽象的卷积层，进行channelwise的随机置零，也即spatial dropout。
<center>
<img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/012/012_06.png?raw=true" width = "55%" />
</center>

本文中核心问题是检测，spatial dropout作为一个技术细节被提及，具体的result不是本文的重点。

<center>
<img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/012/012_07.png?raw=true" width = "90%" />
</center>

## 4. DropPath

### paperlink

- [Larsson G, Maire M, Shakhnarovich G. Fractalnet: Ultra-deep neural networks without residuals[J]. arXiv preprint arXiv:1605.07648, 2016.](https://arxiv.org/pdf/1605.07648.pdf)

### keys
实际上，随着inception和resnet的大热，许多存在skip connection、multi path的网络结构被设计出来。

<center>
<img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/012/012_02.png?raw=true" width = "80%" />
</center>

这意味着，原本作用于FC的dropout思想又多了一个可以施展的维度：<span style="color:red">将multi path network中的路径以一定的概率drop掉，即为droppath。

<center>
<img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/012/012_03.png?raw=true" width = "80%" />
</center>

### result

<center>
<img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/012/012_04.png?raw=true" width = "80%" />
</center>

可以看到对于作者提出的复杂多路并行网络，dropath是一种makesense的正则方式，在许多情况下能够涨点。

## 5. ScheduledDropPath

### paperlink

- [Zoph B, Vasudevan V, Shlens J, et al. Learning transferable architectures for scalable image recognition[J]. arXiv preprint arXiv:1707.07012, 2017, 2(6).](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zoph_Learning_Transferable_Architectures_CVPR_2018_paper.pdf)

### keys

本文为google brain在NAS方向上的工作。文章在实验中发现了一件有趣的事情，也即利用DropPath本身无法给NasNet带来任何好处；<span style="color:red">但是如果随着训练逐渐增加drop probability，则能够极大提升NasNet的performance</span>。作者将这种trick命名为ScheduledDropPath

## 6. StochasticDepth (RandomDrop)

### paperlink

- [Huang G, Sun Y, Liu Z, et al. Deep networks with stochastic depth[C] European Conference on Computer Vision. Springer, Cham, 2016: 646-661.](https://arxiv.org/pdf/1603.09382.pdf)

### keys
显然上面的两种思路只能适用于含有inception结构的网络或者是ResNeXt结构的网络。对于纯粹的ResNet并不实用。

故而基本同时，应用于超深（超过1000层）的ResNet网络的dropping思路被提出，也即StochasticDepth，随机深度网络。

<center>
<img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/012/012_11.png?raw=true" width = "80%" />
</center>

核心的想法非常直白，就是对于极深的ResNet网络，<span style="color:red">在训练的时候所有的Residual Block被以一定的概率drop掉；而在推理的时候则保持网络的完整性，只是乘以本层drop probability的均值以校准网络</span>。

于是被random drop的网络就相当于有了stochastic depth（因此这个工作一般用这两个名字简称），从而给训练过程中的特征抽象引入随机性，进一步对抗参数量巨大的深度网络overfitting的问题。

本文也<span style="color:blue">采用了线性变化的drop rate，相当于是一种“空间域的schedule”，重要的浅层block以更大概率保留，越深层的block被drop的概率越大</span>。

$$
P_l = 1 - \frac{l}{L}(1-P_L)
$$

其中 $l$ 为resdual block的index；$L$为总block数；$P_l$为第 $l$ 个block的residual部分被drop掉的概率；$P_L$为给定的drop rate，是超参数。

### results

重要的参数：$P_L=0.5$，cifar中使用Kaiming原文中的ResNet110结构。

<center>
<img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/012/012_12.png?raw=true" width = "80%" />
</center>

可以看到，stochastic depth确实有不错的涨点，达到了当时的state-of-the-art结果。比较重要的是，对于深度的resnet网络这是一种通用的白捞好处的正则化手段。

## 7. Shake-Shake

### paperlink

- [Gastaldi X. Shake-shake regularization[J]. arXiv preprint arXiv:1705.07485, 2017.](https://arxiv.org/pdf/1705.07485.pdf)

### keys

shake-shake是个奇异的名字（不得不让我们想起百度的paddle-paddle），不过不可否认它的出现代表了cifar的new state-of-the-art。

shake-shake做了一件之前很少有人做过的事情，即<span style="color:red">在前传和反传引入不同的随机性。作者将这波奇异的操作命名为颤抖（shake）</span>，大概是设想到一个前传反传包含不同随机性的网络，在训练中会不断的shake-shake and shake again。

<center>
<img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/012/012_13.png?raw=true" width = "80%" />
</center>

具体来说，shake-shake必须应用在类似resNeXt的结构上，也即存在多路同构的residual path。

训练过程中，在前传的时候两路residual path会被随机加权后合并，且权重的和为1；反传的时候也一样，但是加权用的随机变量是重新生成的。

在推理的时候，多路同构的residual path会同权重接入网络，且权重为随机权重变量的均值（为了校准网络）。

### results

<center>
<img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/012/012_14.png?raw=true" width = "80%" />
</center>

对任意的training pass(forward / backward)
- shake 代表独立随机采样
- keep 代表采用和另一个方向相同的权重
- even 代表不采样，平均分配权重

另外：
- Image 代表对每个数据独立采样
- Batch 代表对整个batch仅采样一次，数据间公用权重

可见，cifar10上最佳的方案为shake-shake-image，这也是本文名称的由来。

shake-shake主要是一种正则化的方式，但是也提出了自己独立的backbone，故而可以和其他的backbone的性能进行对比。可以看到在点数上确实是state-of-the-art的。而且所需要的网络层数只有26层，可见并不是靠无脑增大网络实现突破。
<center>
<img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/012/012_15.png?raw=true" width = "80%" />
</center>

## 8. ShakeDrop

### paperlink

- [Yamada Y, Iwamura M, Kise K. ShakeDrop regularization[J]. arXiv preprint arXiv:1802.02375, 2018.](https://arxiv.org/pdf/1802.02375.pdf)

- [DeVries T, Taylor G W. Dataset augmentation in feature space[J]. arXiv preprint arXiv:1702.05538, 2017.
](https://arxiv.org/pdf/1702.05538.pdf)

### keys

shake-drop解决了如何将shake-shake的训练方式运用在更common的backbone的问题。

首先说明，如果强行将shake-shake中的方法用在single branch上，会导致训练的崩溃。希望运用这样的方法首先需要分析shake-shake work的原因和single branch shake-shake崩溃的原因。

> 今年ICLR workshop中的文章 _Dataset augmentation in feature space_ 提出可以通过对网络中feature进行插值或者噪声附加，实现全网络的数据增强。
> 
> 而本文提出，shake-shake之所以work，可以理解为 __前传的时候通过随机权重合并两个同构支路的feature，在网络的每一个地方都完成了恰如其分的增强__。
> 
> 而 __反传的时候的随机权重本质上是一种扰乱__，使得前传时候两个同构支路的随机合并有意义。（试想如果正常反传，随着训练进行两个支路理论上是完全趋同的，就达不到增强的效果）
> 
> （实际上如果网络能够抽象图像的根本性特征，那么同一张图无论怎么引入两个支路上分布的随机性，这些共同的根本性特征是不变的。shake-shake其实利用了这一点，在网络的各个层次进行增强操作，从而一举突破到state-of-the-art。）
> 
> 以上特性使我们需要保留的，下面需要知道single branch shake-shake崩溃的原因以避免之
> 
> 显然在shake-shake的训练过程中，相比于正常的网络，反传的梯度乘上了如下的常数。
> $$ \frac{\beta}{\alpha} \;;\; \frac{1 - \beta}{1 - \alpha}$$
> __如果$\alpha$十分靠近0或者1，则上两个式子会有一个奇异。这时候另外一个还是完好的，能一定程度挽回性能的损失。__
> 
> 显然single branch shake-shake就不存在这样的机制。故而会导致bug of stabilization

综上，文章作者给出了自己的single branch shake-shake方案，也即所谓的shake drop

<center>
<img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/012/012_16.png?raw=true" width = "80%" />
</center>

从上图能非常清楚地看到四种方式的对比

<center>
<img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/012/012_17.png?raw=true" width = "55%" />
</center>

上式为shake drop的迭代公式，可以看到，这个公式是shake-shake和random drop的结合，且在极限情况下：

<center>
<img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/012/012_20.png?raw=true" width = "55%" />
</center>

因此，shake-drop解决上述问题的手段如下：
1. 通过在residual path上附加噪声实现shaking augment，于是摆脱了必须存在同构分枝的约束。
2. 引用random drop的思想，赋予unstable的反传一个概率，使得整体网络变为shake-shake和vanilla的加权和，其中权重（即概率）是一个超参。这样结构的stabilization就大大提高了。

### results

在cifar10上结果如下
<center>
<img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/012/012_21.png?raw=true" width = "80%" />
</center>

这里通过PyramidNet+shakedrop获取了2.67的state-of-the-art结果，但是其实结果的可比较性比较差。

在ImageNet上结果如下
<center>
<img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/012/012_19.png?raw=true" width = "80%" />
</center>

可以看到基本也能够取得一定的好处。总结来看shake-drop当中提到了很多有意义的分析，对更深入理解shake-shake很有帮助。另外也找到了在通用网络中使用shake-shake策略的方式。

## 9. Dropblock

### paperlink

- [Ghiasi G, Lin T Y, Le Q V. DropBlock: A regularization method for convolutional networks[C]//Advances in Neural Information Processing Systems. 2018: 10748-10758.](http://papers.nips.cc/paper/8271-dropblock-a-regularization-method-for-convolutional-networks.pdf)

### keys

dropblock为另一条支线上的成果。继承了dropout-spatial dropout的脉络，可以视为dropout在卷积层上的泛化和推广，整体思路非常简单。

<center>
<img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/012/012_22.png?raw=true" width = "80%" />
</center>

由于前述原因，对卷积层需要考虑整个feature map上的块状/整体dropping才有意义。

dropblock用如下机制生成一个随机的块状drop area，从而在通用的卷积神经网络中make sense地插入dropping操作。

<center>
<img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/012/012_23.png?raw=true" width = "80%" />
</center>

### results

<center>
<img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/012/012_24.png?raw=true" width = "80%" />
</center>

可以看到从文中的实验结果来看，dropblock用于resnet有十分好的效果。事实上从复现结果看，还没有得到一个稳定的产出方式，距离paper report的结果有一定差距。

# Data Augment

## 1. Cutout

### paperlink

- [DeVries T, Taylor G W. Improved regularization of convolutional neural networks with cutout[J]. arXiv preprint arXiv:1708.04552, 2017.](https://arxiv.org/pdf/1708.04552.pdf)

### keys

一张图就足以说明cutout做的所有事情
<center>
<img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/012/012_25.png?raw=true" width = "80%" />
</center>

cutout的insights来自检测中常遇到的遮挡问题，思路就是要打败敌人首先要了解敌人，人为生成大量含有恰当遮挡的图片交付给网络。这样真正遇到遮挡的时候网络会像打过疫苗一样，对这种情况产生很强的抗性。

令人惊喜的是，通过恰当遮挡原图的一部分，网络更倾向于深入学习图片当中的语义信息。于是cutout不再是纯粹抵抗遮挡的功能，而是起到数据增强的效果。

### results

实验表明cutout有两个关键之处：
1. 必须仔细寻找和精确设定遮挡区域的大小。对于cifar10而言是16个pixel为边长的方形区域；对于cifar100而言则变为8个pixel。
2. 必须允许遮挡残缺的存在，也即允许遮挡区域的生成在图的外面。

注意以上tricks后得到如下的结果：
<center>
<img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/012/012_26.png?raw=true" width = "80%" />
</center>

可见shake-shake + cutout达到了2.56的cifar10 state-of-the-art，超过了上面的shakedrop。

经过综合的实验发现，在cifar上文章给出的参数确实有奇效，作用于很多backbone都能稳定涨点。也许cutout能成为小图片数据任务的关键增强技术之一。

## 2. Mixup

### paperlink

- [Zhang H, Cisse M, Dauphin Y N, et al. mixup: Beyond empirical risk minimization[J]. arXiv preprint arXiv:1710.09412, 2017.](https://arxiv.org/pdf/1710.09412.pdf)

### keys

本文描述了一种直观上十分魔幻的增强技术，同时对data和label进行线性组合后，作为新的增强后数据输入网络训练。

<center>
<img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/012/012_32.png?raw=true" width = "80%" />
</center>

### results

可以看到在cifar上，mix up能够实现明显的涨点。
<center>
<img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/012/012_33.png?raw=true" width = "80%" />
</center>

另外，从文中的实验还能看到，mix up能够赋予网络更强的对抗label corruption和adversarial attack的能力，且能让GANs的训练更佳稳定。

## 3. Auto Augment

### paperlink

- [Cubuk E D, Zoph B, Mane D, et al. AutoAugment: Learning Augmentation Policies from Data[J]. arXiv preprint arXiv:1805.09501, 2018.](https://arxiv.org/pdf/1805.09501.pdf)

### keys

AA是非常强大的工具。

我们之前常见的数据增强都遵循一定的套路，这些套路是经典文章中使用的方式，具有较好的效果。例如imagenet中的RandomResizedCrop + RandomHorizontalFlip等。但是并没有人证明这就是最好的策略。

本文的作者提出利用RNN来对不同的数据集进行数据增强的搜索，以获得针对性的最佳策略。（Imagenet上的一种好策略如下）

<center>
<img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/012/012_27.png?raw=true" width = "80%" />
</center>

实际上，上图中的策略是从下设的搜索空间里，利用RL搜索获得的。
<center>
<img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/012/012_28.png?raw=true" width = "80%" />
</center>

通过RL（分组和搜索的细节不表），我们最终能对每个数据集获得25组增强策略；每组策略分为两个Operation，增强的时候依次进行即可。

例如在cifar上搜索获得的一组最佳策略如下：
<center>
<img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/012/012_29.png?raw=true" width = "80%" />
</center>

注意，在每次训练的时候，对每个batch的数据，都要随机抽取一组策略进行增强操作。

### results

在cifar10上的结果如下，这次是和pyramid-shakedrop结合。如果复现成果的话，基本已经把cifar10刷爆了。
<center>
<img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/012/012_30.png?raw=true" width = "80%" />
</center>

在imagenet上的结果如下，涨点也非常明显。
<center>
<img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/012/012_31.png?raw=true" width = "80%" />
</center>

# Insights

实际上，网络正则化和数据增强对训练结果的影响堪比backbone的结构设计。现在综合来看存在两个值得思考的问题：

1. 数据增强和正则化是否存在相似的数学本质？能否互相借鉴和融合？例如shake-drop中提出了下表，可以说是一个不错的开始
<center>
<img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/012/012_compare.png?raw=true" width = "80%" />
</center>

2. 能否用一种方式，给出现有如此多技术的“完美”融合，或者针对不同给出一个“最佳组合”？