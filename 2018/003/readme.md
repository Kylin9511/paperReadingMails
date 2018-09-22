# Headline

大家好：

 <b><span style="color:green">这是2018年度第3篇Arxiv Weekly。</span> </b>

本文是 __网络压缩__ 方向的文章。

# Highlight

> __通过channel wise的网络软剪枝保留更多网络描述能力，从而在同样的pruning rate下取得更高的performance！__

# Information

### Title
_Soft Filter Pruning for Accelerating Deep Convolutional Neural Networks_

### Link
https://www.ijcai.org/proceedings/2018/0309.pdf 

### Codes
https://github.com/he-y/soft-filter-pruning

### Accepted by
2018年度人工智能国际联合大会（International Joint Conference on Artificial Intelligence，IJCAI 2018）<span style="color:grey">[为人工智能方向A类会]</span>

### Source

- 南方科技大学（Southern University of Science and Technology）
- 悉尼科技大学（University of Technology Sydney）
- 复旦大学（Fudan University）

# Introduction

本文提出了一种通过pruning加速inference的方法：卷积核软剪枝（<span style="color:brown">soft filter pruning ,SFP</span>），并且在pruning后网络还能进一步训练。也就是说，__pruning的过程是伴随着训练进行的__。

SFP有两个主要的优势：
1. <span style="color:blue">保留更多模型性能</span>。由于pruning后还能够训练，SFP能够保留更多模型性能，学习到更好的特征。
2. <span style="color:blue">对pre-trained model的依赖性更低</span>。SFP能够同时完成training和pruning，而不是像之前的pruning一样需要在well pre-trained model上进行。

# Keys

### 1.问题描述
传统的filter pruning技术有两个“流派”，分别存在一些问题。
- __细粒度剪枝（Fine-grained pruning）__，指元素级的pruning，将值较小的或者经过测试不重要的elements置零。
- __粗粒度/结构化剪枝（Coarse-grained / structured pruning）__，指成块地整体剪枝。最常见的就是channel-wise pruning，也即一次剪掉一个channel。另外也有一次减掉一行、一列、一个小块这样的形式。

显然细粒度剪枝的可操作性更大，因此通常能够实现同样performance下更高的压缩率。

然而由于计算架构的优化现状，粗粒度剪枝（尤其是channel-wise pruning）相比细粒度剪枝而言计算效率更高。<span style="color:grey">[例如Han、Guo等人提出的element-wise pruning会把模型弄得“千疮百孔”，导致需要稀疏编码和解码，并且针对完整filter的通用优化计算全会失效。]</span>

而本文提出的SFP属于粗粒度剪枝中的channel-wise pruning的最新工作。相比于前序工作，主要解决了如下两个问题：
- <span style="color:red">model capacity reduction</span> 传统的channel-wise pruning实质上是hard pruning，也即设定压缩率把完整的模型打残，让残废的模型自己往终点爬。<span style="color:grey">[也即模型的描述能力受到了断崖式损伤]</span>
而SFP则是把模型实时打残实时上药，而且看不行了就扶一下。<span style="color:grey">[在训练过程中保留模型的全部描述能力，只不过限制描述的效果]</span>
因此虽然大家最后都是残废了，但是SFP的方式打残的模型能跑得更远。<span style="color:grey">[如下图]</span>

<center><img src="./003_01.png?raw=true" width = "50%" /></center>

- <span style="color:red">dependence on pre-trained models</span>。一般的hard pruning需要对完整模型的充分训练结果作为输入。而SFP由于是一边训练一边pruning，对pre-trained model没有需求，也自然不会被pre-trained model的性能影响。

### 2.SFP思路和算法

SFP的关键思路可以总结如下：
1. 整个网络设置一个压缩率$P\%$，目标是把每个layer的filter数都压到原本的(1-P%)倍。
2. 于是在正常的训练每个epoch的训练后，都对每层各个filter中$l_2$范数最小的$P\%$个强行置零。<span style="color:blue">但是仍然允许它们参与下一轮的训练和参数更新（称为Reconstruction）。这样一来，其中有一些在后续迭代中较为重要的filter还有机会“重出江湖”</span>。<span style="color:grey">[如下图]</span>

<center><img src="./003_02.png?raw=true" width = "90%" /></center>

SFP的算法伪代码如下图所示

<center><img src="./003_03.png?raw=true" width = "45%" /></center>

其中

<center><img src="./003_04.png?raw=true" width = "50%" /></center>

p.s.在整个网络训练结束后，将最后一轮SFP中pruning掉的filter直接干掉就能得到实际的小模型了。这里会存在一个前后对应的问题，而<span style="color:red">这个问题在ResNet结构的网络中会变得尤其突出，甚至影响到了SFP的实际pruning rate和效果对比可靠性</span>。

### 3. SFP的压缩效率理论值
我们设第i个layer输出的feature map size为$C_i \times W_i \times H_i$，设第i个layer中filter的pruning rate为$P_i$，显然有：
在pruning之前计算第i+1层的feature map的计算量为：
$$Cal=N_{i+1}\times N_i \times k^2 \times H_{i+1} \times W_{i+1}$$
而经过SPF后的计算量为：
$$Cal=(1-P_{i+1})N_{i+1}\times (1-P_i)N_i \times k^2 \times H_{i+1} \times W_{i+1}$$
也即这一层的压缩率为：
$$1-(1-P_i)(1-P_{i+1})$$
如果整体SPF只有一个pruning rate $P$，则整体理论压缩率为
$$Ratio=1-(1-p)^2=2p-p^2$$
# Results

在CIFAR上超过了很多传统的pruning算法。

在ImageNet+ResNet的标准benchmark上，基本能稳住Acc掉1个点的时候，prune掉40%左右的flops。
其中最好的一个case是ImageNet+ResNet101 ，在prune掉42.2%的flops时，反而使得top1/top5 Acc上升了0.14/0.2个点。

结果大表如下

<center>
<img src="./003_05.png?raw=true" width = "80%" />

CIFAR上几种方法对不同深度的ResNet进行pruning的结果对比
</center>

<center>
<img src="./003_06.png?raw=true" width = "80%" />

ImageNet上几种方法对不同深度的ResNet进行pruning的结果对比
</center>

__我们自己也有相应的复现结果，总体来看是不错的，但是对比上还需要进一步的实验。__

# Insights

可以看到，这里的“soft” filter pruning，主要“软”在不在最开始就把网络弄小，也不在最终定型了之后一股脑儿压缩，而是一边训练一边约束。

比较有意思的是，在约束了之后，模型的结构和filter真正的层数是没有改变的，被prune掉的filter后面还有“复活”的希望。因此pruning操作对模型的描述能力的损伤显然是比硬pruning小的。

不过这样的pruning方法在ResNet结构上会面临两个支路挑选的channel不能重合的问题，需要进一步计算实际的pruning rate。