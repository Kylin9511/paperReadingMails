# Headline

大家好：

 <b><span style="color:green">这是2018年度第7篇Arxiv Weekly。</span> </b>

本文是 __人脸数据__ 方向的文章。

# Highlight

> __利用类似聚类的思想，依托已有的标注数据，将海量无标注数据自动标注后引入训练，提升训练效果__

# Information

### Title
_Consensus-Driven Propagation in Massive Unlabeled Data for Face Recognition_

### Link
https://arxiv.org/pdf/1809.01407.pdf

### Codes
https://github.com/XiaohangZhan/cdp/

### Source

- 香港中文大学-商汤联合实验室（CUHK - SenseTime Joint Lab, The Chinese University of Hong Kong）
- 商汤科技（SenseTime Group Limited）
- 南洋理工大学（Nanyang Technological University ）

# Introduction

近年来，随着模型性能的飞跃和大量标注数据的采集，人脸识别的performance有了很大的提高。然而，进一步倍增人脸的数据集变得非常困难<span style="color:gray">[基数太大，增加得少没用，增加得多成本高昂]</span>

本文发现未标注的人脸数据也能像标注的数据一样发挥价值。为了通过实验说明这个问题，本文模拟了真实世界中简易图像采集的场景，也即 __在无限制的场景下采集大量未标注的数据，且和我们已经获取的标注数据没有直接的联系__。<span style="color:red">显然这个场景的图像获取比较而言几乎是没有成本的，同样在传统意义上也是没有用的</span>。本文的思路就是如何“废物利用”，“变废为宝”。

本文主要采用的方式是自底向上建立一个relational graph，来衡量自由图像和标注图像代表之间的语义相关度，从而将已有的label最可靠地赋予自由图像。文章提出u了

Our main in-
sight is that although the class information is not available, we can still
faithfully approximate these semantic relationships by constructing a re-
lational graph in a bottom-up manner. We propose Consensus-Driven
Propagation (CDP) to tackle this challenging problem with two mod-
ules, the “committee” and the “mediator”, which select positive face
pairs robustly by carefully aggregating multi-view information. Exten-
sive experiments validate the effectiveness of both modules to discard
outliers and mine hard positives. With CDP, we achieve a compelling
accuracy of 78.18% on MegaFace identification challenge by using only
9% of the labels, comparing to 61.78% when no unlabeled data are used
and 78.52% when all labels are employed.

# Keys

May the force be with you

# Results

May the force be with you

# Insights

May the force be with you

-  <span style="color:red">May the force be with you</span> <span style="color:grey">[May the force be with you]</span>

<center><img src="./001_01.png?raw=true" width = "80%" /></center>