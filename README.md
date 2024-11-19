<h3 align="center">自用改进模块仓库</h3>
<h3 align="center">Self-use Improved Model project</h3>

<div align="center">

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

</div>

---


## 📝 Table of Contents

- [📝 Table of Contents](#-table-of-contents)
- [🧐 About ](#-about-)
- [🎈 Model ](#-model-)
- [✍️ Recently ](#️-recently-)
- [✍️ Authors ](#️-authors-)

## 🧐 About <a name = "about"></a>

- 这是一个自用仓库，主要方向CV，主要目的是为了保存自己修改的各种模块改进和各种Net的改进
有时候也会上传其他东西
- This is a self-use repository, the main direction of CV, the main purpose is to save the various module improvements and various Net improvements that you have modified, and sometimes upload other things

## 🎈 Model <a name = "model"></a>
- <h4>C2F方向改进：</h4>
  <p><b>1、CCE-一种融合了ELA注意力、以CCA重新设计bottleneck、使用CGAFusion进行双尺度特征融合的C2f改进模块(在NEU-DET数据集上表现良好)</b></p>
  <p >CCE-an improved C2f module that fuses ELA attention module, redesigns bottleneck with CCA, and uses CGAFusion for dual-scale feature fusion (performs well on NEU-DET dataset)</p>
  <p><b>2、star-C2f-使用StarNet的block替换C2f的bottleneck，更加轻量化的C2f改进模块</b></p>
  <p> star-C2f - Replace the bottleneck of C2f with StarNet's block, and improve the module of C2f with lightweight</p>
  <p><b>3、CPCA-C2f-使用CPCA注意力模块与C2f的bottleneck融合，增强空间关系的提取能力，提高特征的表征能力</b></p>
  <p>CPCA-C2f - The bottleneck fusion of CPCA attention module and C2f is used to enhance the extraction ability of spatial relationships and improve the ability to characterize features</p>
  <p><b>4、GB-Concat-在Concat模块中引入GLSA机制（由全局空洞自注意力（GASA）和局部窗口自注意力（LWSA）机制组成）自适应地将需要使用Concat的特征图进行上下文的整合，同时使用BiFPN，一种多尺度加权特征融合机制，学习不同输入特征的重要性</b></p>
  <p>GB-Concat - Introduce the GLSA mechanism in the Concat module (composed of the global empty space self-attention (GASA) and local window self-attention (LWSA) mechanism) to adaptively integrate the feature maps that need to be used in Concat for contextualization, and use BiFPN, a multi-scale weighted feature fusion mechanism, to learn the importance of different input features</p>
  <p><b>5、A-SPPF-在SPPF空间金字塔池化中加入add操作，促进梯度流动</b></p>
  <p>A-SPPF - A-SPPF-Add an add operation to SPPF spatial pyramid pooling to promote gradient flow</p>

## ✍️ Recently <a name = "recently"></a>
<h3>近日更新说明：</h3>
<p><b>将个人改进项目模块整体打包，同时附带所需注意力模块（部分可酌情使用），baseline为YOLOv8，使用ultralytics8.2.0架构，部分结构的yaml已被删除（暂时不能公开），具体改进可参考./cfg/models/v8路径下的部分改进yaml，训练源码已提供（train.py）。项目同时提供了FPS计算（已整合至训练中，也可使用FPS.py单独计算）、基本的微调训练、剪枝代码（感兴趣的可以参考代码进行进一步设计，也可使用Torch-Pruning库对代码进行改进，部分剪枝模块不公开。</b></p>
## ✍️ Authors <a name = "authors"></a>
- [@GuGuGuGun](https://github.com/GuGuGuGun) - Idea & Initial work
---
<h3>遵循MIT开源协议</h3>
<h3>License:MIT</h3>
