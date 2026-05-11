
# msModelSlim

## 介绍

MindStudio ModelSlim，昇腾模型压缩工具。 

昇腾模型压缩工具，一个以加速为目标、压缩为技术、昇腾为根本的亲和压缩工具。支持训练加速和推理加速，包括模型低秩分解、稀疏训练、训练后量化、量化感知训练等功能，昇腾AI模型开发用户可以灵活调用Python API接口，对模型进行性能调优，并支持导出不同格式模型，在昇腾AI处理器上运行。


## 环境和依赖

- 硬件环境请参见《[昇腾产品形态说明](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/quickstart/productform/hardwaredesc_0001.html)》。
- 物理机、容器、虚拟机场景下驱动固件和CANN软件的安装方案参见《[安装方案](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/softwareinst/instg/instg_0002.html?Mode=PmIns&InstallType=local&OS=Ubuntu&Software=cannToolKit)》。硬件配套的软件下载资源参见《[配套资源下载](https://www.hiascend.com/developer/download/commercial/result?module=cann)》，来安装昇腾设备开发或运行环境。
- PyTorch框架为必选依赖；torch_npu 插件则视目标设备而定，仅在 NPU 上进行量化时需要，CPU 上不依赖。


## 版本配套

| 条件 | 要求 |
|---|---|
| CANN版本 | >= 8.0.RC1.alpha001 |
| 硬件要求 | Atlas 800I A2 推理服务器、Atlas 300I Duo 推理卡|


## 注意事项
8.0.RC3 及之前的 CANN 包已包含msModelSlim代码，安装CANN包即可使用；8.0.RC4 及之后的 CANN 包需与msModelSlim代码仓配套使用。

**【Notice！！！】** 非多模态模型，如果量化后的权重需要在MindIE 2.1.RC1版本前部署，请在执行量化命令时加上 **--mindie_format** 参数。


## msModelSlim安装方式

msModelSlim当前处于逐步开源过程中，计划通过CANN的8.0.RC2、8.0.RC3、8.0.0三个版本进行过渡。  

版本交替期间提供两种方式使用msModelSlim工具：

方式一：
- 下载安装CANN（仅限8.0.RC3及之前的版本）并配置环境变量后即可使用msModelSlim。可以参考[安装CANN软件包](https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/softwareinst/instg/instg_0007.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit)  
**注意** ：8.0.RC2版本存在已知问题，使用modelslim调用接口时，部分功能存在异常。请使用msmodelslim调用。 

方式二：  
（CANN8.0.RC3之后的版本，将会只支持开源方式使用，通过CANN包直接使用的方式将不再受支持。后续功能优化、新增将更新在开源版本中。）
- 下载安装CANN（8.0.RC3之后的版本）及开源版本的msModelSlim  
    **操作步骤：**
    - 下载安装CANN并设置环境变量，可以参考[安装CANN软件包](https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/softwareinst/instg/instg_0007.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit)
    - git clone下载msit仓代码；
    - 进入到msit/msmodelslim的目录 `cd msit/msmodelslim`；并在进入的msmodelslim目录下，运行安装脚本 `bash install.sh`;
    - (可选，稀疏量化场景下需要此步骤)进入python环境下的site_packages包管理路径 `cd {python环境路径}/site-packages/msmodelslim/pytorch/weight_compression/compress_graph/`  
    以下是以/usr/local/为用户所在目录，以3.7.5为python版本的样例代码：
    ```
    cd usr/local/lib/python3.7/site-packages/msmodelslim/pytorch/weight_compression/compress_graph/
    ```
    - (可选，稀疏量化场景下需要此步骤)编译weight_compression组件 `sudo bash build.sh {CANN包安装路径}/ascend-toolkit/latest`
    - (可选，稀疏量化场景下需要此步骤)上一步编译操作会得到bulid文件夹，给build文件夹相关权限 `chmod -R 550 build`
    - (可选，使用Precision Tool需要此步骤)参考[precision_tool使用方法说明](precision_tool/readme.md/#precision-tool-使用方法说明)里的步骤设置环境变量


## 特性清单
- msModelSlim针对开发者的差异化需求，提供了以下模型压缩方案：

| 功能名称                          | 功能简介                                                                                                                                                  |
|-------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| [模型低秩分解](msmodelslim/pytorch/low_rank_decompose)                       | 低秩分解是一种矩阵分解技术，它可以将一个大型矩阵分解为若干个较小矩阵的乘积，这些较小矩阵的秩相对较低。低秩分解在很多领域都有应用，如数据分析、机器学习、图像处理等。                                                                    |
| [模型稀疏加速训练](msmodelslim/pytorch/sparse)                      | 稀疏加速算法是一种旨在通过减少模型参数的数量来提高计算效率的训练方法。这种算法基于网络扩增训练的思想，通常涉及到在训练过程中引入额外的参数，然后通过某种方式对这些参数进行筛选或修剪，以实现模型的稀疏化。                                                 |
| [模型蒸馏](msmodelslim/common/knowledge_distill)                          | 蒸馏调优是一种模型压缩技术，它将一个大型、复杂的教师模型的知识转移到一个小型的学生模型中。在这个过程中，学生模型试图模仿教师模型的输出，通常是通过训练学生模型来匹配教师模型的输出或中间层的激活。                                                     |
| [大模型量化](msmodelslim/pytorch/llm_ptq)                         | 大模型量化是一种模型压缩技术，它通过减少模型权重和激活的数值表示的精度来降低模型的存储和计算需求。量化工具通常会将高位浮点数转换为低位定点数，从而直接减少模型权重的体积。                                                                 |
| [大模型稀疏量化](msmodelslim/pytorch/llm_sparsequant)和[权重压缩](msmodelslim/pytorch/weight_compression)                  | 大模型稀疏量化工具结合了模型量化与模型稀疏化两种技术，旨在通过减少模型体积和降低内存及带宽消耗来提升模型的性能。                                                                                              |
| [长序列压缩](msmodelslim/pytorch/ra_compression/README.md)                | 长序列压缩通过一种免训练的KV-Cache的缓存压缩算法（RazorAttention），直接应用于KV-Cache管理策略中，通过这种集成，Transformer模型能够在处理长序列时更加高效，同时保持或提升模型的性能。                                                                                              |
| 训练后量化([onnx](msmodelslim/onnx)/[pytorch](msmodelslim/pytorch/quant/ptq_tools)/[mindspore](msmodelslim/mindspore/quant/ptq_quant)) | 训练后量化不需要重新训练模型，而是在模型训练完成后直接对模型进行量化。                                                                                                                   |
| [量化感知训练](msmodelslim/pytorch/quant/qat_tools)                        | 量化感知训练是一种在模型训练过程中模拟量化效果的训练方法。通过在训练过程中加入量化操作，模型可以适应量化带来的精度损失，从而在量化后的模型上保持较高的性能。                                                                        |
| [Transformer类模型权重剪枝调优](msmodelslim/pytorch/prune/transformer_prune)          | 模型权重剪枝是一种通过移除模型中不重要的权重（即那些对模型性能影响较小的权重）来减少模型复杂度的技术。剪枝后的模型权重更少，从而可以减少模型的存储需求，并可能加快模型的推理速度。                                                             |
| [基于重要性评估的剪枝调优](msmodelslim/pytorch/prune)                  | 基于重要性评估进行剪枝调优是一种常用的方法，它涉及到评估模型中每个权重的重要性，并据此决定哪些权重应该被剪枝。  基于重要性评估的剪枝调优可以显著减少模型的大小，提高模型的推理效率，同时尽量保持模型的性能。这种方法在深度学习模型压缩和加速中非常有用，特别是在需要部署模型到资源受限的环境中的情况下。 |
| [多模态推理优化工具](msmodelslim/pytorch/multi_modal) | 针对大规模多模态生成模型的推理优化解决方案，专注于提升推理效率和资源利用率。当前支持自适应采样优化等特性，可显著提高稳定扩散模型的推理效率。 |

### [Python API接口说明](docs/Python-API接口说明) 


## [大模型已验证列表](docs/大模型已验证列表.md)

## 使用案例及调优指南

[Attention量化使用说明](docs/FA量化使用说明.md)
<br>[w8a8精度调优策略](docs/w8a8精度调优策略.md)
<br>[w8a16精度调优策略](docs/w8a16精度调优策略.md)
<br>[稀疏量化精度调试案例](docs/稀疏量化精度调试案例.md)
<br>[量化及稀疏量化场景导入代码样例](msmodelslim/pytorch/llm_ptq/量化及稀疏量化场景导入代码样例.md)
<br>[msModelSlim量化权重转AutoAWQ&AutoGPTQ使用指南](docs/msModelSlim量化权重转AutoAWQ&AutoGPTQ使用指南.md)
<br>[低显存量化特性使用说明](docs/低显存量化特性使用说明.md)
<br>[mindspeed-llm框架量化使用说明](msmodelslim/pytorch/mindspeed_adapter/README.md)

#### 许可证
[Apache License 2.0](/LICENSE)
