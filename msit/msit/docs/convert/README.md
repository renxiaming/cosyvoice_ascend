# msit convert功能使用指南

## 简介
convert模型转换工具依托ATC（Ascend Tensor Compiler），AOE（Ascend Optimization Engine），MindIE（Mind Inference Engine）推理引擎，提供由ONNX、TensorFlow、Caffe、MindSpore模型至om模型的转换及调优功能。

* ATC (Ascend Tensor Compiler)
> 昇腾张量编译器（Ascend Tensor Compiler，简称ATC）是异构计算架构CANN体系下的模型转换工具， 它可以将开源框架的网络模型以及Ascend IR定义的单算子描述文件（json格式）转换为昇腾AI处理器支持的.om格式离线模型。
>
> 模型转换过程中，ATC会进行算子调度优化、权重数据重排、内存使用优化等具体操作，对原始的深度学习模型进行进一步的调优，从而满足部署场景下的高性能需求，使其能够高效执行在昇腾AI处理器上。
>
> [更多说明](https://www.hiascend.com/document/detail/zh/canncommercial/80RC22/devaids/auxiliarydevtool/atlasatc_16_0005.html)
* AOE (Ascend Optimization Engine)
> AOE（Ascend Optimization Engine）是一款自动调优工具，作用是充分利用有限的硬件资源，以满足算子和整网的性能要求。
>
> AOE通过生成调优策略、编译、在运行环境上验证的闭环反馈机制，不断迭代出更优的调优策略，最终得到最佳的调优策略，从而可以更充分利用硬件资源，不断提升网络的性能，达到最优的效果。
>
> [更多说明](https://www.hiascend.com/document/detail/zh/canncommercial/80RC22/devaids/auxiliarydevtool/auxiliarydevtool_0014.html)
* MindIE (Mind Inference Engine)
> MindIE是华为昇腾针对AI全场景业务的推理加速套件。通过分层开放AI能力，支撑用户多样化的AI业务需求，使能百模千态，释放昇腾硬件设备算力。其中MindIE-RT（Ascend MindIE Runtime）能够将不同的深度学习框架（PyTorch、ONNX等）上完成训练的算法模型统一为计算图表示，具备多粒度模型优化、整图下发以及推理部署等功能。
>
> [更多说明](https://www.hiascend.com/document/detail/zh/mindie/100/whatismindie/mindie_what_0001.html)

## 工具安装

- 如果使用MindIE-RT做模型转换，需要在安装convert前安装MindIE，MindIE安装请参考MindIE[官方网站](https://www.hiascend.com/software/mindie)。

  注：MindIE-RT的python API调用要求python为3.10版本，在其他python版本下安装会报错。
  MindIE 1.0.RC1版本的报错如下，请忽略，不影响模型转换：
  ```shell
  AscendIE python api Install failed, please install python3.10 firstly!
  ```
  MindIE 1.0.RC2版本后，安装失败会清空整个安装目录，因此必须在python3.10下安装才能使用模型转换功能，具体的安装报错信息如下：
  ```shell
  Install AscendIE python api failed, detail info can be checked in {install_path}/mindie_rt_install.log.
  ```

- 工具安装请见 [msit一体化工具使用指南](../install/README.md)

## 工具使用

一站式msit工具使用命令格式说明如下：
```shell
msit convert [subcommand]
```
msit convert目前支持以下3种子命令：

| subcommand | 说明                      |
| ---------- | ------------------------- |
| atc        | 使用atc进行模型转换       |
| aoe        | 使用aoe进行模型转换及调优 |
| aie        | 使用MindIE-RT进行模型转换(aie现已合入MindIE)  |

### atc命令
使用ATC后端进行模型转换，命令格式如下：
```shell
msit convert atc [args]
```
参数定义严格遵从ATC的参数定义，由于参数较多，详情可参考：https://www.hiascend.com/document/detail/zh/canncommercial/80RC22/devaids/auxiliarydevtool/atlasatc_16_0039.html#ZH-CN_TOPIC_0000001949484154__section6351244132417
使用示例：
```shell
msit convert atc --model resnet50.onnx --framework 5 --soc_version <soc_version> --output resnet50
```
### aoe命令
使用AOE后端进行模型转换，命令格式如下：
```shell
msit convert aoe [args]
```
参数定义严格遵从AOE的参数定义，由于参数较多，详情可参考：https://www.hiascend.com/document/detail/zh/canncommercial/80RC22/devaids/auxiliarydevtool/auxiliarydevtool_0014.html

使用示例：
```shell
msit convert aoe --model resnet50.onnx --job_type 2 --output resnet50
```
### aie命令
使用MindIE-RT后端进行模型转换，目前仅支持ONNX模型的转换，命令格式如下：
```shell
msit convert aie [args]
```
参数说明如下：

| 参数                  | 说明                                                      | 是否必选 |
|---------------------|---------------------------------------------------------|------|
| -gm, --golden-model | 标杆模型输入路径，支持onnx模型                                       | 是    |
| -of, --output-file  | 输出文件，需要有后缀 .om, 当前支持基于 MindIE-RT(Ascend MindIE Runtime) 的模型转换 | 是    |
| -soc, --soc-version | 芯片类型                 | 是    |
| -h, --help | 帮助信息                 | 否    |

命令示例如下：

```shell
msit convert aie --golden-model resnet50.onnx --output-file resnet50.om --soc-version <soc_version> 
```

#### 使用案例
更多关于aie子命令的介绍请移步[convert工具使用示例](../../examples/cli/convert/01_basic_usage)


## FAQ
使用convert组件进行模型转换时如遇问题，请先行查阅[FAQ](FAQ.md)