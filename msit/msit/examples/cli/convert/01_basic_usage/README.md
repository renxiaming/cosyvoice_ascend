# aie子命令

## 介绍

msit convert aie命令依托MindIE（Mind Inference Engine）推理引擎，提供由ONNX模型转换至om模型的功能。

## 使用场景约束
1. 当前仅支持**Atlas200/300/500推理产品以及Atlas推理系列产品**平台的AIE转换；
2. 当前仅支持**FP16**精度下的模型转换
3. 目前msit convert aie命令支持以下4个模型：

|  **序号**                  |  **onnx模型名称**                                |  **模型链接**  |
|---------------------|----------------------------------------------------------|------|
| 1 | Resnet50 | https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/Resnet50_Pytorch_Infer |
| 2 | DBNet_MobileNetV3 | https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/DBNet_MobileNetV3 |
| 3 | CRNN | https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/CRNN_BuiltIn_for_Pytorch |
| 4 |YOLOX-s| https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/YoloXs_for_Pytorch |


## 运行示例

```shell
msit convert aie --golden-model resnet50.onnx --output-file resnet50.om --soc-version <soc_version>
```

结果输出如下：
```shell
[INFO] Execute command:['./msit_convert', 'resnet50.onnx', 'resnet50.om', '<soc_version>']
[INFO] AIE model convert finished, the command: ['./msit_convert', 'resnet50.onnx', 'resnet50.om', '<soc_version>']
[INFO] convert model finished.
```
