# Qwen-VL 量化案例

## 模型介绍

- [Qwen-VL](https://github.com/QwenLM/Qwen-VL) 是阿里云研发的大规模视觉语言模型（Large Vision Language Model, LVLM）。 Qwen-VL 可以以图像、文本、检测框作为输入，并以文本和检测框作为输出。 Qwen-VL 系列模型性能强大，具备多语言对话、多图交错对话等能力，并支持中文开放域定位和细粒度图像识别与理解。
#### Qwen-VL模型当前已验证的量化方法

- W8A8量化：Qwen-VL

#### 此模型仓已适配的模型版本权重获取地址
##### Qwen-VL
- [Qwen-VL](https://huggingface.co/Qwen/Qwen-VL/tree/main)

## 环境配置

- 环境配置请参考[使用说明](https://gitee.com/ascend/msit/blob/master/msmodelslim/README.md)

## 量化权重生成

- 量化权重统一使用[quant_qwenvl.py](./quant_qwenvl.py)脚本生成，以下提供Qwen模型量化权重生成快速启动命令。


#### 量化参数说明
| 参数名 | 含义 | 默认值 | 使用方法 | 
| ------ | ---- | --- | -------- | 
| model_path | 浮点权重路径 | 无默认值 | 必选参数；<br>输入Qwen-VL权重目录路径。 |
| calib_images | 校准集图片路径 | ./calibImages | 可选参数；<br>输入校准数据集的目录路径。本示例中图片来源于公开数据集[COCO](https://cocodataset.org/)。 |
| save_directory | 量化权重路径 | 无默认值 | 必选参数；<br>输出量化结果目录路径。 |
| part_file_size | 量化权重文件大小，单位是GB | 无限制 | 可选参数；<br>生成量化权重文件大小，请用户自定义单个量化权重文件的最大限制。|
| w_bit | 权重量化bit | 8 | 大模型量化场景下，可配置为8或16； <br>大模型稀疏量化场景下，需配置为4。 |
| a_bit | 激活值量化bit | 8 |大模型量化场景下，可配置为8或16； <br>大模型稀疏量化场景下，需配置为8。 |
| device_type | device类型 | cpu | 可选值：['cpu', 'npu']。 |
| trust_remote_code | 是否信任自定义代码 | False | 指定`trust_remote_code=True`让修改后的自定义代码文件能够正确地被加载。(请确保加载的自定义代码文件的安全性)。 |

- 更多参数配置要求，请参考量化过程中配置的参数 [QuantConfig](https://gitee.com/ascend/msit/blob/dev/msmodelslim/docs/Python-API接口说明/大模型压缩接口/大模型量化接口/PyTorch/QuantConfig.md)
  以及量化参数配置类 [Calibrator](https://gitee.com/ascend/msit/blob/dev/msmodelslim/docs/Python-API接口说明/大模型压缩接口/大模型量化接口/PyTorch/Calibrator.md)

### 使用案例
- 请将{浮点权重路径}和{量化权重路径}替换为用户实际路径。
- 如果需要使用npu多卡量化，请先配置环境变量，仅支持1~3卡量化，暂不支持Atlas推理系列产品：
  ```shell
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
  ```
- 若加载自定义模型，调用`from_pretrained`函数时要指定`trust_remote_code=True`让修改后的自定义代码文件能够正确地被加载。(请确保加载的自定义代码文件的安全性)
  
#### 1. Qwen-VL系列
##### Qwen-VL W8A8量化
生成Qwen-VL模型量化权重，anti-outlier使用m2算法配置，在NPU上进行运算
  ```shell
  python quant_qwenvl.py  --model_path {浮点权重路径} --calib_images {校准图片路径}  --save_directory {量化权重保存路径} --w_bit 8 --a_bit 8 --device_type npu --trust_remote_code True
  ```
