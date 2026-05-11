# LLaVA 量化案例

## 模型介绍

- [LLaVA（Large Language and Vision Assistant）](https://github.com/haotian-liu/LLaVA)是一个多模态大模型，由威斯康星大学麦迪逊分校、微软研究院和哥伦比亚大学研究者共同发布。能完成图像描述、视觉问答、图像查询、根据图片写代码等任务，还能用于多模态聊天、科学问答，帮助理解图像内容并生成相应的自然语言文本。

#### LLaVA模型当前已验证的量化方法

- W8A8量化：LLaVA-1.5-7b-hf

#### 此模型仓已适配的模型版本权重获取地址
##### LLaVA.5
- [LLaVA-1.5-7b-hf](https://huggingface.co/llava-hf/llava-1.5-7b-hf/tree/main)

## 环境配置

- 环境配置请参考[使用说明](https://gitee.com/ascend/msit/blob/master/msmodelslim/README.md)
- transformers版本需要配置安装为4.37.2

## 量化权重生成

- 量化权重统一使用[quant_llava.py](./quant_llava.py)脚本生成，以下提供LLaVA模型量化权重生成快速启动命令。


#### 量化参数说明
| 参数名 | 含义 | 默认值 | 使用方法 | 
| ------ | ---- | --- | -------- | 
| model_path | 浮点权重路径 | 无默认值 | 必选参数；<br>输入LLaVA权重目录路径。 |
| calib_images | 校准集图片路径 | ./calibImages | 可选参数；<br>输入校准数据集的目录路径。本示例中图片来源于公开数据集[COCO](https://cocodataset.org/)。 |
| save_directory | 量化权重路径 | 无默认值 | 必选参数；<br>输出量化结果目录路径。 |
| part_file_size | 量化权重文件大小，单位是GB | 无限制 | 可选参数；<br>生成量化权重文件大小，请用户自定义单个量化权重文件的最大限制。|
| w_bit | 权重量化bit | 8 | 大模型量化场景下，可配置为8或16； <br>大模型稀疏量化场景下，需配置为4。 |
| a_bit | 激活值量化bit | 8 |大模型量化场景下，可配置为8或16； <br>大模型稀疏量化场景下，需配置为8。 |
| device_type | device类型 | cpu | 可选值：['cpu', 'npu']。 |
| trust_remote_code | 是否信任自定义代码 | False | 指定`trust_remote_code=True`让修改后的自定义代码文件能够正确的被加载。(请确保加载的自定义代码文件的安全性)。 |

- 更多参数配置要求，请参考量化过程中配置的参数 [QuantConfig](https://gitee.com/ascend/msit/blob/dev/msmodelslim/docs/Python-API接口说明/大模型压缩接口/大模型量化接口/PyTorch/QuantConfig.md)
  以及量化参数配置类 [Calibrator](https://gitee.com/ascend/msit/blob/dev/msmodelslim/docs/Python-API接口说明/大模型压缩接口/大模型量化接口/PyTorch/Calibrator.md)

### 使用案例
- 请将{浮点权重路径}和{量化权重路径}替换为用户实际路径。
- 如果需要使用npu多卡量化，请先配置环境变量，支持多卡量化，暂不支持Atlas推理系列产品：
  ```shell
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
  ```
- 若加载自定义模型，调用`from_pretrained`函数时要指定`trust_remote_code=True`让修改后的自定义代码文件能够正确的被加载。(请确保加载的自定义代码文件的安全性)
  
#### 1. LLaVA系列
##### LLaVA1.5-7b W8A8量化
生成LLaVA1.5-7b模型量化权重，antioutlier使用m2算法配置，在NPU上进行运算
  ```shell
  python quant_llava.py  --model_path {浮点权重路径} --calib_images {校准图片路径}  --save_directory {量化权重保存路径} --w_bit 8 --a_bit 8 --device_type npu --trust_remote_code True
  ```
