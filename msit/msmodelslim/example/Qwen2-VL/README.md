# Qwen2-VL 量化案例

## 模型介绍

- [Qwen2-VL](https://github.com/QwenLM/Qwen2-VL) 是阿里云研发的大规模视觉语言模型（Large Vision Language Model, LVLM）的新一代。Qwen2-VL 基于 Qwen2 打造，相比 Qwen-VL，它具有以下特点：
    - 读懂不同分辨率和不同长宽比的图片：Qwen2-VL 在 MathVista、DocVQA、RealWorldQA、MTVQA 等视觉理解基准测试中取得了全球领先的表现。
    - 理解20分钟以上的长视频：Qwen2-VL 可理解长视频，并将其用于基于视频的问答、对话和内容创作等应用中。
    - 能够操作手机和机器人的视觉智能体：借助复杂推理和决策的能力，Qwen2-VL 可集成到手机、机器人等设备，根据视觉环境和文字指令进行自动操作。
    - 多语言支持：为了服务全球用户，除英语和中文外，Qwen2-VL 现在还支持理解图像中的多语言文本，包括大多数欧洲语言、日语、韩语、阿拉伯语、越南语等。

#### Qwen2-VL模型当前已验证的量化方法

- W8A8量化：Qwen2-VL

#### 此模型仓已适配的模型版本权重获取地址
##### Qwen2-VL
- [Qwen2-VL](https://huggingface.co/collections/Qwen/qwen2-vl-66cee7455501d7126940800d)

## 环境配置

- 环境配置请参考[使用说明](../../README.md)
- transformers版本请参照模型路径下config.json配置
- 另外需要单独安排pip包：qwen_vl_utils
    - pip install qwen_vl_utils
- transformers版本需要配置安装为4.46.0

## 量化权重生成

- 量化权重统一使用[quant_qwen2vl.py](./quant_qwen2vl.py)脚本生成，以下提供Qwen2VL模型量化权重生成快速启动命令。


#### 量化参数说明
| 参数名 | 含义 | 默认值 | 使用方法 | 
| ------ | ---- | --- | -------- | 
| model_path | 浮点权重路径 | 无默认值 | 必选参数；<br>输入QWen2-VL权重目录路径。 |
| calib_images | 校准集图片路径 | ./coco_pic | 可选参数；<br>输入校准数据集的目录路径。本示例中图片来源于公开数据集[COCO](https://cocodataset.org/)。 需要选取其中30张图片。|
| save_directory | 量化权重路径 | 无默认值 | 必选参数；<br>输出量化结果目录路径。 |
| part_file_size | 量化权重文件大小，单位是GB | 无限制 | 可选参数；<br>生成量化权重文件大小，请用户自定义单个量化权重文件的最大限制。|
| w_bit | 权重量化bit | 8 | 大模型量化场景下，可配置为8或16； <br>大模型稀疏量化场景下，需配置为4。 <br>Qwen2-VL当前仅支持配置为8。|
| a_bit | 激活值量化bit | 8 |大模型量化场景下，可配置为8或16； <br>大模型稀疏量化场景下，需配置为8。 <br>Qwen2-VL当前仅支持配置为8。|
| device_type | device类型 | cpu | 可选值：['cpu', 'npu']。 |
| trust_remote_code | 是否信任自定义代码 | False | 指定`trust_remote_code=True`让修改后的自定义代码文件能够正确的被加载。(请确保加载的自定义代码文件的安全性)。 |

- 更多参数配置要求，请参考量化过程中配置的参数 [QuantConfig](../../docs/Python-API接口说明/大模型压缩接口/大模型量化接口/PyTorch/QuantConfig.md)
  以及量化参数配置类 [Calibrator](../../docs/Python-API接口说明/大模型压缩接口/大模型量化接口/PyTorch/Calibrator.md)

### 使用案例
- 请将{浮点权重路径}和{量化权重路径}替换为用户实际路径。
- 如果需要使用npu多卡量化(特别是Qwen2-VL-72B这种大模型)，请先配置环境变量，暂不支持Atlas推理系列产品，以下npu 8卡量化为例：
  ```shell
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
  ```
- 若加载自定义模型，调用`from_pretrained`函数时要指定`trust_remote_code=True`让修改后的自定义代码文件能够正确的被加载。(请确保加载的自定义代码文件的安全性)
  
#### 1. Qwen2-VL系列
##### Qwen2-VL W8A8量化
生成Qwen2-VL模型量化权重，AntiOutlier异常值抑制使用m2算法配置（当前仅支持m2），在NPU上进行运算。
  ```shell
  python quant_qwen2vl.py  --model_path {浮点权重路径} --calib_images {校准图片路径}  --save_directory {量化权重保存路径} --w_bit 8 --a_bit 8 --device_type npu --trust_remote_code True
  ```
