# InternVL 2.0 量化案例

## 模型介绍

- [InternVL 2.0](https://internvl.github.io/blog/2024-07-02-InternVL-2.0/) 是上海人工智能实验室联合商汤科技开发的书生·万象多模态大模型。InternVL 2.0的关键评测指标比肩国际顶尖商用闭源模型，支持图像、视频、文本、语音、三维、医疗多种模态支持百种下游任务，性能媲美任务专用模型。书生·万象在处理复杂多模态数据方面具有强大能力，尤其是在数学、科学图表、通用图表、文档、信息图表和OCR等任务中表现优异。

#### InternVL 2.0模型当前已验证的量化方法

- W8A8量化：InternVL 2.0

#### 此模型仓已适配的模型版本权重获取地址
##### InternVL 2.0
- [InternVL 2.0-8B](https://huggingface.co/OpenGVLab/InternVL2-8B/tree/main)
- [InternVL 2.0-40B](https://huggingface.co/OpenGVLab/InternVL2-40B/tree/main)

## 环境配置

- 环境配置请参考[使用说明](../../README.md)
- transformers版本需要配置安装为4.46.0
- 另需安装pip包：pip install timm, fastchat

## 量化权重生成

- 量化权重统一使用[quant_internvl2.py](./quant_internvl2.py)脚本生成，以下提供InternVL 2.0模型量化权重生成快速启动命令。


#### 量化参数说明
| 参数名 | 含义 | 默认值 | 使用方法 | 
| ------ | ---- | --- | -------- | 
| model_path | 浮点权重路径 | 无默认值 | 必选参数；<br>输入InternVL 2.0权重目录路径。 |
| calib_images | 校准集图片路径 | ./textvqa_val | 可选参数；<br>输入校准数据集的目录路径。本示例中图片来源于公开数据集[textvqa](https://huggingface.co/datasets/maoxx241/textvqa_subset)。 当前示例仅支持该校准数据集。|
| calib_num     | 从校准数据中随机选择的数量 | 30 | 可选参数；<br>根据需要从校准集中选择一定数量的数据用于校准。 |
| save_directory | 量化权重路径 | 无默认值 | 必选参数；<br>输出量化结果目录路径。 |
| part_file_size | 量化权重文件大小，单位是GB | 无默认值 | 可选参数；<br>生成量化权重文件大小，请用户自定义单个量化权重文件的最大限制。|
| w_bit | 权重量化bit | 8 | 大模型量化场景下，可配置为8或16； <br>大模型稀疏量化场景下，需配置为4。 <br>InternVL 2.0当前仅支持配置为8。|
| a_bit | 激活值量化bit | 8 |大模型量化场景下，可配置为8或16； <br>大模型稀疏量化场景下，需配置为8。 <br>InternVL 2.0当前仅支持配置为8。|
| device_type | device类型 | cpu | 可选值：['cpu', 'npu']。 |
| is_8B_model      | 是否使用8B的模型  | 不开启  | 可选参数；<br>根据需要选择使用8B大小模型或40B大小模型，开启即指定8B大小模型。 |  
| trust_remote_code | 是否信任自定义代码 | False | 指定`trust_remote_code=True`让修改后的自定义代码文件能够正确的被加载。(请确保加载的自定义代码文件的安全性)。 |

- 更多参数配置要求，请参考量化过程中配置的参数 [QuantConfig](../../docs/Python-API接口说明/大模型压缩接口/大模型量化接口/PyTorch/QuantConfig.md)
  以及量化参数配置类 [Calibrator](../../docs/Python-API接口说明/大模型压缩接口/大模型量化接口/PyTorch/Calibrator.md)

### 使用案例
- 请将{浮点权重路径}和{量化权重路径}替换为用户实际路径。
- 如果需要使用npu多卡量化(特别是InternVL 2.0-40B这种大模型)，请先配置环境变量，暂不支持Atlas推理系列产品，以下npu 8卡量化为例：
  ```shell
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
  ```
- 若加载自定义模型，调用`from_pretrained`函数时要指定`trust_remote_code=True`让修改后的自定义代码文件能够正确的被加载。(请确保加载的自定义代码文件的安全性)
  
#### 1. InternVL 2.0系列
##### InternVL 2.0-8B W8A8量化
生成InternVL 2.0-8B模型量化权重，AntiOutlier异常值抑制使用m2算法配置（当前仅支持m2），在NPU上进行运算。
  ```shell
  python quant_internvl2.py  --model_path {浮点权重路径} --calib_images {校准图片路径}  --save_directory {量化权重保存路径} --w_bit 8 --a_bit 8 --device_type npu --is_8B_model --trust_remote_code True
  ```

##### InternVL 2.0-40B W8A8量化
生成InternVL 2.0-40B模型量化权重，AntiOutlier异常值抑制使用m2算法配置（当前仅支持m2），在NPU上进行运算。
  ```shell
  python quant_internvl2.py  --model_path {浮点权重路径} --calib_images {校准图片路径}  --save_directory {量化权重保存路径} --w_bit 8 --a_bit 8 --device_type npu --trust_remote_code True
  ```
  ```python
  # 若使用32G显存机器，并且出现由于显存分布不均导致Out of memory的现象，可在模型加载时增加显存限制，示例代码如下：
  model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=device_map,
        use_safetensors=True,
        trust_remote_code=True,
        max_memory={0: "20GB", 1: "20GB", 2: "20GB", 3: "20GB", 4: "20GB", 5: "20GB", 6: "20GB", 7: "20GB", "cpu": "20GB"}).eval()
```
