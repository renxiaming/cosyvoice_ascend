# Qwen2.5-VL 量化案例

## 模型介绍

- [Qwen2.5-VL](https://qwenlm.github.io/zh/blog/qwen2.5-vl/) 是阿里云研发的 Qwen 模型家族的旗舰视觉语言模型，对比此前发布的 Qwen2-VL 实现了巨大的飞跃。Qwen2.5-VL 的主要特点如下所示：
    - 感知更丰富的世界：Qwen2.5-VL 不仅擅长识别常见物体，如花、鸟、鱼和昆虫，还能够分析图像中的文本、图表、图标、图形和布局。
    - Agent：Qwen2.5-VL 直接作为一个视觉 Agent，可以推理并动态地使用工具，初步具备了使用电脑和使用手机的能力。
    - 理解长视频和捕捉事件：Qwen2.5-VL 能够理解超过 1 小时的视频，并且这次它具备了通过精准定位相关视频片段来捕捉事件的新能力。
    - 视觉定位：Qwen2.5-VL 可以通过生成 bounding boxes 或者 points 来准确定位图像中的物体，并能够为坐标和属性提供稳定的 JSON 输出。
    - 结构化输出：对于发票、表单、表格等数据，Qwen2.5-VL 支持其内容的结构化输出，惠及金融、商业等领域的应用。

### Qwen2.5-VL模型当前已验证的量化方法

- W8A8量化：Qwen2.5-VL

### 此模型仓已适配的模型版本权重获取地址
#### Qwen2.5-VL
- [Qwen2.5-VL](https://huggingface.co/collections/Qwen/qwen25-vl-6795ffac22b334a837c0f9a5)

## 环境配置

- 具体环境配置请参考[使用说明](../../README.md)
- 还需要执行以下命令安装qwen_vl_utils依赖
    - pip install qwen_vl_utils
- 针对Qwen2.5-VL，transformers版本需要配置安装为4.49.0
    - pip install transformers==4.49.0

## 使用案例
- 如果需要使用npu多卡量化(特别是Qwen2.5-VL-72B这种大模型)，请先配置环境变量，暂不支持Atlas推理系列产品，以下npu 8卡量化为例：
  ```shell
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
  ```
- 若加载自定义模型，在[quant_qwen2_5vl.py](./quant_qwen2_5vl.py)中调用`from_pretrained`函数时要指定`trust_remote_code=True`让修改后的自定义代码文件能够正确的被加载。请确保加载的自定义代码文件的安全性。
- 可通过执行下面Qwen2.5-VL系列中不同场景下的命令快速生成Qwen2.5-VL量化权重。
  
### 1. Qwen2.5-VL系列
#### 1.1 Qwen2.5-VL W8A8量化 异常值抑制算法使用m2
生成Qwen2.5-VL模型量化权重，AntiOutlier异常值抑制使用m2算法配置，在NPU上进行运算。
  ```shell
  python quant_qwen2_5vl.py  --model_path {浮点权重路径} --calib_images {校准图片路径}  --save_directory {量化权重保存路径} --w_bit 8 --a_bit 8 --device_type npu --trust_remote_code True --anti_method m2
  ```
#### 1.2 Qwen2.5-VL W8A8量化 异常值抑制算法使用m4
生成Qwen2.5-VL模型量化权重，AntiOutlier异常值抑制使用m4算法配置，在NPU上进行运算。
  ```shell
  python quant_qwen2_5vl.py  --model_path {浮点权重路径} --calib_images {校准图片路径}  --save_directory {量化权重保存路径} --w_bit 8 --a_bit 8 --device_type npu --trust_remote_code True --anti_method m4
  ```

#### 1.3 Qwen2.5-VL W4A8量化 异常值抑制算法使用m4
生成Qwen2.5-VL模型量化权重，使用4bit per-group量化权重，8bit per-token量化激活值，AntiOutlier异常值抑制使用m4算法配置，在NPU上进行运算。
  ```shell
  python quant_qwen2_5vl.py  --model_path {浮点权重路径} --calib_images {校准图片路径}  --save_directory {量化权重保存路径} --w_bit 4 --a_bit 8 --act_method 1 --device_type npu --trust_remote_code True --anti_method m4 --open_outlier False --is_dynamic True --is_lowbit True --group_size 256
  ```

## 量化脚本说明

- 量化权重统一使用[quant_qwen2_5vl.py](./quant_qwen2_5vl.py)脚本生成，以下进行一些关键量化参数的说明。


### 量化参数说明
| 参数名 | 含义 | 默认值 | 使用方法 | 
| ------ | ---- | --- | -------- | 
| model_path | 浮点权重路径 | 无默认值 | 必选参数；<br>输入Qwen2.5-VL权重目录路径。 |
| calib_images | 校准集图片路径 | ./coco_pic | 必选参数；<br>输入校准数据集的目录路径。本示例中图片来源于公开数据集[COCO](https://cocodataset.org/)。 需要选取其中30张图片。|
| save_directory | 量化权重路径 | 无默认值 | 必选参数；<br>输出量化结果目录路径。 |
| part_file_size | 量化权重文件大小，单位是GB | 默认为None，无限制 | 可选参数；<br>生成量化权重文件大小，请用户自定义单个量化权重文件的最大限制。|
| w_bit | 权重量化bit | 8 | 大模型量化场景下，可配置为8或16； <br>大模型稀疏量化场景下，需配置为4。 <br>Qwen2.5-VL当前仅支持配置为8。|
| a_bit | 激活值量化bit | 8 |大模型量化场景下，可配置为8或16； <br>大模型稀疏量化场景下，需配置为8。 <br>Qwen2.5-VL当前仅支持配置为8。|
| device_type | device类型 | cpu | 可选值：['cpu', 'npu']。 |
| trust_remote_code | 是否信任自定义代码 | False | 指定`trust_remote_code=True`让修改后的自定义代码文件能够正确的被加载。(请确保加载的自定义代码文件的安全性)。|
| anti_method | 异常值抑制算法 | m2 | 选择的异常值抑制算法，当前大语言模型支持异常值抑制算法m1~m6，当前Qwen2.5-VL支持m2，m4。|
| act_method | 激活值量化方法 | 2 | (1) 1代表Label-Free场景的min-max量化方式。 <br>(2) 2代表Label-Free场景的histogram量化方式。 <br>(3) 3代表Label-Free场景的自动混合量化方式，LLM大模型场景下推荐使用。 |
| open_outlier | 是否开启权重异常值划分 | True | 可以配置为True或者False。 <br>设置为True时开启权重异常值划分，反之则关闭。|
| is_dynamic | 是否使用动态量化，即w8a8中的activation动态生成 | False | 可以配置为True或者False。 <br>设置为True时使用动态量化，反之则不使用。|
| is_lowbit | 是否使用稀疏量化的low bit算法 | False | 可以配置为True或者False。 <br>设置为True时使用稀疏量化的low bit算法，反之则不使用。 <br>在`w4a8_dynamic per-group`量化场景下需要设置为True。|
| group_size | per-group量化的分组数量 | 64 | <br>设置为64，128，256，512。 <br>在`w4a8_dynamic per-group`量化场景下仅支持256。|

- 更多参数配置要求，请参考量化过程中配置的参数 [QuantConfig](../../docs/Python-API接口说明/大模型压缩接口/大模型量化接口/PyTorch/QuantConfig.md)
  以及量化参数配置类 [Calibrator](../../docs/Python-API接口说明/大模型压缩接口/大模型量化接口/PyTorch/Calibrator.md)