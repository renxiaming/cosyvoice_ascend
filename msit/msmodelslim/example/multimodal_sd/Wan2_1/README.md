# Wan2.1 量化使用说明

## Wan2.1 模型介绍

Wan2.1是阿里巴巴发布的一套全面且开放的视频基础模型，它突破了视频生成的界限。支持多种生成任务：

- **文本到视频 (T2V)**: 根据文本描述生成视频
- **图像到视频 (I2V)**: 根据输入图像生成视频
- **文本到图像 (T2I)**: 根据文本描述生成图像

Wan2.1模型基于Transformer架构，支持逐层量化，能够显著降低量化过程中的内存占用。

## Wan2.1 量化支持

### 已验证的量化类型

| 量化类型 | 描述 | 适用场景 | 配置示例 |
|----------|------|----------|----------|
| w8a8_dynamic | 权重8bit per-channel对称量化，激活值8bit动态量化 | 推荐使用，适应不同输入 | [wan2_1_w8a8_dynamic.yaml](../../../lab_practice/wan2_1/wan2_1_w8a8_dynamic.yaml) |

### 量化特性

- **逐层量化**: 支持逐层处理，大幅降低内存占用
- **单卡量化**: 结合逐层量化特性，可实现在Atlas 800I/800T A2(64G)设备上的单卡量化
- **动态激活值量化**: 激活值使用per_token量化范围，提高精度

## 量化命令

### 方式一：使用quant_type参数

```bash
msmodelslim quant \
    --model_path /path/to/wan2_1_float_weights \
    --save_path /path/to/wan2_1_quantized_weights \
    --device npu \
    --model_type Wan2_1 \
    --quant_type w8a8 \
    --trust_remote_code True
```

### 方式二：使用配置文件

```bash
msmodelslim quant \
    --model_path /path/to/wan2_1_float_weights \
    --save_path /path/to/wan2_1_quantized_weights \
    --device npu \
    --model_type Wan2_1 \
    --config_path /path/to/wan2_1_w8a8_dynamic.yaml \
    --trust_remote_code True
```

## 配置文件说明

### 基础配置结构

```yaml
apiversion: multimodal_sd_modelslim_v1
metadata:
  config_id: wan2_1_w8a8_dynamic
  score: 90
  verified_model_types:
    - Wan2_1
  label:
    w_bit: 8
    a_bit: 8
    is_sparse: False
    kv_cache: False

spec:
  process:
    - type: "linear_quant"
      qconfig:
        act:
          scope: "per_token"   # 激活值量化范围
          dtype: "int8"        # 激活值量化数据类型
          symmetric: False      # 是否对称量化
          method: "minmax"      # 量化方法
        weight:
          scope: "per_channel"   # 权重量化范围
          dtype: "int8"        # 权重量化数据类型
          symmetric: True       # 是否对称量化
          method: "minmax"      # 量化方法
      include: ["*"]           # 包含的层模式
      exclude: ["*ffn.2*"]     # 排除的层模式

  save:
    - type: "mindie_format_saver"
      part_file_size: 0

  multimodal_sd_config:
    dump_config:
      capture_mode: "args"
      dump_data_dir: ""
    model_config:
      prompt: "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about."
      offload_model: True
      frame_num: 121
```

### 关键配置参数

#### 量化配置 (process)

- **type**: 处理器类型，固定为"linear_quant"
- **qconfig.act**: 激活值量化配置
  - `scope`: 量化范围，推荐使用"per_token"
  - `dtype`: 数据类型，固定为"int8"
  - `symmetric`: 是否对称量化，推荐False
  - `method`: 量化方法，推荐"minmax"
- **qconfig.weight**: 权重量化配置
  - `scope`: 量化范围，固定为"per_channel"
  - `dtype`: 数据类型，固定为"int8"
  - `symmetric`: 是否对称量化，推荐True
  - `method`: 量化方法，推荐"minmax"

#### 保存配置 (save)

- **type**: 保存器类型，使用"mindie_format_saver"
- **part_file_size**: 分片文件大小，0表示不分片

#### 多模态配置 (multimodal_sd_config)

- **dump_config**: 校准数据捕获配置
  - `capture_mode`: 捕获模式，推荐"args"
  - `dump_data_dir`: 校准数据保存目录，空字符串表示使用默认路径
- **model_config**: 模型加载与推理配置，具体可配置的参数需要参考原始推理工程仓[Wan2.1模型仓库](https://modelers.cn/models/MindIE/Wan2.1)
  - `prompt`: 校准用的提示词
  - `offload_model`: 是否启用模型卸载
  - `frame_num`: 生成帧数

## 常见问题

### Q1: 是否支持w8a8静态量化？
**A**: 可以修改配置文件中的process部分，调整qconfig.act.scope为"per_tensor"启动静态量化，但精度损失严重，不推荐。

### Q2: 如何自定义量化配置？

**A**: 可以修改配置文件中的process部分，调整量化参数和层选择策略。

## 相关资源

- [Wan2.1模型仓库](https://modelers.cn/models/MindIE/Wan2.1)
- [配置协议说明](../../../docs/一键量化/配置协议说明.md)
- [逐层量化特性说明](../../../docs/一键量化/features/layer_wise_quantization.md)