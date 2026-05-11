# Qwen3-MOE 量化案例

## 模型介绍

- [Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B)、[Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) Qwen3 是 Qwen 系列中最新一代大型语言模型，提供全面的密集模型和混合专家 (MoE) 模型。Qwen3 基于丰富的训练经验，在推理、指令遵循、代理能力和多语言支持方面取得了突破性进展。其中Qwen3-MoE结构模型典型代表模型有Qwen3-235B-A22B和Qwen3-30B-A3B。

#### Qwen3-MoE模型当前已验证的量化方法

- W8A8量化：Qwen3-235B-A22B, Qwen3-30B-A3B

#### 此模型仓已适配的模型版本

- [Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B)
- [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B)

## 环境配置

- 环境配置请参考[使用说明](https://gitee.com/ascend/msit/blob/master/msmodelslim/README.md)
- transformers版本需要配置安装4.51.0以上版本
    - pip install transformers>=4.51.0

## 量化权重生成

- 量化权重可使用[quant_qwen_moe_w8a8.py](./quant_qwen_moe_w8a8.py)脚本生成。

#### quant_qwen_moe_w8a8.py 量化参数说明

| 参数名           | 含义           | 默认值  | 使用方法                          | 
|---------------|--------------|------|-------------------------------| 
| model_path    | 浮点权重路径       | 无默认值 | 必选参数；<br>输入DeepSeek权重目录路径。    |
| save_path     | 量化权重路径       | 无默认值 | 必选参数；<br>输出量化结果目录路径。          |
| layer_count   | 模型层数       | 0 | 可选参数；<br>用于调试，实际量化的层数。0表示使用所有层。        |
| anti_dataset  | 反异常值校准数据集路径 | ./anti_prompt_50.json | 可选参数；<br>用于反异常值处理的校准数据集路径。 |
| calib_dataset | 量化校准数据集路径   | ./calib_prompt_50.json | 可选参数；<br>量化校准集路径。             |
| batch_size     | 输入batch size | 4  | 可选参数；<br>生成量化校准数据时使用的batch size。batch size越大，校准速度越快，但也要求更多的显存和内存，如资源受限，请降低batch size。  |
| mindie_format | 是否开启旧的权重配置文件保存格式 | False | 开启`mindie_format`时保存的量化权重格式能够兼容MindIE 2.1.RC1及之前的版本。 |
| trust_remote_code | 是否信任自定义代码 | False | 指定`trust_remote_code=True`使修改后的自定义代码文件能够正确的被加载。(请确保加载的自定义代码文件的安全性)。 |


注：在量化脚本里面通过transformers库对模型进行加载时，调用`from_pretrained`函数时会指定`trust_remote_code=True`让修改后的modeling文件能够正确的被加载。(请确保加载的modeling文件的安全性)


更多参数配置要求，请参考量化过程中配置的参数 [QuantConfig](https://gitee.com/ascend/msit/blob/dev/msmodelslim/docs/Python-API接口说明/大模型压缩接口/大模型量化接口/PyTorch/QuantConfig.md)
以及量化参数配置类 [Calibrator](https://gitee.com/ascend/msit/blob/dev/msmodelslim/docs/Python-API接口说明/大模型压缩接口/大模型量化接口/PyTorch/Calibrator.md)

### 使用案例

- 请将{浮点权重路径}和{量化权重路径}替换为用户实际路径。
- 如果需要使用npu多卡量化，请先配置环境变量，支持多卡量化：
  ```shell
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
  ```

#### Qwen3-30B-A3B

##### Qwen3-30B-A3B w8a8 混合量化(Attention:w8a8量化，MoE:w8a8 dynamic量化)
- 生成Qwen3-30B-A3B模型 w8a8 混合量化权重
  ```shell
  python3 quant_qwen_moe_w8a8.py --model_path {浮点权重路径} --save_path {W8A8量化权重路径} --trust_remote_code True
  ```

#### Qwen3-235B-A22B

##### Qwen3-235B-A22B w8a8 混合量化(Attention:w8a8量化，MoE:w8a8 dynamic量化)
- 生成Qwen3-235B-A22B模型 w8a8 混合量化权重
  ```shell
  python3 quant_qwen_moe_w8a8.py --model_path {浮点权重路径} --save_path {W8A8量化权重路径} --trust_remote_code True
  ```

