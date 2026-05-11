# Qwen 量化案例

## 模型介绍
- 千问（qwen）语言大模型是阿里巴巴集团推出的大型语言模型，具备强大的自然语言处理能力，能够理解和生成文本，应用于智能客服、内容生成、问答系统等多个场景，助力企业智能化升级。

#### Qwen模型当前已验证的量化方法
- W8A8量化：Qwen-7B，Qwen-14B，Qwen1.5-14B，Qwen1.5-32B，Qwen2-7B，Qwen2-72B，Qwen2.5-7B，Qwen2.5-14B，Qwen2.5-32B，Qwen3-14B，Qwen3-32B，QwQ-32B
- W8A16量化：Qwen-72B，Qwen1.5-72B，Qwen1.5-110B，Qwen2-72B
- W4A4 Flatquant Dynamic量化：Qwen3-32B
- 稀疏量化：Qwen1.5-14B，Qwen2-7B，Qwen2-72B，Qwen2.5-7B，Qwen2.5-14B，QwenCode2.5-7B，QwQ-32B，Qwen3-8B，Qwen3-14B，Qwen3-32B
- KV cache量化：Qwen2-72B
- Attention 量化：Qwen2.5-72B

#### 此模型仓已适配的模型版本
##### Qwen
- [Qwen-7B](https://huggingface.co/Qwen/Qwen-7B/tree/main)
- [Qwen-14B](https://huggingface.co/Qwen/Qwen-14B/tree/main)
- [Qwen-72B](https://huggingface.co/Qwen/Qwen-72B/tree/main)
##### Qwen1.5
- [Qwen1.5-14B](https://huggingface.co/Qwen/Qwen1.5-14B/tree/main)
- [Qwen1.5-32B](https://huggingface.co/Qwen/Qwen1.5-32B/tree/main)
- [Qwen1.5-72B](https://huggingface.co/Qwen/Qwen1.5-72B/tree/main)
##### Qwen2
- [Qwen2-7B](https://huggingface.co/Qwen/Qwen2-7B/tree/main)
- [Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct/tree/main)
##### Qwen2.5
- [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/tree/main)
- [Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct/tree/main)
- [Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/tree/main)
- [Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/tree/main)
##### Qwen3
- [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B/tree/main)
- [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B/tree/main)
- [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B/tree/main)
##### QwQ
- [QwQ-32B](https://modelscope.cn/models/Qwen/QwQ-32B)

## 环境配置

- 环境配置请参考[使用说明](https://gitee.com/ascend/msit/blob/master/msmodelslim/README.md)

## 量化权重生成

- 量化权重统一使用[quant_qwen.py](./quant_qwen.py)脚本生成，以下提供Qwen模型量化权重生成快速启动命令。


#### 量化参数说明
| 参数名 | 含义 | 默认值 | 使用方法 | 
| ------ | ---- | --- | -------- | 
| model_path | 浮点权重路径 | 无默认值 | 必选参数；<br>输入Qwen权重目录路径。 |
| save_directory | 量化权重路径 | 无默认值 | 必选参数；<br>输出量化结果目录路径。 |
| part_file_size | 量化权重文件大小，单位是GB | 无限制 | 可选参数；<br>生成量化权重文件大小，请用户自定义单个量化权重文件的最大限制。|
| calib_texts | 量化校准数据列表 | 无 | 校准数据集。 |
| calib_file | 量化校准数据文件 | teacher_qualification.jsonl | 存放校准数据的json文件。 |
| w_bit | 权重量化bit | 8 | 大模型量化场景下，可配置为8或16； <br>大模型稀疏量化场景下，需配置为4。 |
| a_bit | 激活值量化bit | 8 |大模型量化场景下，可配置为8或16； <br>大模型稀疏量化场景下，需配置为8。 |
| disable_names | 手动回退的量化层名称 | qwen1 回退所有c_proj层 <br> 其他模型默认回退所有down_proj层 | 用户可根据精度要求手动设置，默认回退隐藏层的降维投影层。 |
| device_type | device类型 | cpu | 可选值：['cpu', 'npu']。 |
| fraction | 模型权重稀疏量化过程中被保护的异常值占比  |0.01| 取值范围[0.01,0.1]。|
| act_method | 激活值量化方法 | 1 |(1) 1代表Label-Free场景的min-max量化方式。 <br>(2) 2代表Label-Free场景的histogram量化方式。 <br>(3) 3代表Label-Free场景的自动混合量化方式，LLM大模型场景下推荐使用。|
| co_sparse | 是否开启稀疏量化功能 | False | True: 使用稀疏量化功能；<br>False: 不使用稀疏量化功能。 |
| anti_method | 离群值抑制参数 | 无默认值 |'m1': SmoothQuant算法。<br>'m2': SmoothQuant加强版算法。<br>'m3': AWQ算法。<br>'m4': smooth优化算法 。<br>'m5': CBQ量化算法。<br>'m6': Flex smooth量化算法。|
| disable_level | L自动回退等级 | L0 | 配置示例如下：<br>'L0'：默认值，不执行回退。<br>'L1'：回退1层。<br>'L2'：回退2层。<br>'L3'：回退3层。<br>'L4'：回退4层。<br>'L5'：回退5层。|
| do_smooth | 是否使用smooth量化 | False | True: 使用smooth量化；<br>False: 不使用smooth量化。 |
| use_sigma | 是否启动sigma功能 | False|True: 开启sigma功能；<br>False: 不开启sigma功能。 |
| use_reduce_quant | 权重量化是否是lccl all reduce量化 | False | 用于MindIE推理的标识。 |
| tp_size | 模拟多卡量化时的卡数 | 1 | 数据取值范围为[1,2,4,8,16]，默认值为1，不启用模拟多卡量化。<br>设置为2、4、8、16时，对于通信层的linear会进行模拟多卡，每张卡使用不同的scale和offset进行量化。 |
| sigma_factor | sigma因子 | 3.0 | 数据取值范围为[1.0, 3.0]，默认值为3.0。 |
| is_lowbit | 是否开启lowbit量化功能 | False|(1) 当w_bit=4，a_bit=8时，为大模型稀疏量化场景，表示开启lowbit稀疏量化功能。<br>(2) 其他场景为大模型量化场景，会开启量化自动精度调优功能。<br>当前量化自动精度调优框架支持W8A8，W8A16量化。|
| w_sym | 权重量化是否对称 | True | True: 使用对称量化；<br>False: 使用非对称量化。 |
| use_kvcache_quant | 是否使用kvcache量化功能 | False | True: 使用kvcache量化功能；<br>False: 不使用kvcache量化功能。|
| use_fa_quant | 是否使用FA3量化 | False | True: 使用FA3量化类型；<br>False: 不使用FA3量化类型。|
| fa_amp | 自动回退的layer数量 | 0 | 数据类型为int，默认值为0。数据取值范围是大于等于0，并且小于等于模型layer数量，如果超出模型的layer数量将会取模型的最大layer数量为回退层数。 |
| open_outlier | 是否开启权重异常值划分 | True | True：开启权重异常值划分。<br>False：关闭权重异常值划分。<br>说明：(1)仅在lowbit设置为True时生效。(2)per_group量化场景下，需协同设置is_lowbit为True，open_outlier为False。|
| group_size | per_group量化中group的大小 | 64 | 默认值为64，支持配置为32，64，128。<br>说明:仅适用于per_group量化场景，需协同设置is_lowbit为True，open_outlier为False。|
| is_dynamic | 是否使用per-token动态量化功能 | False | True: 使用per-token动态量化；<br>False: 不使用per-token动态量化。 |
| input_ids_name | 指定分词结果中输入 ID 对应的键名 | input_ids | 无 |
| attention_mask_name | 指定分词结果中注意力掩码对应的键名 | attention_mask | 无 |
| tokenizer_args | 加载自定义tokenizer时传入的自定义参数 | 无 | 以字典方式传入。 |
| disable_last_linear | 是否回退最后linear层 | True | True：回退最后linear层。<br>False：不回退最后linear层。 |
| model_name | 模型名称，可选参数 | None | 用于控制异常值抑制参数。 |
| model_type | qwen模型类型 | qwen2|  若使用qwen1模型，请输入该参数为qwen1。 |
| anti_calib_file | 离群值抑制校准数据文件 | None | 用于离群值抑制的校准数据文件路径(.json或.jsonl)。|
| disable_threshold | 自动回退阈值 | 0 | 当值大于0时，会根据阈值自动选择需要回退的层。|
| pdmix | 是否使用PDMix量化类型 | False | True: 使用PDMix量化类型；<br>False: 不使用PDMix量化类型。|
| trust_remote_code | 是否信任自定义代码 | False | 指定`trust_remote_code=True`让修改后的自定义代码文件能够正确的被加载。(请确保加载的自定义代码文件的安全性)。 |
| layer_count | 加载模型时的层数 | 0 | 默认值为0表示量化模型所有层；<br>用于调试，实际量化的层数，当设置为N时，将从第0层开始量化到第N-1层（如设置为5，则量化0,1,2,3,4这5层）。<br>取值范围：[0, 模型总层数]。 |
| mindie_format | 非多模态模型量化后的权重配置文件是否兼容MindIE现有版本 | False | 开启`mindie_format`时保存的量化权重格式能够兼容MindIE 2.1.RC1及之前的版本。 |
| w_method | 权重量化方法 | MinMax | 可选值：['MinMax', 'GPTQ', 'HQQ', 'NF']。 |


- 更多参数配置要求，请参考量化过程中配置的参数 [QuantConfig](https://gitee.com/ascend/msit/blob/dev/msmodelslim/docs/Python-API接口说明/大模型压缩接口/大模型量化接口/PyTorch/QuantConfig.md)
  以及量化参数配置类 [Calibrator](https://gitee.com/ascend/msit/blob/dev/msmodelslim/docs/Python-API接口说明/大模型压缩接口/大模型量化接口/PyTorch/Calibrator.md)

#### W4A4 Flatquant Dynamic量化专用参数说明 (w4a4.py)
| 参数名 | 含义 | 默认值 | 使用方法 | 
| ------ | ---- | --- | -------- | 
| model_path | 浮点权重路径 | 无默认值 | 必选参数；<br>输入Qwen权重目录路径。 |
| save_directory | 量化权重路径 | 无默认值 | 必选参数；<br>输出量化结果目录路径。 |
| layer_count | 加载模型时的层数 | 0 | 默认值为0表示量化模型所有层；<br>用于调试，实际量化的层数，当设置为N时，将从第0层开始量化到第N-1层（如设置为5，则量化0,1,2,3,4这5层）。<br>取值范围：[0, 模型总层数]。 |
| calib_file | 量化校准数据文件 | ../common/wiki.jsonl | 用于校准的数据文件路径，支持.jsonl格式文件。<br>文件中每行应包含'inputs_pretokenized'字段。 |
| batch_size | 校准时的批处理大小 | 4 | 生成量化校准数据时使用的batch size。<br>取值范围：[1, 16] |
| mindie_format | 非多模态模型量化后的权重配置文件是否兼容MindIE现有版本 | False | 开启`mindie_format`时保存的量化权重格式能够兼容MindIE迭代四B050前版本。|
| trust_remote_code | 是否信任自定义代码 | False | 指定`trust_remote_code=True`使修改后的自定义代码文件能够正确的被加载。(请确保加载的自定义代码文件的安全性) |

### 使用案例
- 请将{浮点权重路径}和{量化权重路径}替换为用户实际路径。
- 如果需要使用npu多卡量化，请先配置环境变量，支持多卡量化：
  ```shell
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
  ```
- 若加载自定义模型，调用`from_pretrained`函数时要指定`trust_remote_code=True`让修改后的自定义代码文件能够正确的被加载。(请确保加载的自定义代码文件的安全性)
  
#### 1. Qwen1系列
##### Qwen1-14b W8A8量化
生成Qwen1-14b模型量化权重，antioutlier使用m2算法配置，使用min-max量化方式，在CPU上进行运算
  ```shell
  python3 quant_qwen.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file ../common/boolq.jsonl --w_bit 8 --a_bit 8 --device_type cpu  --anti_method m2 --act_method 1 --model_type qwen1 --trust_remote_code True
  ```

##### Qwen1-72b W8A16量化
生成Qwen1-14b模型量化权重，激活值量化使用自动混合量化方式，在CPU上进行运算
  ```shell
  python3 quant_qwen.py --model_path {浮点权重路径} --save_directory {W8A16量化权重路径} --calib_file ../common/ceval.jsonl --w_bit 8 --a_bit 16 --device_type cpu  --act_method 3 --model_type qwen1 --trust_remote_code True
  ```
#### 2. Qwen1.5系列
##### Qwen1.5-14b, Qwen1.5-32b W8A8 量化
生成量化权重，使用min-max量化方式，校准数据集使用50条BoolQ数据，在npu上进行运算
  ```shell
  python3 quant_qwen.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file ../common/boolq.jsonl --w_bit 8 --a_bit 8 --device_type npu --trust_remote_code True
  ```
 稀疏量化权重请使用以下指令生成
  ```shell
  python3 quant_qwen.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file ../common/cn_en.jsonl --w_bit 4 --a_bit 8 --device_type npu --fraction 0.011 --use_sigma True --is_lowbit True --trust_remote_code True
  ```
##### Qwen1.5-72b W8A16量化
  ```shell
  python3 quant_qwen.py --model_path {浮点权重路径} --save_directory {W8A16量化权重路径}  --w_bit 8 --a_bit 16 --device_type npu --trust_remote_code True
  ```
#### 3. Qwen2系列
##### Qwen2-7b W8A8量化
  ```shell
  python3 quant_qwen.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file ../common/boolq.jsonl --w_bit 8 --a_bit 8 --device_type npu --trust_remote_code True
  ```
##### Qwen2-72b W8A16量化
  ```shell
  python3 quant_qwen.py --model_path {浮点权重路径} --save_directory {W8A16量化权重路径}  --w_bit 8 --a_bit 16 --device_type npu --trust_remote_code True
  ```
稀疏量化权重请使用以下指令生成
  ```shell
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False

  python3 quant_qwen.py --model_path {浮点权重路径} --save_directory {W8A8s量化权重路径} --calib_file ../common/cn_en.jsonl --w_bit 4 --a_bit 8 --fraction 0.011 --co_sparse True --device_type npu --use_sigma True --is_lowbit True --trust_remote_code True
  ```

##### Qwen2-72b KV Cache W8A8量化
  ```shell
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False

  python3 quant_qwen.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file ../common/boolq.jsonl --w_bit 8 --a_bit 8 --device_type npu   --use_kvcache_quant True --trust_remote_code True
  ```

#### 4. Qwen2.5系列
##### Qwen2.5-7b, Qwen2.5-14b, Qwen2.5-32b W8A8 量化
  ```shell
  python3 quant_qwen.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file ../common/boolq.jsonl --w_bit 8 --a_bit 8 --device_type npu --trust_remote_code True
  ```
##### Qwen2.5-72B 支持Attention量化
- 需修改`modeling_qwen2.py`文件和`config.json`文件，配置方法参考[FA量化使用说明](https://gitee.com/ascend/msit/blob/master/msmodelslim/docs/FA%E9%87%8F%E5%8C%96%E4%BD%BF%E7%94%A8%E8%AF%B4%E6%98%8E.md)。 
- 相比于W8A8量化，需额外设置`use_fa_quant`参数为True
  ```shell
  python3 quant_qwen.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file ../common/boolq.jsonl --w_bit 8 --a_bit 8 --device_type npu --anti_method m4 --act_method 1 --use_fa_quant True --trust_remote_code True
  ```
##### Qwen2.5-Coder-7B 稀疏量化 
  ```shell
  python3 quant_qwen.py --model_path {浮点权重路径} --save_directory {W4A8量化权重路径} --calib_file ../common/humaneval_x.jsonl --w_bit 4 --a_bit 8 --device_type cpu --fraction 0.02 --co_sparse True  --use_sigma True --is_lowbit False --trust_remote_code True
  ```
##### Qwen2.5-72b W8A8-pdmix量化(prefill阶段 W8A8动态量化, decode阶段 W8A8量化) 搭配 KV cache int8量化
  ```shell
  python3 quant_qwen_pdmix.py --model_path {浮点权重路径} \
  --save_directory {W8A8-pdmix量化权重路径} \
  --calib_file ./calib_data/calib_prompt_72b_pdmix.json  \
  --anti_calib_file ./calib_data/anti_prompt_72b_pdmix.json \
  --device_type npu \
  --anti_method m6 \
  --act_method 2 \
  --use_kvcache_quant True \
  --pdmix True \
  --trust_remote_code True \
  --disable_names model.layers.0.mlp.down_proj model.layers.1.mlp.down_proj model.layers.2.mlp.down_proj model.layers.79.mlp.down_proj
  ```

##### Qwen2.5-72B-Instruct W4A16 量化
当传入 mindie_format 参数时，量化权重不会将 int4 打包成 int8，当不传入 mindie_format 参数时，量化权重将 int4 打包成 int8，减少存储空间，同时减少模型部署时加载时间。
  ```shell
  python quant_qwen.py \
            --model_path {浮点权重路径} \
            --save_directory {量化权重路径} \
            --device_type npu \
            --calib_file ./calib_data/mix_dataset.json \
            --w_bit 4 \
            --a_bit 16 \
            --is_lowbit True \
            --open_outlier False \
            --group_size 128 \
            --anti_method m3 \
            --trust_remote_code True
  ```

#### 5. Qwen3 系列
##### Qwen3-32B W8A8量化

该模型的量化支持已经集成至[一键量化](https://gitee.com/ascend/msit/blob/master/msmodelslim/msmodelslim/app/naive_quantization/readme.md)。

  ```shell
  msmodelslim quant --model_path {浮点权重路径} --save_path {W8A8量化权重路径} --device npu --model_type Qwen3-32B --quant_type w8a8 --trust_remote_code True
  ```

##### Qwen3-32B 稀疏量化

该模型的量化支持已经集成至[一键量化](https://gitee.com/ascend/msit/blob/master/msmodelslim/msmodelslim/app/naive_quantization/readme.md)。

  ```shell
  msmodelslim quant --model_path {浮点权重路径} --save_path {W8A8量化权重路径} --device npu --model_type Qwen3-32B --quant_type w8a8s --trust_remote_code True
  ```

##### Qwen3-32b W4A4 Flatquant Dynamic量化 
  ```shell
  python3 w4a4.py --model_path {浮点权重路径} --save_directory {w4a4量化权重路径} --calib_file ../common/wiki.jsonl --trust_remote_code True
  ```
##### Qwen3-14B W8A8量化

该模型的量化支持已经集成至[一键量化](https://gitee.com/ascend/msit/blob/master/msmodelslim/msmodelslim/app/naive_quantization/readme.md)。

  ```shell
  msmodelslim quant --model_path {浮点权重路径} --save_path {W8A8量化权重路径} --device npu --model_type Qwen3-14B --quant_type w8a8 --trust_remote_code True
  ```
##### Qwen3-14B 稀疏量化

该模型的量化支持已经集成至[一键量化](https://gitee.com/ascend/msit/blob/master/msmodelslim/msmodelslim/app/naive_quantization/readme.md)。

  ```shell
  msmodelslim quant --model_path {浮点权重路径} --save_path {W8A8量化权重路径} --device npu --model_type Qwen3-14B --quant_type w8a8s --trust_remote_code True
  ```

##### Qwen3-8B 稀疏量化

该模型的量化支持已经集成至[一键量化](https://gitee.com/ascend/msit/blob/master/msmodelslim/msmodelslim/app/naive_quantization/readme.md)。

  ```shell
  msmodelslim quant --model_path {浮点权重路径} --save_path {W8A8量化权重路径} --device npu --model_type Qwen3-8B --quant_type w8a8s --trust_remote_code True
  ```

#### QwQ 系列
##### QwQ-32b W8A8量化
  ```shell
  python3 quant_qwen.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file ../common/boolq.jsonl --w_bit 8 --a_bit 8 --device_type npu --anti_method m1
  ```
##### QwQ-32b 稀疏量化 
  ```shell
  python3 quant_qwen.py --model_path {浮点权重路径} --save_directory {W8A8s量化权重路径} --calib_file ../common/cn_en.jsonl --w_bit 4 --a_bit 8 --device_type npu --fraction 0.011 --use_sigma True --is_lowbit True
  ```
