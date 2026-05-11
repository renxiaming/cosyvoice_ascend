# vLLM 服务化性能采集工具
## 概述
- vLLM 是一款广受欢迎的大模型推理框架，具备投机推理和自动前缀缓存等关键功能，使其在学术界和工业界都得到了广泛应用。本工具基于 vLLM，提供性能数据采集能力，结合 msserviceprofiler 的数据解析与可视化能力，进行 vLLM 服务化推理调试调优
- **关键指标特性**
  - 请求状态变化
  - KV Cache 等存储资源消耗情况
  - 组 batch 以及模型执行过程记录
## 版本支持情况
  |  配套CANN版本   | vLLM V0 | vLLM V1 |
  |:-------:|:----:|:----:|
  | 8.2.RC1 |  v0.9.1.RC2  | v0.9.1.RC2 |
  | 8.2.RC1 |  v0.9.1.RC1  | v0.9.1.RC1 |
  | 8.1.RC1 |  v0.8.5.RC1  | / |
  | 8.1.RC1 |  v0.8.4   | / |
  | 8.0.RC3 |  v0.6.3   | / |

## 环境准备
- 根据不同版本需求，参考 [Ascend-vLLM 准备推理环境](https://support.huaweicloud.com/bestpractice-modelarts/modelarts_llm_infer_91203.html)，或 [vLLM Ascend installation](https://vllm-ascend.readthedocs.io/en/latest/installation.html) 成功启动推理服务
- **通过源码方式使用 msserviceprofiler（推荐）**
  ```sh
  git clone https://gitee.com/ascend/msit.git
  export PYTHONPATH=$PWD/msit/msserviceprofiler/:$PYTHONPATH
  ```
- **pip 安装 msserviceprofiler（目前仅支持v0.8.5.RC1及之前的版本）**
  ```sh
  pip install -U msserviceprofiler
  ```
***

# 使用方式
## 1. 在 vLLM 中导入采集工具接口
- 输入 `pip show vllm`，查看 vllm 安装路径，如果存在 `Editable project location` 路径则使用 `Editable project location`，否则为 `Location` 路径 ，记为 `${vllm_install_path}`，
- 在 `${vllm_install_path}/vllm/__init__.py` 文件中所有其他模块导入的最后添加如下代码：
  ```
  import msserviceprofiler.vllm_profiler
  ```
## 2. 准备性能采集配置文件
- 在任意路径下创建使能采集配置 json 文件，如 `ms_service_profiler_config.json`，并指定其中的采集落盘路径 `"prof_dir"`，此处指定为 `vllm_prof`
  ```
  {
    "enable": 1,
    "prof_dir": "vllm_prof",
    "profiler_level": "INFO",
    "host_system_usage_freq": -1,
    "npu_memory_usage_freq": -1,
    "acl_task_time": 0,
    "acl_prof_task_time_level": "",
    "api_filter": "",
    "kernel_filter": "",
    "timelimit": 0,
    "domain": ""
  }
  ```

- 配置文件中支持的参数说明

  |   参数   | 说明                                                                                                                                                                                                                                                                                                                                                                      | 是否必选  | 
  |:------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----:|
  |   enable   | 是否开启性能数据采集的开关：<br/>0：关闭<br/>1：开启<br/>默认值：0                                                                                                                                                                                                                                                                                                                              |   是   |
  |   prof_dir   | 采集到性能数据的存放路径，支持用户自定义。<br/>默认值：$HOME/.ms_service_profiler                                                                                                                                                                                                                                                                                                                |   否   |
  |   profiler_level   | 数据采集等级。默认值为"INFO"，指普通级别的性能数据。                                                                                                                                                                                                                                                                                                                                           |   否   |
  |   host_system_usage_freq   | CPU和内存系统指标采集频率，默认关闭不采集。范围整数1~50，单位hz，表示每秒采集的次数。设置为-1时关闭采集该指标。   <br/>说明：开启该功能可能占用较大内存                                                                                                                                                                                                                                                                                   |   否   |
  |   npu_memory_usage_freq   | NPU Memory使用率指标的采集频率，默认关闭不采集。范围整数1~50，单位hz，表示每秒采集的次数。设置为-1时关闭采集该指标。 <br/>说明：开启该功能可能占用较大内存                                                                                                                                                                                                                                                                               |   否   |
  |   acl_task_time   | 开启采集算子下发耗时、算子执行耗时数据的开关，取值为：<br/>0：关闭。默认值，配置为0或其他非法值均表示关闭。<br/>1：开启。该功能开启时调用aclprofCreateConfig接口的ACL_PROF_TASK_TIME_L0参数。<br/>2：开启基于MSPTI接口的数据落盘。该功能开启时调用MSPTI接口进行性能数据采集，需要配置如下环境变量：export LD_PRELOAD=$ASCEND_TOOLKIT_HOME/lib64/libmspti.so                                                                                                               |   否   |
  |   acl_prof_task_time_level   | 设置性能数据采集的Level等级和时长，取值为：<br/>L0：Level0等级，表示采集算子下发耗时、算子执行耗时数据。与L1相比，由于不采集算子基本信息数据，采集时性能开销较小，可更精准统计相关耗时数据。<br/>L1：Level1等级，采集AscendCL接口的性能数据，包括Host与Device之间、Device间的同步异步内存复制时延；采集算子下发耗时、算子执行耗时数据以及算子基本信息数据，提供更全面的性能分析数据。<br/> time：采集时长，取值范围为1~999的正整数，单位s。<br/>默认未配置本参数，表示采集L0数据，且采集到程序执行结束。配置其他非法值时取默认值。<br/>采集的Level等级和时长可同时配置，例如"acl_prof_task_time_level": "L1,10"。 |   否   |
  |   api_filter   | 对性能数据进行过滤，配置该参数可自定义采集配置的API性能数据，例如传入“matmul”会落盘所有API数据中name字段包含matmul的性能数据。str类型，区分大小写，多个不同的筛选目标用“；”隔开，默认为空，表示落盘所有数据。<br/>仅当acl_task_time参数值为2时生效。                                                                                                                                                                                                                      |   否   |
  |   kernel_filter   | 对性能数据进行过滤，配置该参数可自定义采集配置的kernel性能数据，例如传入“matmul”会落盘所有kernel数据中name字段包含matmul的性能数据。str类型，区分大小写，多个不同的筛选目标用“；”隔开，默认为空，表示落盘所有数据。<br/>仅当acl_task_time参数值为2时生效。                                                                                                                                                                                                                |   否   |
  |   timelimit   | 设置服务化性能数据采集的时长，配置该参数后，采集进程将在运行指定的时间后自动停止，取值范围为0~7200的整数，单位s，默认值0（表示不限制采集时间）                                                                                                                                                                                                                                                                                             |   否   |
  |   domain   | 设置采集指定domain域下的性能数据，减少采集数据量。输入参数为字符串格式，英文分号作为分隔符，区分大小写，例如："Request; KVCache"。<br/>默认为空，表示采集当前所有domain域内性能数据。 <br/>当前已有domain域为：Request、KVCache、ModelExecute、BatchSchedule、Communication。 <br/>说明：<br/>若指定domain域不全，采集数据不满足解析输出件生成时，会有告警提示                                                                                                                                         |   否   |
## 3. 采集数据
1. 指定 `SERVICE_PROF_CONFIG_PATH` 为采集文件配置路径
  ```sh
  export SERVICE_PROF_CONFIG_PATH=ms_service_profiler_config.json
  ```
2. 服务端启动 vLLM 服务，以 `Qwen/Qwen-3B` 为例，使用时以实际启动方式为准
  ```sh
  python3 -m vllm.entrypoints.openai.api_server --model Qwen/Qwen-3B --max-model-len=4096 --trust-remote-code
  ```
3. 客户端发送请求，以curl命令为例，使用时以实际请求的发送形式为准
  ```sh
  curl -X POST http://${docker_ip}:8080/generate -H "Content-Type: application/json" \
  -d '{"prompt": "hello", "max_tokens": 100, "temperature": 0}'
  ```
## 4. 查看采集结果
- 请求发送结束后，可在配置文件中 `"prof_dir"` 指定的路径下，查看落盘的性能原始数据，其中包含了 db 格式的数据库落盘文件
- **执行推理处理时间数据**

  | 字段      | 含义             | 内容说明                                                  |
  | --------- | ---------------- | --------------------------------------------------------- |
  | modelExec | 模型执行时间     | rid: 请求ID, batch_type: batch类型, batch_size: batch大小 |
  | Forward   | 模型前向计算时间 | rid: 请求ID                                               |

- **请求队列管理状态变化，及组 batch 过程数据**

  | 字段             | 含义                 | 内容说明                                                                                                                           |
  | ---------------- | -------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
  | Enqueue, Dequeue | 表示请求入队、出队   | rid: 请求ID, QueueSize：当前队列大小, scope：队列名称，通常含有WAITING、RUNNING队列                                                |
  | BatchSchedule    | 表示调度信息         | rid: 当前调度batch中的请求ID列表, QueueSize：当前队列大小, iter_size:当前迭代返回token长度                                         |
  | ReqState         | 表示请求状态变化信息 | rid: 请求ID, WAITING+ 等待 / RUNNING+ 执行 / FINISHED+ 结束：请求状态名，值为1表示当前状态，值为-1表示前一状态，值为+1表示后一状态 |

- **kv cache 数据**

  | 字段            | 含义                                 | 内容说明                                                                               |
  | --------------- | ------------------------------------ | -------------------------------------------------------------------------------------- |
  | Allocate        | 表示请求分配kvcache block字段        | domain：表示当前为kvcache相关信息, rid: 请求ID, deviceBlock: 分配的block数量           |
  | AppendSlot      | 表示请求过程中新增内存进行缓存的字段 | domain：表示当前为kvcache相关信息, rid: 请求ID, deviceBlock: 追加的block数量           |
  | Free            | 请求过程中释放的缓存字段             | domain：表示当前为kvcache相关信息, rid: 请求ID, deviceBlock: 释放的block数量           |
  | GetCacheHitRate | 请求过程中缓存的命中率               | domain：表示当前为kvcache相关信息, cpuHitCache: cpu缓存命中率, hitCache: npu缓存命中率 |

- **request 数据**

  | 字段    | 含义         | 内容说明                                                                                                           |
  | ------- | ------------ | ------------------------------------------------------------------------------------------------------------------ |
  | httpReq | 表示请求到达 | domain：表示当前为http请求相关信息, rid: 请求ID                                                                    |
  | httpRes | 表示请求返回 | domain：表示当前为http请求相关信息, rid: 请求ID, recvTokenSize：表示请求输入长度, replyTokenSize：表示请求输出长度 |
## 5. 数据解析
- 需要使用 CANN toolkit 中的 `ms_service_profiler` 工具，详细结果参照 CANN 手册
- 调用方式，如落盘路径为 `vllm_prof`
  ```sh
  # 获取最近一次的落盘数据路径，也可以手动指定
  LATEST_PROF_PATH=`ls vllm_prof/* -1td | head -n 1` && echo "LATEST_PROF_PATH=$LATEST_PROF_PATH"

  # 处理路径权限
  sudo chown $USER:$USER $LATEST_PROF_PATH -R && sudo chmod g-w $LATEST_PROF_PATH -R && sudo chmod o-w $LATEST_PROF_PATH -R

  # 调用解析，其中 `--input-path` 为落盘数据路径，`--output-path` 为解析后输出路径
  python -m ms_service_profiler.parse --input-path=$LATEST_PROF_PATH --output-path  output
  ```
  将在 `--output-path` 下生成解析后文件 `profiler.db` `chrome_tracing.json` `request.csv` `kvcache.csv` `batch.csv`，具体内容可参照 CANN 手册
