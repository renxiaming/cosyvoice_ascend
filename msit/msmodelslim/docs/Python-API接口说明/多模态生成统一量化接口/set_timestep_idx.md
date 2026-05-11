## set_timestep_idx()

### 功能说明
在时间步量化中，需要针对不同时间步(timestep)采用不同的量化策略(w8a8静态或w8a8动态)，此函数用于设置当前量化进行到的时间步。

### 函数原型
```python
set_timestep_idx(t_idx: int)
```

### 参数说明
| 参数名 | 输入/返回值 | 含义 | 使用限制 |
| ------ | ---------- | ---- | -------- |
| t_idx | 输入 | 多模态生成模型推理的当前时间步。 | 必选。<br>数据类型：int。|

### 调用示例

```python
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.timestep.manager import TimestepManager

for step_id, t in enumerate(timesteps):
    TimestepManager.set_timestep_idx(step_id)
    model_output = pipeline(...)
```