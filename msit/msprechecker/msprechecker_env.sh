#!/bin/bash
# --------------------------------------------------
# 环境变量管理脚本 (直接 source 执行)
# 使用方式:
#   source msprechecker_env.sh    # 应用预期配置
#   source msprechecker_env.sh 0  # 还原为原始状态
# --------------------------------------------------

if [ "$1" = "0" ]; then
    export MIES_CONTAINER_IP="<missing>"
    export PYTORCH_NPU_ALLOC_CONF="<missing>"
    export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE="<missing>"
    export ATB_WORKSPACE_MEM_ALLOC_GLOBAL="<missing>"
    export HCCL_OP_EXPANSION_MODE="<missing>"
    export NPU_MEMORY_FRACTION="<missing>"
    export ATB_LLM_HCCL_ENABLE="<missing>"
    export ATB_LAYER_INTERNAL_TENSOR_REUSE="<missing>"
    export HCCL_CONNECT_TIMEOUT="<missing>"
    export HCCL_EXEC_TIMEOUT="<missing>"
    export ATB_LLM_ENABLE_AUTO_TRANSPOSE="<missing>"
    export MINDIE_ASYNC_SCHEDULING_ENABLE="<missing>"
    export MINDIE_LOG_TO_FILE="<missing>"
    export MINDIE_LOG_TO_STDOUT="<missing>"
    export OMP_NUM_THREADS="<missing>"
else
    export MIES_CONTAINER_IP="172.18.50.239" # MIES_CONTAINER_IP表示当前容器IP地址，建议设置为当前容器IP
    export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True" # 需要开启torch_npu虚拟内存机制
    export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE="3" # workspace内存分配算法选择，建议设置为3，最大优化显存碎片与workspace空间
    export ATB_WORKSPACE_MEM_ALLOC_GLOBAL="1" # 建议开启全局中间tensor内存分配算法，提升显存利用率
    export HCCL_OP_EXPANSION_MODE="AIV" # HCCL_OP_EXPANSION_MODE建议设置为AIV，设置通信算法的编排展开位置在Device侧的AI Vector Core计算单元
    export NPU_MEMORY_FRACTION="0.96" # A3单机DeepSeek场景下，NPU显存比建议设置为0.96，可根据实际业务场景加大，出现out of memory时可以尝试调大该值
    export ATB_LLM_HCCL_ENABLE="1" # 需要开启HCCL通信后端
    export ATB_LAYER_INTERNAL_TENSOR_REUSE="1" # 需要开启复用Layer间的中间Tensor复用
    export HCCL_CONNECT_TIMEOUT="7200" # HCCL建链超时时间建议设置为7200秒
    export HCCL_EXEC_TIMEOUT="0" # HCCL执行超时时间建议设置为0，不限制超时
    export ATB_LLM_ENABLE_AUTO_TRANSPOSE="0" # 不能开启权重右矩阵自动转置
    export MINDIE_ASYNC_SCHEDULING_ENABLE="1" # 建议开启MindIE异步调度特性，提升推理性能
    export MINDIE_LOG_TO_FILE="1" # 建议将MindIE日志写入文件，便于排查问题
    export MINDIE_LOG_TO_STDOUT="1" # 建议将MindIE日志打屏，便于通过查看k8s日志查看程序运行状态
    export OMP_NUM_THREADS="16" # OpenMP并行数建议设置为16
fi
