# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os 
import sys
import json
import time
import re
import multiprocessing as mp
import threading

import numpy as np
import pandas as pd
from tqdm import tqdm

from speculative_moe import ExpSolver, ExpILPSolver, second_optim, all_to_all_algorithm_multi_process
from components.utils.file_open_check import ms_open
from components.utils.constants import JSON_FILE_MAX_SIZE
from components.utils.security_check import ms_makedirs
from components.utils.log import logger
from components.expert_load_balancing.elb.constant import A2, A3, SUPPORTED_COMBINATIONS, \
                        ALGORITHM_SPECULATIVE_MOE_LEVEL_1, ALGORITHM_SPECULATIVE_MOE_LEVEL_2, \
                        ALGORITHM_SPECULATIVE_MOE_LEVEL_1_MIXED, ALGORITHM_SPECULATIVE_MOE_LEVEL_2_MIXED


class AppArgs:
    def __init__(self, 
                 expert_popularity_csv_load_path,
                 output_dir,
                 num_nodes,
                 num_npus,
                 share_expert_devices,
                 num_redundancy_expert,
                 algorithm,
                 device_type):

        self.trace_path = expert_popularity_csv_load_path
        self.deploy_fp = output_dir
        self.n_nodes = num_nodes
        self.n_devices = num_npus
        self.n_share_expert_devices = share_expert_devices
        self.redundant_experts = num_redundancy_expert
        self.algorithm = algorithm
        self.device_type = device_type
        self.selected_layers = [-1, -1]
        self.n_layers = 58
        self.n_experts = 0
        self.max_time_in_seconds = 300
        self.eplb_map = "./"
        self.n_selected_expert = 8
        self.collection_interval = 16
        self.cpu_per_process = 12
        self.num_stages = 8
        self.enhanced = False
        self.black_box_annealing = False
        self.all2all_balance = False
        self.n_shared_experts = 1
        self.mixed_shared_expert = False


def check_input(new_args):
    if (new_args.redundant_experts + new_args.n_experts) % new_args.n_devices != 0:
        raise ValueError("The sum of origin expert and redundant expert must be a positive multiple of devices.")


def validate_args(new_args):
    supported_algorithms = SUPPORTED_COMBINATIONS.get(new_args.device_type)
    if supported_algorithms is None:
        raise ValueError(f"device '{new_args.device_type}' is not supported.")

    if new_args.algorithm not in supported_algorithms:
        raise ValueError(
            f"device '{new_args.device_type}' does not support algorithm '{new_args.algorithm}'."
        )


def numerical_sort_key(filename):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', filename)]


def parse_ep_file(ep_file_path, ep_file=None, n_share_expert_devices=0):
    experts_table = {}
    if ep_file is None:
        with ms_open(ep_file_path, max_size=JSON_FILE_MAX_SIZE) as handle:
            ep_file = json.load(handle)

    layer_count = ep_file["moe_layer_count"]
    layer_list = ep_file["layer_list"]
    for layer_list_idx in range(layer_count):
        layer_info = layer_list[layer_list_idx]
        layer_id = layer_info["layer_id"]
        device_count = layer_info["device_count"]
        device_list = layer_info["device_list"]
        layer_expert_table = []
        for device_list_idx in range(device_count):
            device_info = device_list[device_list_idx]
            device_expert = device_info["device_expert"]
            layer_expert_table.append(device_expert)

        experts_table[layer_id] = layer_expert_table[n_share_expert_devices:]

    return experts_table


def convert(data):
    if isinstance(data, (np.integer, np.floating)):
        return data.item()  # 转为 Python 原生 int/float
    elif isinstance(data, np.ndarray):
        return data.tolist()  # 如果是 numpy array，转 list
    raise TypeError(f"Object of type {type(data)} is not JSON serializable")


def generate_json(data, new_args):
    ms_makedirs(new_args.deploy_fp, exist_ok=True)
    if new_args.has_decode:
        json_file = os.path.join(new_args.deploy_fp, "decode_global_deployment.json")
        with ms_open(json_file, "w") as f:
            json.dump(data, f, indent=4, default=convert)
        logger.info(f"save in {new_args.deploy_fp}")
    if new_args.has_prefill:
        json_file = os.path.join(new_args.deploy_fp, "prefill_global_deployment.json")
        with ms_open(json_file, "w") as f:
            json.dump(data, f, indent=4, default=convert)
        logger.info(f"save in {new_args.deploy_fp}")


def has_prefill_decode(new_args):
    has_decode = False
    has_prefill = False

    folder_path = new_args.trace_path
    # 第一次扫描：判断是否存在 decode_/prefill_ 文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            if filename.startswith('decode_'):
                has_decode = True
            elif filename.startswith('prefill_'):
                has_prefill = True
    if has_decode and has_prefill:
        raise ValueError("Make sure you only have one of decode and prefill")
    if not has_decode and not has_prefill:
        raise ValueError("No decode and prefill files")
    return has_decode, has_prefill


def refresh_dependent_args(new_args):
    # 如果有config，则是传入新的mindie文件,num_stages是8, 四种场景的配置 a2和a3 × base和enhance
    new_args.n_share_expert_devices_files = new_args.n_share_expert_devices
    if os.path.isfile(os.path.join(new_args.trace_path, "model_gen_config.json")):
        path = new_args.trace_path.rstrip('/')
        dirname = os.path.dirname(path)
        basename = os.path.basename(path) + "_selected"
        new_args.trace_path = os.path.join(dirname, basename)
        new_args.dump_vresion = False
        has_decode, has_prefill = has_prefill_decode(new_args)
        new_args.has_decode = has_decode
        new_args.has_prefill = has_prefill
        with ms_open(os.path.join(new_args.trace_path, "model_gen_config.json")) as handle:
            config = json.load(handle)
            if "num_moe_layers" in config:
                new_args.n_layers = config["num_moe_layers"]
            if "num_of_experts" in config:
                new_args.n_experts = config["num_of_experts"]
            if "eplb_expert_map_file" in config:
                new_args.eplb_map = config["eplb_expert_map_file"]
            if not os.path.exists(new_args.eplb_map):
                raise FileExistsError("Check the eplb expert map file path in the config.")
            if "collection_Interval" in config:
                new_args.collection_interval = config["collection_Interval"]
            if "num_of_selected_expert" in config:
                new_args.n_selected_expert = config["num_of_selected_expert"][0]
            if "num_dangling_shared_experts" in config:
                new_args.n_share_expert_devices_files = max(config["num_dangling_shared_experts"], 0)
            new_args.num_stages = 8
            if new_args.algorithm == ALGORITHM_SPECULATIVE_MOE_LEVEL_2 and \
                new_args.device_type == A2:
                new_args.enhanced = True
                new_args.black_box_annealing = True
                new_args.all2all_balance = True
            elif new_args.algorithm == ALGORITHM_SPECULATIVE_MOE_LEVEL_2 and \
                new_args.device_type == A3:
                new_args.enhanced = True
                new_args.black_box_annealing = True
                new_args.all2all_balance = False
            elif new_args.algorithm == ALGORITHM_SPECULATIVE_MOE_LEVEL_1:
                new_args.enhanced = False
                new_args.black_box_annealing = False
                new_args.all2all_balance = False
            
            elif new_args.algorithm == ALGORITHM_SPECULATIVE_MOE_LEVEL_1_MIXED and \
                new_args.device_type == A2:
                new_args.enhanced = False
                new_args.black_box_annealing = False
                new_args.all2all_balance = False
                new_args.n_experts += new_args.n_shared_experts
                new_args.redundant_experts += new_args.n_devices - 1
                new_args.mixed_shared_expert = True

            elif new_args.algorithm == ALGORITHM_SPECULATIVE_MOE_LEVEL_2_MIXED and \
                new_args.device_type == A2:
                new_args.enhanced = True
                new_args.black_box_annealing = True
                new_args.all2all_balance = True
                new_args.n_experts += new_args.n_shared_experts
                new_args.redundant_experts += new_args.n_devices - 1
                new_args.mixed_shared_expert = True

    # 旧版本的输入文件
    # num_stages为1，
    # 四种场景的配置 a2和a3 × base和enhance
    else:
        new_args.dump_vresion = True
        new_args.enhanced = False
        new_args.eplb_map = ""
        new_args.num_stages = 1
        new_args.black_box_annealing = False
        new_args.all2all_balance = False
        if new_args.algorithm == "3":
            new_args.enhanced = True
        has_decode, has_prefill = has_prefill_decode(new_args)
        new_args.has_decode = has_decode
        new_args.has_prefill = has_prefill
    # selected_layers
    if new_args.selected_layers == [-1, -1]:
        new_args.selected_layers = [0, new_args.n_layers - 1]

    return new_args


def get_csv_path(csv_file_path, file_name):
    """
    从指定目录得到decode_info.csv和prefill_info.csv文件路径。
    
    参数:
    - csv_file_path: 包含CSV文件的目录路径。
    
    返回:
    - decode_df: decode_info.csv 的内容，若文件不存在或读取失败则为 None。
    - prefill_df: prefill_info.csv 的内容，若文件不存在或读取失败则为 None。
    """
    file_path = os.path.join(csv_file_path, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist."
                                f"Check whether the output path in the input parameter -o is correct.")

    return file_path


def get_csv_dimensions(input_csv_path):
    """
    读取CSV文件并返回其行数（第0维）和列数（第1维）。

    参数:
        input_csv_path (str): CSV文件路径。

    返回:
        tuple: (行数, 列数) 的元组，失败时返回 (None, None)。
    """
    dimensions = (None, None)
    try:
        df = pd.read_csv(input_csv_path)
        if df.empty:
            raise ValueError(f"文件包含空数据 {input_csv_path}")
        row_count = df.shape[0]  # 第0维（行数）58
        col_count = df.shape[1]  # 第1维（列数）256
        dimensions = (row_count, col_count)
        
        logger.info(f"Successfully read: {input_csv_path}.")
        logger.info(f"The input file: Number of row={row_count}, Number of columns={col_count}.")
        
    except pd.errors.EmptyDataError as e:
        raise ValueError(f"文件无有效数据 {input_csv_path}") from e
    except pd.errors.ParserError as e:
        raise ValueError(f"CSV格式解析失败 {input_csv_path}") from e
    except Exception as e:
        raise RuntimeError(f"未知错误") from e
    
    return dimensions


def get_dynamic_expert_hot_from_csv(
        root_folder,
        n_layer=58,
        n_expert=256,
        eplb_map="",
        n_share_expert_devices_files=0,
        n_selected_expert=8,
        collection_interval=16,
        topk_info=False,
        n_devices=-1,
        mse=False,
        n_shared_experts=1,
):
    def get_expert_id(layer, idx):
        if isinstance(eplb_map, dict):  # 处理已解析好的 map
            deployment = np.array(eplb_map[layer]).flatten()
            return deployment[idx]
        return idx

    # 如果是路径字符串，则解析部署文件
    if isinstance(eplb_map, str) and os.path.isfile(eplb_map):
        eplb_map = parse_ep_file(eplb_map, n_share_expert_devices=n_share_expert_devices_files)

    # 收集 decode_*.csv（排除 decode_topk_*.csv）
    all_files = os.listdir(root_folder)
    hotness_files = sorted([
        os.path.join(root_folder, f)
        for f in all_files
        if (f.startswith("decode_") or f.startswith("prefill_")) and "topk" not in f and f.endswith(".csv")
    ], key=numerical_sort_key)

    # 加载 csv 数据并拼接为三维数组
    data_list = [np.loadtxt(f, delimiter=",", dtype=int) for f in hotness_files]
    min_size = min(len(d) for d in data_list)
    data = np.asarray([d[:min_size] for d in data_list[n_share_expert_devices_files:]])

    if len(data.shape) == 2:
        data = data[..., np.newaxis]
    data = data.transpose(1, 0, 2)
    # 划分迭代，差分处理
    n_iteration = data.shape[0] // n_layer
    data = data[:n_iteration * n_layer].reshape((n_iteration, n_layer, -1))
    data[1:] -= data[:-1].copy()
    # 动态热度筛选
    heat = data.sum(axis=(1, 2))
    threshold = 2 * len(hotness_files) * n_selected_expert * collection_interval * n_layer
    mask = heat > threshold
    data = data[mask]
    n_iteration = data.shape[0]
    # 构造动态专家热度矩阵
    dynamic_expert_hot = np.zeros((n_layer, n_expert, n_iteration), dtype=int)

    for j in range(n_layer):
        for k in range(data.shape[2]):
            expert_id = get_expert_id(j, k)
            dynamic_expert_hot[j, expert_id, :] += data[:, j, k]
    
    if "prefill" in all_files[0]:
        logger.info("caculate expert 0 hot in prefill")
        dynamic_expert_hot[:, 0] = np.mean(dynamic_expert_hot)

    if mse:
        shared_expert_hotness = dynamic_expert_hot[:, :-n_shared_experts].sum(1) / 8
        for j in range(1, n_shared_experts + 1):
            dynamic_expert_hot[:, -j] = shared_expert_hotness

    topk_files = sorted([
        os.path.join(root_folder, f)
        for f in all_files
        if "topk" in f and f.endswith(".csv")
    ], key=numerical_sort_key)

    # 如果需要 topk 信息
    if topk_info and topk_files:
        topk_data_list = [np.loadtxt(f, delimiter=",", dtype=np.float32) for f in topk_files]
        min_topk_size = min(min(len(k) for k in topk_data_list), min_size)
        topk_data = np.asarray([d[:min_topk_size] for d in topk_data_list[n_share_expert_devices_files:]])
        topk_data = topk_data.transpose(1, 0, 2)  # (iteration, layer, topk)
        topk = topk_data.shape[-1]

        n_iteration_topk = topk_data.shape[0] // n_layer
        topk_data = topk_data[:n_iteration_topk * n_layer].reshape((n_iteration_topk, n_layer, -1, topk))
        topk_data = topk_data[mask[:min_topk_size]]
        topk_data = topk_data.transpose(1, 2, 0, 3)  # (layer, expert, iteration, topk)
        if n_devices != -1:
            topk_data = topk_data.reshape((n_layer, -1, topk))
            trace_total = topk_data.shape[1]
            trace_per_device = trace_total // n_devices
            if trace_per_device == 0:
                raise FileNotFoundError(
                    "all2all_balance optimization needs topk data, but topk data is not enough."
                )
            topk_data = topk_data[:, :trace_per_device * n_devices, :]
            topk_data = topk_data.reshape(n_layer, n_devices, -1, topk)
        # 随机采样 32 个迭代点
        if topk_data.shape[2] >= 32:
            sampled_indices = np.random.choice(topk_data.shape[2], size=32, replace=False)
        else:
            sampled_indices = np.arange(topk_data.shape[2])  # 不足 32，返回所有
        return dynamic_expert_hot, topk_data[:, :, sampled_indices]
    return dynamic_expert_hot, None


def process_speculative_moe(args):
    valid_keys = {
        'expert_popularity_csv_load_path',
        'output_dir',
        'num_nodes',
        'num_npus',
        'share_expert_devices',
        'num_redundancy_expert',
        'algorithm',
        'device_type'
    }
    filtered_args = {k: v for k, v in vars(args).items() if k in valid_keys}
    new_args = AppArgs(**filtered_args)
    validate_args(new_args)
    new_args = refresh_dependent_args(new_args)
    check_input(new_args)
    # 如果是旧版本的情况需要传入层数和专家数，且需要将文件夹拆分为decode或者prefill
    if new_args.dump_vresion:
        logger.info("old version")
        if new_args.has_decode:
            logger.info("has decode")  
            input_csv_path = get_csv_path(new_args.deploy_fp, "decode_info.csv")
            dimensions = get_csv_dimensions(input_csv_path)  
            new_args.n_layers = dimensions[0]
            new_args.n_experts = dimensions[1]
            process_prefill_or_decode(new_args)
        if new_args.has_prefill:
            logger.info("has prefill")
            input_csv_path = get_csv_path(new_args.deploy_fp, "prefill_info.csv")
            dimensions = get_csv_dimensions(input_csv_path)
            new_args.n_layers = dimensions[0]
            new_args.n_experts = dimensions[1] 
            process_prefill_or_decode(new_args)
    else:
        logger.info("new version")
        process_prefill_or_decode(new_args)


def process_prefill_or_decode(new_args):
    start_time = time.time()
    logger.info("<<< loading data >>>")
    shared_status = dict()
    total_steps = 6  # 总共六个主要阶段
    progress_bar = tqdm(total=total_steps, desc="Processing Prefill/Decode", unit="step")
    expert_hotness, expert_topk = get_dynamic_expert_hot_from_csv(
        new_args.trace_path,
        n_layer=new_args.n_layers,
        n_expert=new_args.n_experts,
        eplb_map=new_args.eplb_map,
        n_share_expert_devices_files=new_args.n_share_expert_devices_files,
        n_selected_expert=new_args.n_selected_expert,
        collection_interval=new_args.collection_interval,
        topk_info=new_args.all2all_balance,
        n_devices=new_args.n_devices,
        mse=new_args.mixed_shared_expert,
        n_shared_experts=new_args.n_shared_experts
    )
    progress_bar.update(1)  # Step 1 
    model = ExpSolver(expert_hotness, new_args, shared_status)
    d2e_tables, objs, n_duplicates_basic, n_duplicates_enhanced, deploy_table = model.fit(1)
    progress_bar.update(1)  # Step 2 

    # 使用增强型算法
    if new_args.enhanced:
        ilp_model = ExpILPSolver(expert_hotness, new_args, shared_status, n_duplicates_basic, n_duplicates_enhanced,
                                 d2e_tables)
        d2e_tables, objs, deploy_table = ilp_model.fit(cpu_per_process=new_args.cpu_per_process)
    progress_bar.update(1)  # Step 3

    # 使用增强型算法
    if new_args.black_box_annealing:
        for i in range(new_args.selected_layers[0], new_args.selected_layers[1] + 1):
            shared_status[i] = ("Black Box Annealing.", "")
        deploy_table = second_optim(deploy_table, expert_hotness, new_args, shared_status)
    progress_bar.update(1)  # Step 4 

    # 使用分层all2all增强型算法
    if new_args.all2all_balance and expert_topk is not None:
        for i in range(new_args.selected_layers[0], new_args.selected_layers[1] + 1):
            shared_status[i] = ("All2AllBalance.", "")
        deploy_table = all_to_all_algorithm_multi_process(deploy_table, expert_topk,
                                                          new_args, shared_status, new_args.selected_layers,
                                                          cpu_per_process=1)
    progress_bar.update(1)  # Step 5 done
    generate_json(deploy_table, new_args)
    progress_bar.update(1)  # Step 6 dones
    progress_bar.close()
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"running time: {elapsed_time / 60} min")