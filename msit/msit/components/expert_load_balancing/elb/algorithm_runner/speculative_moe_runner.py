# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import os
import json

import numpy as np

from components.expert_load_balancing.elb.data_loader.base_loader import DataType
from components.expert_load_balancing.elb.algorithm_runner.base_algorithm_runner import BaseAlgorithmRunner, \
    AlgorithmType, DEPLOYMENT_JSON_FILE
from components.expert_load_balancing.elb.constant import A2, A3
from components.utils.security_check import check_int, get_valid_read_path
from components.utils.file_open_check import ms_open
from speculative_moe import ExpSolver, ExpILPSolver, second_optim, all_to_all_algorithm_multi_process


class SpeculativeArgs:
    def __init__(self, args):
        self.enhanced = False
        self.black_box_annealing = False
        self.all2all_balance = False
        self.num_stages = 8

        self.mixed_shared_expert = False

        self.cpu_per_process = 12
        self.max_time_in_seconds = 300

        self.n_layers = 58
        self.n_experts = 256
        self.collection_interval = 8
        self.n_selected_expert = 8
        self.eplb_map = ""
        self.n_shared_experts = 1  # deepseek固定为1 后续修改支持多种模型

        # 从命令行的args中加载参数
        self.device_type = args.device_type
        self.deploy_fp = args.output_dir
        self.n_nodes = args.num_nodes
        self.n_devices = args.num_npus
        self.redundant_experts = args.num_redundancy_expert
        self.n_share_expert_devices = args.share_expert_devices
        self.n_share_expert_devices = 0
        self.n_share_expert_devices_files = args.share_expert_devices

        if "config_json" in args:
            self.process_split_format_args(args)
        else:
            self.process_sum_format_args(args)

        self.selected_layers = [0, self.n_layers - 1]

    def process_split_format_args(self, args):
        self.load_from_config_json(args.config_json)

        if args.algorithm == AlgorithmType.SPECULATIVE_MOE_LEVEL_2 and \
            self.device_type == A2:
            self.enhanced = True
            self.black_box_annealing = True
            self.all2all_balance = True
        elif args.algorithm == AlgorithmType.SPECULATIVE_MOE_LEVEL_2 and \
            self.device_type == A3:
            self.enhanced = True
            self.black_box_annealing = True
            self.all2all_balance = False
        elif args.algorithm == AlgorithmType.SPECULATIVE_MOE_LEVEL_1:
            self.enhanced = False
            self.black_box_annealing = False
            self.all2all_balance = False
        elif args.algorithm == AlgorithmType.SPECULATIVE_MOE_LEVEL_1_MIXED and \
            self.device_type == A2:
            self.enhanced = False
            self.black_box_annealing = False
            self.all2all_balance = False
            self.redundant_experts += self.n_devices - 1
            self.mixed_shared_expert = True
        elif args.algorithm == AlgorithmType.SPECULATIVE_MOE_LEVEL_2_MIXED and \
            self.device_type == A2:
            self.enhanced = True
            self.black_box_annealing = True
            self.all2all_balance = True
            self.redundant_experts += self.n_devices - 1
            self.mixed_shared_expert = True

    def load_from_config_json(self, config):
        self.n_layers = config.get("num_moe_layers", 1)
        check_int(self.n_layers, min_value=1, param_name="num_moe_layers")
        self.n_experts = config.get("num_of_experts", 1)
        check_int(self.n_experts, min_value=1, param_name="num_of_experts")
        self.collection_interval = config.get("collection_Interval", 8)
        check_int(self.collection_interval, min_value=1, param_name="collection_Interval")
        num_selected_expert = config.get("num_of_selected_experts", None)
        if num_selected_expert and isinstance(num_selected_expert, list):
            self.n_selected_expert = num_selected_expert[0]
            check_int(self.n_selected_expert, min_value=1, param_name="num_of_selected_expert")
        else:
            raise ValueError("num_of_selected_expert in model_gen_config.json should be valued list.")
        n_shared_expert_devices_files = config.get("num_dangling_shared_experts", 0)
        check_int(n_shared_expert_devices_files, param_name="num_dangling_shared_experts")
        self.n_share_expert_devices_files = max(n_shared_expert_devices_files, 0)

        eplb_map_path = config.get("eplb_expert_map_file", None)
        if eplb_map_path is not None:
            eplb_map_path = get_valid_read_path(eplb_map_path)
            self.eplb_map = parse_ep_file(eplb_map_path, n_share_expert_devices=self.n_share_expert_devices_files)
        else:
            self.eplb_map = None

    def process_sum_format_args(self, args):
        self.num_stages = 1
        if args.algorithm == AlgorithmType.SPECULATIVE_MOE_LEVEL_2:
            self.enhanced = True
        self.n_experts = args.n_experts
        self.n_layers = args.n_layers


class SpeculativeMoeRunner(BaseAlgorithmRunner):
    def __init__(self, args):
        super().__init__(args)
    
    def run_algorithm(self, data):

        for period in ["prefill", "decode"]:
            period_data = data.get(period, None)
            if period_data is None:
                continue
            algorithm_args = SpeculativeArgs(self.args)

            period_data, topk_data = \
                process_data(period_data, algorithm_args, self.args.data_type, data.get(period + "_topk", None))
            shared_status = dict()

            model = ExpSolver(period_data, algorithm_args, shared_status)
            d2e_tables, objs, n_duplicates_basic, n_duplicates_enhanced, deploy_table = model.fit(1)

            if algorithm_args.enhanced:
                ilp_model = ExpILPSolver(
                    period_data, 
                    algorithm_args, 
                    shared_status, 
                    n_duplicates_basic, 
                    n_duplicates_enhanced, 
                    d2e_tables
                    )
                d2e_tables, objs, deploy_table = ilp_model.fit(cpu_per_process=algorithm_args.cpu_per_process)

            if algorithm_args.black_box_annealing:
                for i in range(algorithm_args.selected_layers[0], algorithm_args.selected_layers[1] + 1):
                    shared_status[i] = ("Black Box Annealing.", "")
                deploy_table = second_optim(deploy_table, period_data, algorithm_args, shared_status)

            if algorithm_args.all2all_balance and topk_data is not None:
                for i in range(algorithm_args.selected_layers[0], algorithm_args.selected_layers[1] + 1):
                    shared_status[i] = ("All2AllBalance.", "")
                deploy_table = all_to_all_algorithm_multi_process(
                    deploy_table, 
                    topk_data, 
                    algorithm_args, 
                    shared_status, 
                    algorithm_args.selected_layers, 
                    cpu_per_process=1
                    )
            output_file = os.path.join(self.args.output_dir, DEPLOYMENT_JSON_FILE.format(period))
            self.save_json(deploy_table, output_file, convert)


def process_data(data, args, data_type, topk_data=None):
    if data_type == DataType.MINDIE_SPLITED_CSV or data_type == DataType.MINDIE_SPLITED_CSV_WITH_TOPK:
        return process_split_data(data, args, topk_data)
    else:
        return process_sum_data(data), None


def process_split_data(target_data, args, topk_data):
    rank_num = len(target_data)
    target_data = np.concatenate(target_data, axis=-1)
    length = target_data.shape[0] // args.n_layers * args.n_layers
    target_data = target_data[:length, :]
    if args.n_experts != target_data.shape[1]:
        raise ValueError("Shape of exeprt hot does not match num_epxerts in model_gen_config.json.")
    target_data = target_data.reshape([-1, args.n_layers, args.n_experts])  # iteration, num_layers, total_experts
    target_data[1:] -= target_data[:-1].copy()  # 相邻取差，data从累加数据变为单次数据
    iteration = target_data.shape[0]
    heat = target_data.sum(axis=(1, 2))
    threadshold = 2 * rank_num * args.n_selected_expert * args.collection_interval * args.n_layers
    mask = heat > threadshold
    target_data = target_data[mask]
    iteration = target_data.shape[0]

    if topk_data is not None:
        topk_length = min(min([data.shape[0] for data in topk_data]), length)
        topk_iteration = topk_length // args.n_layers
        topk_length = topk_iteration * args.n_layers
        topk_data = np.array([data[:topk_length, :] for data in topk_data[args.n_share_expert_devices_files:]])

        # shape: [rank, iteration * layer, topk] -> [iteration * layer, rank, topk]
        topk_data = topk_data.transpose([1, 0, 2])
        n_topk = topk_data.shape[-1]
        topk_data = topk_data.reshape([topk_iteration, args.n_layers, -1, n_topk])

        topk_data = topk_data[mask[:topk_length]]
        topk_data = topk_data.transpose([1, 2, 0, 3])
        if args.n_devices != -1:
            topk_data = topk_data.reshape([args.n_layers, -1, n_topk])
            trace_total = topk_data.shape[1]
            trace_per_device = trace_total // args.n_devices
            if trace_per_device == 0:
                raise FileNotFoundError("All2all_balance optimization needs topk data, but topk data is not enough.")
            topk_data = topk_data[:, :trace_per_device * args.n_devices, :]
            topk_data = topk_data.reshape([args.n_layers, args.n_devices, -1, n_topk])
        # 随机采样32个点
        if topk_data.shape[2] >= 32:
            sampled_index = np.random.choice(topk_data.shape[2], size=32, replace=False)
        else:
            sampled_index = np.arange(topk_data.shape[2])
        sampled_topk = topk_data[:, :, sampled_index]
    else:
        sampled_topk = None
    
    if args.mixed_shared_expert:
        args.n_experts += args.n_shared_experts
    
    dynamic_expert_hot = np.zeros((args.n_layers, args.n_experts, iteration), dtype=int)

    if args.eplb_map is None:
        for i in range(args.n_layers):
            for j in range(target_data.shape[-1]):
                dynamic_expert_hot[i, j, :] += target_data[:, i, j]
    else:
        if dynamic_expert_hot.shape[:2] != args.eplb_map.shape:
            raise ValueError("eplb_expert_map_file does not match args in model_gen_config.json.")
        for i in range(args.n_layers):
            for j in range(target_data.shape[-1]):
                expert_id = args.eplb_map[i][j]
                dynamic_expert_hot[i, expert_id, :] += target_data[:, i, j]
    
    if args.mixed_shared_expert:
        shared_expert_hotness = dynamic_expert_hot[:, :-args.n_shared_experts].sum(1) / args.n_selected_expert
        for i in range(1, args.n_shared_experts + 1):
            dynamic_expert_hot[:, -i] = shared_expert_hotness
    return dynamic_expert_hot, sampled_topk


def process_sum_data(target_data):
    return target_data[..., np.newaxis]


def parse_ep_file(ep_file_path, ep_file=None, n_share_expert_devices=0):
    experts_table = {}
    if ep_file is None:
        with ms_open(ep_file_path) as handle:
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
   
    sorted_experts = sorted(experts_table.items(), key=lambda x: x[0])
    experts_table = [value for _, value in sorted_experts]
    experts_table = np.array(experts_table)
    experts_table = experts_table.reshape(experts_table.shape[0], -1)
    return experts_table


def convert(data):
    if isinstance(data, (np.integer, np.floating)):
        return data.item()
    if isinstance(data, np.ndarray):
        return data.tolist()
    raise ValueError(f"Object of type {type(data)} is not Json serializable.")
