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
import glob
import re
import shutil
from pathlib import Path

import json
import pandas as pd
import torch

from c2lb import lb_and_intra_layer_affinity_redundancy_deploy
from c2lb_dynamic import lb_redundancy_deploy_for_dynamic
from c2lb_a3 import lb_and_intra_layer_affinity_redundancy_deploy_a3
from components.utils.log import logger
from components.utils.file_open_check import ms_open
from components.expert_load_balancing.elb.preprocessing import process_speculative_moe
from components.expert_load_balancing.elb.constant import PREFILL, DECODE, DECODE_FILE_NAME, \
                        PREFILL_FILE_NAME, ALGORITHM_C2LB, ALGORITHM_DYNAMIC_C2LB, A2, A3, \
                        SPECULATIVE_MOE_ALGORITHM


def load_expert_popularity_csv(csv_file_path):
    """
    从指定目录加载 decode_info.csv 和 prefill_info.csv 文件，支持以下场景：
    - 仅存在其中一个文件。
    - 同时存在两个文件。
    
    参数:
    - csv_file_path (str): 包含 CSV 文件的目录路径。
    
    返回:
    - (decode_df, prefill_df): 返回元组。
        - decode_df: decode_info.csv 的内容，若文件不存在或读取失败则为 None。
        - prefill_df: prefill_info.csv 的内容，若文件不存在或读取失败则为 None。
    """
    decode_info = None
    prefill_info = None

    for file_name in ["decode_info.csv", "prefill_info.csv"]:
        file_path = os.path.join(csv_file_path, file_name)
        
        if not os.path.isfile(file_path):
            logger.warning(f"File not found: {file_path}")
            continue
        
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Successfully loaded {file_name}.")
            
            if file_name == "decode_info.csv":
                decode_info = df
            elif file_name == "prefill_info.csv":
                prefill_info = df
        except pd.errors.EmptyDataError as e:
            raise ValueError(f"decode_info or prefill_info is empty: {file_path}") from e
        except pd.errors.ParserError as e:
            raise ValueError(f"Failed to parse CSV: {file_path}") from e
        except Exception as e:
            raise RuntimeError(f"Error reading {file_path}: {str(e)}") from e

    is_decode_missing = decode_info is None or decode_info.empty
    is_prefill_missing = prefill_info is None or prefill_info.empty
    if is_decode_missing and is_prefill_missing:
        raise FileNotFoundError(f"No decode_info.csv or prefill_info.csv found in {csv_file_path}."
                                f"Check that the input file is correct to generate the files.")
    
    return decode_info, prefill_info



def merge_csv_columns(csv_path, pattern_prefix):
    """
    合并指定目录下以特定前缀开头的所有csv文件的列。

    参数:
    - csv_path: 文件所在的目录路径。
    - pattern_prefix: 文件名前缀。
    
    返回:
    - 生成的DataFrame对象。
    """
    file_pattern = os.path.join(csv_path, f"{pattern_prefix}_*.csv")
    
    def sort_filenames(filenames):
        return sorted(filenames, key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    
    files = glob.glob(file_pattern)
    filtered_files = [f for f in files if re.match(rf".*{pattern_prefix}_(\d+)\.csv", f)]
    if not filtered_files:
        raise FileNotFoundError(f"No files found matching the pattern {file_pattern}.")

    filtered_files = sort_filenames(filtered_files)

    logger.debug(f"{pattern_prefix} file number is {len(filtered_files)}")
    all_columns = []
    column_count = 0
    
    for file in filtered_files:
        logger.debug(f"Processing file: {file}.")
        df = pd.read_csv(file, header=None)
        if df.empty:
            raise ValueError(f"File {file} is empty. Check the input decode or prefill file.")
        
        for col in range(df.shape[1]):
            all_columns.append(df.iloc[:, col].rename(f'expert_{column_count}'))
            column_count += 1
    
    if not all_columns:
        raise ValueError("No columns to concatenate. Check whether the ipnut csv path is correct."
                         "Or whether the csv file in the folder is valid.")
    
    final_df = pd.concat(all_columns, axis=1)
    return final_df


def save_matrix_to_json(output_path, file_name, deployment):
    def get_ndim(obj):
        if isinstance(obj, list):
            return 1 + (get_ndim(obj[0]) if obj else 0)
        return 0
    
    cur_dim = get_ndim(deployment)
    if cur_dim != 3:
        raise ValueError(f"部署矩阵必须是三维数组，但当前维度为 {cur_dim}D。")
    num_layers = len(deployment)
    num_cards = len(deployment[0])

    data = {"moe_layer_count": num_layers}
    layer_list = []
    for i in range(num_layers):
        layer = {"layer_id": i, "device_count": num_cards}
        device_list = []
        for j in range(num_cards):
            device = {"device_id": j, "device_expert": list(deployment[i][j])}
            device_list.append(device)
        layer["device_list"] = device_list
        layer_list.append(layer)
    data["layer_list"] = layer_list

    file_name = os.path.join(output_path, f"{file_name}.json")

    # 保存为 JSON 文件
    try:
        with ms_open(file_name, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        raise RuntimeError(f"保存json文件 {deployment} 时出错") from e


def dump_tables(deploy_fp, d2e_tables_list, n_devices):
    layer_list = []

    for layer_idx, d2e_tables in enumerate(d2e_tables_list):
        device_list = [
            {"device_id": d, "device_expert": d2e_tables[layer_idx][d].tolist()} 
            for d in range(n_devices)
        ]
        layer_list.append({
            "layer_id": layer_idx,
            "device_count": n_devices,
            "device_list": device_list
        })
    json_data = {
        "moe_layer_count": len(layer_list),
        "layer_list": layer_list
    }
    logger.debug(json_data)
    with ms_open(deploy_fp, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)


def save_matrix_to_json_a3(output_path, file_name, deployment):
    if deployment.ndim < 3:
        raise ValueError(f"输出文件的实际维度有误: {deployment.ndim}, 需要3维数据")
    num_layers = len(deployment)
    num_cards = len(deployment[0])

    data = {"moe_layer_count": num_layers}
    layer_list = []
    for i in range(num_layers):
        layer = {"layer_id": i, "device_count": num_cards}
        device_list = []
        for j in range(num_cards):
            # 将 1*4 的行矩阵转换为列表
            device = {"device_id": j, "device_expert": deployment[i][j].tolist()}
            device_list.append(device)
        layer["device_list"] = device_list
        layer_list.append(layer)
    data["layer_list"] = layer_list

    file_name = os.path.join(output_path, f"{file_name}.json")
    try:
        with ms_open(file_name, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        raise RuntimeError(f"写入文件 {deployment} 时出错: {e}") from e


def dump_tables_a3(deploy_fp, d2e_tables_obj_list, n_devices, n_share_expert_devices):
    mean_obj = 0
    for layer_idx, d2e_tables_obj in enumerate(d2e_tables_obj_list):
        mean_obj += d2e_tables_obj[1][layer_idx]
    layer_list = []
    for layer_idx, d2e_tables_obj in enumerate(d2e_tables_obj_list):
        d2e_tables = d2e_tables_obj[0]
        device_list = [
            {"device_id": d, "device_expert": [0]} 
            for d in range(n_share_expert_devices)
        ]
        device_list.extend([
            {"device_id": d + n_share_expert_devices, "device_expert": d2e_tables[layer_idx][d].tolist()}
            for d in range(n_devices)
        ])
        layer_list.append({
            "layer_id": layer_idx,
            "device_count": n_devices + n_share_expert_devices,
            "device_list": device_list
        })
    json_data = {
        "moe_layer_count": len(layer_list),
        "layer_list": layer_list
    }
    with ms_open(deploy_fp, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)


def save_dataframes(prefill_final_df, decode_final_df, output_dir):
    """Save prefill and decode dataframes to CSV files if they are not empty."""
    if prefill_final_df is not None and not prefill_final_df.empty:
        prefill_final_df.to_csv(os.path.join(output_dir, 'prefill_info.csv'), index=False)
    if decode_final_df is not None and not decode_final_df.empty:
        decode_final_df.to_csv(os.path.join(output_dir, 'decode_info.csv'), index=False)



def process_c2lb(args, output_dir):
    """Process C2LB algorithm and save results based on available data files.
    
    支持场景:
    - 只有 decode_info.csv 时: 处理 decode 数据。
    - 只有 prefill_info.csv 时: 处理 prefill 数据。
    - 两者都存在时: 同时处理 decode 和 prefill 数据。
    
    参数:
    - args: 包含算法参数的命名空间。
    - output_dir: 输出结果的目录路径。
    """
    logger.info("Generating Files with the C2lb Algorithm for Static Scenarios")
    decode_df, prefill_df = load_expert_popularity_csv(output_dir)

    if decode_df is not None:
        try:
            num_original_expert = decode_df.shape[1] 
            decode_np = decode_df.to_numpy()
            global_deployment = lb_and_intra_layer_affinity_redundancy_deploy(
                decode_np, 
                args.num_redundancy_expert, 
                args.num_npus, 
                num_original_expert, 
            )
            save_matrix_to_json(output_dir, DECODE_FILE_NAME, global_deployment)
            logger.info(f"C2LB processed decode data -> {DECODE_FILE_NAME}.json")
        except Exception as e:
            raise RuntimeError(f"Failed to process decode data: {str(e)}") from e

    if prefill_df is not None:
        try:
            num_original_expert = prefill_df.shape[1] 
            prefill_np = prefill_df.to_numpy()
            global_deployment = lb_and_intra_layer_affinity_redundancy_deploy(
                prefill_np,  
                args.num_redundancy_expert, 
                args.num_npus, 
                num_original_expert,
            )
            save_matrix_to_json(output_dir, PREFILL_FILE_NAME, global_deployment)
            logger.info(f"C2LB processed prefill data -> {PREFILL_FILE_NAME}.json")
        except Exception as e:
            raise RuntimeError(f"Failed to process prefill data: {str(e)}") from e
    
    if decode_df is None and prefill_df is None:
        raise FileNotFoundError("No valid decode/prefill data found in", output_dir)
    

def process_dynamic_c2lb(args, output_dir):
    """Process dynamic C2LB algorithm and save results based on available data files.
    
    支持场景:
    - 只有 decode_info.csv 时: 处理 decode 数据。
    - 只有 prefill_info.csv 时: 处理 prefill 数据。
    - 两者都存在时: 同时处理 decode 和 prefill 数据。
    
    参数:
    - args: 包含算法参数的命名空间。
    - output_dir: 输出结果的目录路径。
    """
    logger.info("Generating Initialization Files with the C2lb Algorithm for Dynamic Scenarios")
    decode_df, prefill_df = load_expert_popularity_csv(output_dir)

    if decode_df is not None:
        try:
            decode_np = decode_df.to_numpy()
            global_deployment = lb_redundancy_deploy_for_dynamic(
                decode_np, 
                args.num_redundancy_expert, 
                args.num_nodes, 
                args.num_npus
            )
            save_matrix_to_json(output_dir, DECODE_FILE_NAME, global_deployment)
            logger.info(f"dynamic C2LB processed decode data -> {DECODE_FILE_NAME}.json")
        except Exception as e:
            raise RuntimeError(f"Failed to process decode data: {str(e)}") from e

    if prefill_df is not None:
        try:
            prefill_np = prefill_df.to_numpy()
            global_deployment = lb_redundancy_deploy_for_dynamic(
                prefill_np,  
                args.num_redundancy_expert, 
                args.num_nodes, 
                args.num_npus
            )
            save_matrix_to_json(output_dir, PREFILL_FILE_NAME, global_deployment)
            logger.info(f"dynamic C2LB processed prefill data -> {PREFILL_FILE_NAME}.json")
        except Exception as e:
            raise RuntimeError(f"Failed to process decode data: {str(e)}") from e

    if decode_df is None and prefill_df is None:
        raise FileNotFoundError(f"No valid decode/prefill data found in {output_dir}")


def process_c2lb_a3(args, output_dir):
    """Process C2LB A3 algorithm and save results based on available data files.
    
    支持场景:
    - 只有 decode_info.csv 时: 处理 decode 数据。
    - 只有 prefill_info.csv 时: 处理 prefill 数据。
    - 两者都存在时: 同时处理 decode 和 prefill 数据。
    
    参数:
    - args: 包含算法参数的命名空间。
    - output_dir: 输出结果的目录路径。
    """
    logger.info("Generating Files with the C2lb A3 Algorithm for Static Scenarios")
    decode_df, prefill_df = load_expert_popularity_csv(output_dir)

    if decode_df is not None:
        try:
            num_original_expert = decode_df.shape[1] 
            decode_np = decode_df.to_numpy()
            global_deployment = lb_and_intra_layer_affinity_redundancy_deploy_a3(
                decode_np, 
                args.num_redundancy_expert, 
                args.num_npus, 
                num_original_expert, 
            )
            save_matrix_to_json_a3(output_dir, DECODE_FILE_NAME, global_deployment)
            logger.info(f"C2LB A3 processed decode data -> {DECODE_FILE_NAME}.json")
        except Exception as e:
            raise RuntimeError(f"Failed to process decode data: {str(e)}") from e

    if prefill_df is not None:
        try:
            num_original_expert = prefill_df.shape[1] 
            prefill_np = prefill_df.to_numpy()
            global_deployment = lb_and_intra_layer_affinity_redundancy_deploy_a3(
                prefill_np,  
                args.num_redundancy_expert, 
                args.num_npus, 
                num_original_expert,
            )
            save_matrix_to_json_a3(output_dir, PREFILL_FILE_NAME, global_deployment)
            logger.info(f"C2LB A3 processed prefill data -> {PREFILL_FILE_NAME}.json")
        except Exception as e:
            raise RuntimeError(f"Failed to process prefill data: {str(e)}") from e
    
    if decode_df is None and prefill_df is None:
        raise FileNotFoundError("No valid decode/prefill data found in", output_dir)


def check_file_type(folder_path):
    """
    Check the folder for files containing 'prefill' or 'decode' in their names.
    Returns a tuple of:
    - detected file types ('prefill', 'decode', or both).
    - number of 'prefill' files.
    - number of 'decode' files.
    """
    file_types = set()
    prefill_count = 0
    decode_count = 0
    
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(".csv"):
            if PREFILL in file_name.lower():
                file_types.add(PREFILL)
                prefill_count += 1
            if DECODE in file_name.lower():
                file_types.add(DECODE)
                decode_count += 1
    
    return file_types, prefill_count, decode_count


def select_algorithm(args):
    # 静态场景下 A2 c2lb 算法
    if args.algorithm == ALGORITHM_C2LB and args.device_type == A2:
        process_c2lb(args, output_dir=args.output_dir)
    # 动态场景下 A2 c2lb 算法
    elif args.algorithm == ALGORITHM_DYNAMIC_C2LB and args.device_type == A2:
        process_dynamic_c2lb(args, output_dir=args.output_dir)
    # 静态场景下 A3 c2lb 算法
    elif args.algorithm == ALGORITHM_C2LB and args.device_type == A3:
        if args.share_expert_devices == 0:
            process_c2lb_a3(args, output_dir=args.output_dir)
        else:
            raise ValueError("Input incorrect share expert devices parameters.")
    # 静态场景下 speculative_moe 算法
    elif args.algorithm in SPECULATIVE_MOE_ALGORITHM:
        process_speculative_moe(args)
    else:
        raise ValueError("Please enter valid parameters")


def is_file_readable(file_path):
    """尝试读取CSV文件以判断是否损坏"""
    try:
        pd.read_csv(file_path)
        return True
    except Exception as e:
        logger.warning(f"File {file_path} unable to read: {e}")
        return False


def copy_file(main_file, bak_file, target_file):
    if is_file_readable(main_file):
        logger.debug(f"copy: {main_file} -> {target_file}")
        shutil.copy(main_file, target_file)
    elif os.path.exists(bak_file) and is_file_readable(bak_file):
        logger.debug(f"The original file is damaged, use the backup file: {bak_file} -> {target_file}")
        shutil.copy(bak_file, target_file)
    else:
        raise RuntimeError("The backup files are corrupted or do not exist!")


def copy_files_with_recovery(src_dir, dest_dir):
    # 创建目标目录
    os.makedirs(dest_dir, exist_ok=True)

    # 获取所有 decode_*.csv 和 prefill_*.csv 文件
    decode_files = [f for f in os.listdir(src_dir) if re.match(r"decode_\d+\.csv", f)]
    prefill_files = [f for f in os.listdir(src_dir) if re.match(r"prefill_\d+\.csv", f)]

    # 排序
    decode_files.sort(key=extract_number)
    prefill_files.sort(key=extract_number)

    def process_files(file_list, file_type):
        for file in file_list:
            n = extract_number(file)
            main_file = os.path.join(src_dir, file)
            bak_file = os.path.join(src_dir, f"{file_type}_{n}_bak.csv")
            target_file = os.path.join(dest_dir, f"{file_type}_{n}.csv")

            topk_file = os.path.join(src_dir, f"{file_type}_topk_{n}.csv")
            topk_bak_file = os.path.join(src_dir, f"{file_type}_topk_{n}_bak.csv")
            topk_target_file = os.path.join(dest_dir, f"{file_type}_topk_{n}.csv")

            if os.path.exists(main_file) and os.path.exists(bak_file):
                copy_file(main_file, bak_file, target_file)
            if os.path.exists(topk_file) and os.path.exists(topk_bak_file):
                copy_file(topk_file, topk_bak_file, topk_target_file)

    logger.debug("=== dealing decode_* file ===")
    process_files(decode_files, "decode")

    logger.debug("\n=== dealing prefill_* file ===")
    process_files(prefill_files, "prefill")

    # 复制 JSON 文件
    json_files = [f for f in os.listdir(src_dir) if f.endswith('.json')]
    if len(json_files) == 0:
        raise FileNotFoundError("dict has no JSON file")
    for jf in json_files:
        json_file = os.path.join(src_dir, jf)
        dest_json = os.path.join(dest_dir, jf)
        shutil.copy(json_file, dest_json)


def check_dump_file_version(csv_path):
    # 新版本的dump
    csv_new_path = None
    if os.path.isfile(os.path.join(csv_path, "model_gen_config.json")):
        last_folder_name = Path(csv_path).resolve().name
        new_folder_name = f"{last_folder_name}_selected"
        parent_dir = Path(csv_path).resolve().parent 
        csv_new_path = parent_dir / new_folder_name
        copy_files_with_recovery(csv_path, csv_new_path)
        return csv_new_path
    return csv_new_path


def extract_number(filename):
    """从 decode_n.csv 或 prefill_n.csv 提取 n 的值用于排序"""
    match = re.match(r"(decode|prefill)_(\d+)\.csv", filename)
    if match:
        return int(match.group(2))
    return None


def process_files_by_type(src_dir, file_type, output_path):
    # Step 1: 获取符合条件的 CSV 文件并排序
    files = [f for f in os.listdir(src_dir) if re.match(rf"^{file_type}_\d+\.csv$", f)]
    files.sort(key=extract_number)
    if not files:
        logger.info(f"No CSV files were found starting with {file_type}_")
        return

    # Step 2: 统计每份文件的行数
    file_row_counts = {}
    dfs = {}

    for file in files:
        file_path = os.path.join(src_dir, file)
        try:
            df = pd.read_csv(file_path, header=None)
            file_row_counts[file] = len(df)
            dfs[file] = df
        except Exception as e:
            logger.warning(f"Unable to read file {file}:{e}")
            continue

    if not file_row_counts:
        raise ValueError(f"Did not successfully read any valid csv files")
    min_row_count = min(file_row_counts.values())

    # Step 3: 读取 JSON 获取 num_moe_layers
    json_path = os.path.join(src_dir, "model_gen_config.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"File model_gen_config.json not found:{json_path}")

    with ms_open(json_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    num_moe_layers = config.get("num_moe_layers")
    if num_moe_layers is None:
        raise KeyError("The num_moe_layers field is missing in the JSON file")

    logger.info(f"[{file_type}] num_moe_layers = {num_moe_layers} Minimum number of rows = {min_row_count}")

    # Step 4: 对每个文件提取对应的行范围
    usable_row_count = (min_row_count // num_moe_layers) * num_moe_layers
    start_idx = usable_row_count - num_moe_layers
    end_idx = min_row_count

    expert_data = []

    for idx, file in enumerate(files):
        df = dfs[file]
        selected_data = df.iloc[start_idx:end_idx].reset_index(drop=True)
        num_cols = selected_data.shape[1]
        selected_data.columns = [f"expert_{idx * num_cols + i}" for i in range(num_cols)]
        expert_data.append(selected_data)

    # Step 5: 合并所有文件的选中行（横向拼接）
    merged_df = pd.concat(expert_data, axis=1)

    output_file = f"{file_type}_info.csv"
    output_path = os.path.join(output_path, output_file)
    merged_df.to_csv(output_path, index=False, header=True)

    logger.info(f"Merge completed, results saved to:{output_path}")


def load_balancing(args):
    """
    Main function to perform load balancing based on the specified algorithm.
    """
    csv_new_path = check_dump_file_version(args.expert_popularity_csv_load_path)
    if csv_new_path:
        # 新版本dump数据，需要专门适配c2lb算法
        if args.algorithm == ALGORITHM_DYNAMIC_C2LB or args.algorithm == ALGORITHM_C2LB:
            process_files_by_type(csv_new_path, "decode", args.output_dir)
            process_files_by_type(csv_new_path, "prefill", args.output_dir)

        select_algorithm(args)
    else:
        #  旧版本dump数据，无需处理
        # Check the file types in the input path
        file_types, prefill_count, decode_count = check_file_type(args.expert_popularity_csv_load_path)

        # Print the file types and their count
        logger.info(f"Detected file types: {', '.join(file_types)}")
        logger.info(f"Total 'prefill' files: {prefill_count}")
        logger.info(f"Total 'decode' files: {decode_count}")

        # Merge CSV columns for prefill and decode
        prefill_final_df = None   
        decode_final_df = None
        if PREFILL in file_types:
            prefill_final_df = merge_csv_columns(args.expert_popularity_csv_load_path, PREFILL)
        if DECODE in file_types:
            decode_final_df = merge_csv_columns(args.expert_popularity_csv_load_path, DECODE)
        
        # Save merged dataframes to CSV files if they are not empty
        save_dataframes(prefill_final_df, decode_final_df, args.output_dir)

        # Define file names for further processing
        file_names = []
        if PREFILL in file_types:
            file_names.append("prefill_info.csv")
        if DECODE in file_types:
            file_names.append("decode_info.csv")

        select_algorithm(args)