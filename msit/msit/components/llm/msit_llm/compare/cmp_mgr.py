# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
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
import re
import itertools
import torch

from msit_llm.common.log import logger
from msit_llm.compare.cmp_op_match import OpMatchMgr
from msit_llm.compare.cmp_data_parse import CompareDataParse, CompareDataTorch, CompareDataATB
from msit_llm.compare.cmp_utils import BasicDataInfo, fill_row_data, save_compare_reault_to_csv, read_data, \
                                    fill_row_data_statistics, save_statistics_compare_reault_to_csv, \
                                    save_compare_reault_to_xlsx
from msit_llm.compare.multi_block import get_multi_tensor_paths, get_cat_dim, get_multi_statistics_paths
from msit_llm.common.constant import RAW_INPUT_PATH, GLOBAL_HISTORY_AIT_DUMP_PATH_LIST



class CompareMgr:
    data_parsers = [CompareDataATB, CompareDataTorch]

    def __init__(self, golden_path: str, my_path: str, args) -> None:
        self.args = args
        self.golden_data: CompareDataParse = self.init_compare_data(golden_path, args)
        self.my_data: CompareDataParse = self.init_compare_data(my_path, args)
        self.golden_raw_path = self.get_raw_path(golden_path)
        self.my_raw_path = self.get_raw_path(my_path)
        os.environ[RAW_INPUT_PATH] = self.golden_raw_path + "|" + self.my_raw_path
        self.op_match: OpMatchMgr = OpMatchMgr(args)
        self.compared_result = []  # 收集结果
        self.compared_results = []  # 全部比对时收集结果

    @staticmethod
    def filter_tensor_paths(golden_tensor_paths, my_tensor_paths, path_prefix='qkv'):
        # 根据 path_prefix 构建对应的目标路径
        if path_prefix == 'qkv':
            q_path = '0_Attention/0_QKVLinearSplitPack/after/outtensor0.bin'
            k_path = '0_Attention/0_QKVLinearSplitPack/after/outtensor1.bin'
            v_path = '0_Attention/0_QKVLinearSplitPack/after/outtensor2.bin'
            rotary_path = '0_Attention/1_RotaryPositionEmbedding/before/intensor0.bin'
        elif path_prefix == 'qkv_split':
            q_path = '0_Attention/0_QKVLinearSplitPack/1_SplitOperation/after/outtensor0.bin'
            k_path = '0_Attention/0_QKVLinearSplitPack/1_SplitOperation/after/outtensor1.bin'
            v_path = '0_Attention/0_QKVLinearSplitPack/1_SplitOperation/after/outtensor2.bin'
            rotary_path = None  # 如果 split 模式下不需要 rotary 路径
        else:
            raise ValueError(f"Unsupported path_prefix: {path_prefix}")

        tensor_filter_patterns = [
            (r'root\.model\.layers\.\d+\.self_attn\.k_proj/output\.pth', k_path),
            (r'root\.model\.layers\.\d+\.self_attn\.v_proj/output\.pth', v_path),
            (r'root\.model\.layers\.\d+\.self_attn\.q_proj/output\.pth', q_path),
        ]
        # 如果不是 split 模式，添加 rotary 的规则
        if rotary_path:
            tensor_filter_patterns.append(
                (r'root\.model\.layers\.\d+\.self_attn\.q_proj/output\.pth', rotary_path)
            )
        matched_my_paths = set()
        for golden_pattern, target_tensor in tensor_filter_patterns:
            if any(re.search(golden_pattern, path) for path in golden_tensor_paths):
                matched = [path for path in my_tensor_paths if target_tensor in path]
                matched_my_paths.update(matched)
                logger.debug(f"Matched pattern: {golden_pattern} => Target: '{target_tensor}")

        return list(matched_my_paths) if matched_my_paths else my_tensor_paths
    
    @staticmethod
    def filter_golden_tensor_paths(golden_tensor_paths):
        exclude_pattern = re.compile(r'output_\d+_\d+\.pth$')
        filtered_paths = [
            path 
            for path in golden_tensor_paths
            if not exclude_pattern.search(path)
        ]

        return filtered_paths

    @classmethod
    def get_raw_path(cls, path: str):
        raw_path = os.path.basename(path)
        if any([raw_path.startswith(x) for x in GLOBAL_HISTORY_AIT_DUMP_PATH_LIST]):
            raw_path = path
        else:
            raw_path = os.path.dirname(os.path.dirname(path))
        return raw_path

    @classmethod
    def _flatten_and_enum_tuple(cls, *arr):
        for item in arr:
            if isinstance(item, (list, tuple)):
                yield from cls._flatten_and_enum_tuple(*item)
            else:
                yield item

    @classmethod
    def _filter_rope_my_tensor_paths(cls, my_tensor_paths):
        seqlen_path = None
        valid_my_tensor_paths = []

        if len(my_tensor_paths) != 5:
            logger_text = f"Expected 5 tensors for RopeOperation but found {len(my_tensor_paths)}."
            logger.debug(logger_text)
            return seqlen_path, my_tensor_paths

        for path in my_tensor_paths:
            if "intensor4" in path.lower():
                seqlen_path = path
            elif "intensor2" in path.lower() or "intensor3" in path.lower():
                valid_my_tensor_paths.append(path)

        if len(valid_my_tensor_paths) != 2:
            logger_text = (f"Expected intensor2.bin and intensor3.bin for RopeOperaton "
                           f"but only found {valid_my_tensor_paths}.")
            logger.debug(logger_text)
            return seqlen_path, my_tensor_paths

        valid_my_tensor_paths.sort(key=lambda path: ("intensor3" in path.lower()))
        return seqlen_path, valid_my_tensor_paths

    @classmethod
    def _get_rope_type(cls, my_tensor_path):
        rope_type = -1

        # rope_type=0 means cos, rope_type=1 means sin
        if "intensor2" in my_tensor_path.lower():
            rope_type = 0
        elif "intensor3" in my_tensor_path.lower():
            rope_type = 1
        else:
            logger_text = f"Failed to get rope_type from {my_tensor_path}."
            logger.debug(logger_text)

        return rope_type

    @classmethod
    def _slice_tensor_by_seq_len(cls, golden_tensor_datas, seq_len, rope_type):
        if seq_len is None:
            logger_text = "seq_len is None, skipping slicing."
            logger.debug(logger_text)
            return golden_tensor_datas

        tensor = golden_tensor_datas[0]
        if tensor.ndimension() == 4:
            if 1 <= seq_len < tensor.shape[2]:
                return [tensor[:, :, seq_len - 1, :].to(dtype=tensor.dtype).squeeze(0)]
            elif 1 <= seq_len < tensor.shape[1]:
                return [tensor[:, seq_len - 1, :, :].to(dtype=tensor.dtype).squeeze(0)]
            logger_text = f"seqLen is out of bounds for tensor with shape {tensor.shape}"
            logger.error(logger_text)
        elif tensor.ndimension() == 3:
            if 1 <= seq_len < tensor.shape[0]:
                return [tensor[seq_len - 1, :, rope_type].to(dtype=tensor.dtype).unsqueeze(0)]
            logger_text = f"seqLen is out of bounds for tensor with shape {tensor.shape}"
            logger.error(logger_text)
        else:
            logger_text = f"Unsupported tensor with dimensions {tensor.ndimension()}. Expected 3 or 4 dimensions."
            logger.debug(logger_text)
        return golden_tensor_datas

    @classmethod
    def _remove_adjacent_repeated_columns(cls, my_tensor_datas):
        tensor = my_tensor_datas[0]
        if tensor.ndimension() != 2 or tensor.shape[0] != 1:
            logger_text = (f"Unexpected tensor with dimensions {tensor.ndimension()}. "
                           f"Expected 2 dimensions with one row.")
            logger.debug(logger_text)
            return my_tensor_datas

        if tensor.shape[1] < 2 or tensor.shape[1] % 2 != 0:
            logger_text = f"Unexpected tensor shape {tensor.shape} to check whether has adjacent repeated columns."
            logger.debug(logger_text)
            return my_tensor_datas

        if (tensor[:, ::2] == tensor[:, 1::2]).all():
            return [tensor[:, ::2].to(dtype=tensor.dtype)]
        logger_text = f"There are no adjacent repeated columns in the tensor {my_tensor_datas}."
        logger.debug(logger_text)
        return my_tensor_datas

    def init_compare_data(self, data_path: str, args) -> CompareDataParse:
        for cls_data in self.data_parsers:
            if cls_data.accept(data_path, args):
                return cls_data(data_path, self.args)
        logger.error(f"cannot parse data path({data_path}). it is not a atb dump path or a torch dump path.")
        return None

    def is_parsed_cmp_path(self):
        return self.golden_data is not None and self.my_data is not None

    def compare(self, output_path="."):
        self.compared_results = []

        op_maps = list(self.op_match.match(self.golden_data, self.my_data))
        # 同步校验tokenid
        golden_tokens = self.golden_data.get_cmp_tokens()
        my_tokens = self.my_data.get_cmp_tokens()
        golden_token_set = set(self._flatten_and_enum_tuple(golden_tokens))
        my_token_set = set(self._flatten_and_enum_tuple(my_tokens))

        logger.debug(
            "compare tokens info ==> golden_tokens: %s, golden_token_set: %s \n my_tokens: %s, my_token_set: %s ",
            str(golden_tokens),
            str(golden_token_set),
            str(my_tokens),
            str(my_token_set),
        )
        if len(op_maps) == 1:
            op_map = op_maps[0]
            self.compared_result = []
            if self.args.stats:
                self.perform_comparison(golden_tokens, my_tokens, op_map, is_statistics=True)
                return save_statistics_compare_reault_to_csv(self.compared_result, output_path)
            else:
                self.perform_comparison(golden_tokens, my_tokens, op_map, is_statistics=False)
                return save_compare_reault_to_csv(self.compared_result, output_path)
        elif len(op_maps) == 3:
            sheet_names = ['layer', 'module', 'api']
            for op_map in op_maps:
                self.compared_result = []
                if self.args.stats:
                    self.perform_comparison(golden_tokens, my_tokens, op_map, is_statistics=True)
                else:
                    self.perform_comparison(golden_tokens, my_tokens, op_map, is_statistics=False)
                self.compared_results.append(self.compared_result)
            return save_compare_reault_to_xlsx(self.compared_results, sheet_names, output_path)
        else:
            return None

    def compare_tokens_stats(self, cmp_tokens, op_map, is_statistics=False):
        for token_id in cmp_tokens:
            logger.debug("comparing tokens is %s ", str(token_id))
            if is_statistics:
                self.compare_statistics(token_id, token_id, op_map)
            else:
                self.compare_token(token_id, token_id, op_map)
    
    def perform_comparison(self, golden_tokens, my_tokens, op_map, is_statistics=False):
        golden_token_set = set(self._flatten_and_enum_tuple(golden_tokens))
        my_token_set = set(self._flatten_and_enum_tuple(my_tokens))

        if not golden_token_set.isdisjoint(my_token_set):
            cmp_tokens = golden_token_set.intersection(my_token_set)
            self.compare_tokens_stats(cmp_tokens, op_map, is_statistics)
        elif self.golden_data.get_token_id() is not None and self.my_data.get_token_id() is not None:
            # 强制比对指定token
            for golden_token, my_token in itertools.product(golden_tokens, my_tokens):
                if is_statistics:
                    self.compare_statistics(golden_token, my_token, op_map)
                else:
                    self.compare_token(golden_token, my_token, op_map)
        else:
            logger.error(
                f"my tokens is {my_tokens} and golden tokens is {golden_tokens}. The two cannot be matched."
            )

    def compare_statistics(self, golden_token_id, my_token_id, op_map):
        for my_op, my_op_location, golden_op, _ in op_map:
            op_type = [my_op.op_type, golden_op.op_type]
            logger.debug("------ compare (%s %s)------", str(my_op.node_name), str(golden_op.node_name))
            golden_tensor_paths = list(self.golden_data.get_csv_path(golden_token_id, golden_op))
            my_tensor_paths = list(self.my_data.get_tensor_path(my_token_id, my_op, my_op_location))
            my_tensor_paths = self.filter_tensor_paths(golden_tensor_paths, my_tensor_paths, 'qkv')
            my_tensor_paths = self.filter_tensor_paths(golden_tensor_paths, my_tensor_paths, 'qkv_split')
            golden_tensor_paths = self.filter_golden_tensor_paths(golden_tensor_paths)
            if len(golden_tensor_paths) == len(my_tensor_paths):
                golden_tensor_paths.sort()
                my_tensor_paths.sort()
                tensor_pairs = zip(golden_tensor_paths, my_tensor_paths)
            else:
                tensor_pairs = itertools.product(golden_tensor_paths, my_tensor_paths)
            for golden_tensor_path, my_tensor_path in tensor_pairs:
                logger.debug("golden_path: %s; my_path:%s", str(golden_tensor_path), str(my_tensor_path))

                _, my_tensor_datas = get_multi_statistics_paths(
                    self.my_data.get_token_path(my_token_id), my_tensor_path, tensor_sub_dir=""
                )
                _, golden_tensor_datas = get_multi_statistics_paths(
                    self.golden_data.get_token_path(golden_token_id), golden_tensor_path, tensor_sub_dir=""
                )
                data_info = BasicDataInfo(golden_tensor_path, my_tensor_path, token_id=my_token_id, op_type=op_type)
                row_data = fill_row_data_statistics(data_info, my_tensor_datas, golden_tensor_datas)
                self.compared_result.append(row_data)

    def compare_token(self, golden_token_id, my_token_id, op_map):
        for my_op, my_op_location, golden_op, golden_op_location in op_map:
            op_type = [my_op.op_type, golden_op.op_type]
            logger.debug("------ compare (%s %s)------", str(my_op.node_name), str(golden_op.node_name))
            # 获取到所有需要比较的 tensor 的路径
            golden_tensor_paths = list(self.golden_data.get_tensor_path(golden_token_id, golden_op, golden_op_location))
            my_tensor_paths = list(self.my_data.get_tensor_path(my_token_id, my_op, my_op_location))
            my_tensor_paths = self.filter_tensor_paths(golden_tensor_paths, my_tensor_paths, 'qkv')
            my_tensor_paths = self.filter_tensor_paths(golden_tensor_paths, my_tensor_paths, 'qkv_split')
            golden_tensor_paths = self.filter_golden_tensor_paths(golden_tensor_paths)
            seq_len, my_tensor_paths = self._handle_rope_operation_paths(my_tensor_paths)

            if len(golden_tensor_paths) == len(my_tensor_paths):
                golden_tensor_paths.sort()
                my_tensor_paths.sort()
                tensor_pairs = zip(golden_tensor_paths, my_tensor_paths)
            else:
                tensor_pairs = itertools.product(golden_tensor_paths, my_tensor_paths)

            # 交叉比对，记录结果
            for golden_tensor_path, my_tensor_path in tensor_pairs:
                logger.debug("golden_path: %s; my_path:%s", str(golden_tensor_path), str(my_tensor_path))

                # 1. get tensor_datas 多卡数据合并
                _, my_tensor_datas = get_multi_tensor_paths(
                    self.my_data.get_token_path(my_token_id), my_tensor_path, tensor_sub_dir=""
                )
                _, golden_tensor_datas = get_multi_tensor_paths(
                    self.golden_data.get_token_path(golden_token_id), golden_tensor_path, tensor_sub_dir=""
                )
                # 2. concat tensor_datas
                if "rotary" in golden_tensor_path:
                    rope_type = self._get_rope_type(my_tensor_path)
                    if rope_type != -1:
                        golden_tensor_datas = self._slice_tensor_by_seq_len(golden_tensor_datas, seq_len, rope_type)
                    my_tensor_datas = self._remove_adjacent_repeated_columns(my_tensor_datas)

                dim_atb = get_cat_dim(my_tensor_datas, golden_tensor_datas)
                dim_torch = get_cat_dim(golden_tensor_datas, my_tensor_datas)
                atb_tensor_data = my_tensor_datas[0] if dim_atb == -1 else torch.cat(my_tensor_datas, dim_atb)
                torch_tensor_data = (
                    golden_tensor_datas[0] if dim_torch == -1 else torch.cat(golden_tensor_datas, dim_torch)
                )
                # 3. compare tensor_datas
                data_info = BasicDataInfo(golden_tensor_path, my_tensor_path, token_id=my_token_id, op_type=op_type)
                row_data = fill_row_data(data_info, atb_tensor_data, torch_tensor_data)
                self.compared_result.append(row_data)

    def _handle_rope_operation_paths(self, my_tensor_paths):
        seq_len = None
        if any("ropeoperation" in path.lower() for path in my_tensor_paths):
            seqlen_path, valid_my_tensor_paths = self._filter_rope_my_tensor_paths(my_tensor_paths)
            if seqlen_path is not None:
                seq_len = read_data(seqlen_path)
                my_tensor_paths = valid_my_tensor_paths
                if seq_len.shape == torch.Size([1]):
                    return seq_len.item(), my_tensor_paths
                logger_text = f"Expected shape of seqLen is torch.Size([1]) but found {seq_len.shape}."
                logger.debug(logger_text)
            else:
                logger_text = "Failed to get input tensors of RopeOperation to compare."
                logger.debug(logger_text)
        else:
            logger_text = f"RopeOperation not found in {my_tensor_paths}."
            logger.debug(logger_text)
        return seq_len, my_tensor_paths

