# -*- coding: utf-8 -*-
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

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from msserviceprofiler.modelevalstate.config.config import BenchMarkPolicy, DeployPolicy, field_to_param
from msserviceprofiler.modelevalstate.config.config import default_support_field, PerformanceIndex, OptimizerConfigField
from msserviceprofiler.modelevalstate.optimizer.optimizer import PSOOptimizer, main


@pytest.fixture
def generate_store(tmpdir):
    _config = Path(tmpdir).joinpath("data_storage_20250815175148.csv")
    datas = [
        """generate_speed,time_to_first_token,time_per_output_token,success_rate,throughput,ttft_max,ttft_min, \
        ttft_p75,ttft_p90,ttft_p99,tpot_max,tpot_min,tpot_p75,tpot_p90,tpot_p99,prefill_batch_size, \
        prefill_batch_size_min,prefill_batch_size_max,prefill_batch_size_p75,prefill_batch_size_p90, \
        prefill_batch_size_p99,decoder_batch_size,decoder_batch_size_min,decoder_batch_size_max, \
        decoder_batch_size_p75,decoder_batch_size_p90,decoder_batch_size_p99,max_batch_size, \
        max_prefill_batch_size,prefill_time_ms_per_req,max_queue_deloy_mircroseconds,support_select_batch, \
        prefill_policy_type,decode_policy_type,CONCURRENCY,REQUESTRATE, \
        error,bakcup,fitness,llm_model,dataset_path,simulator""",
        """1651.9238,25.858309600000002,0.459911,1.0,5.6491,122.56516400000001,0.6183994,23.629147099999997, \
        97.3381163,110.01047340000001,96.42427740000001,0.0196638,0.3387629,0.493587,0.9246494000000001, \
        19.2875,2.0,20.0,20.0,20.0,20.0,116.1429,104.0,119.0,118.0,118.0,118.66,118,20,673,579294, \
        False,0,1,1000,0,,,2.3926191461362146e+65,qwen3-8b, \
        /data/w30031530/ModelEvalState/datasets/short_input_64_2k.jsonl,Simulator""",
        """1983.8449,0.3056046,0.0532928,1.0,7.7568,1.0516901,0.0407746,0.4211261,0.46272840000000004, \
        0.5370466,0.7059757,1.26e-05,0.049732399999999996,0.0842894,0.17034649999999998,3.632,2.0,8.0, \
        4.0,5.0,7.0,89.0854,28.0,126.0,110.0,117.9,125.19,493,296,580,350200,False,1,0,1000,8,,, \
        3.858581664468014,qwen3-8b,/data/w30031530/ModelEvalState/datasets/short_input_64_2k.jsonl,Simulator""",
        """1719.4973,0.30205509999999997,0.045293799999999995,1.0,6.7237,0.6956194,0.0435473,0.4166629, \
        0.45262670000000005,0.5302,0.454728,1.2800000000000001e-05,0.042326,0.051689,0.1506695, \
        3.1847,2.0,7.0,4.0,5.0,6.0,69.519,28.0,96.0,82.0,90.0,95.22,493,296,580,350200,False,1,0,1000, \
        7,,,4.22999054887793,qwen3-8b,/data/w30031530/ModelEvalState/datasets/short_input_64_2k.jsonl,Simulator""",
        """2628.1846,0.3350128,0.14603380000000002,1.0,10.276,0.7852948,0.0529368,0.4539578,0.5215298, \
        0.6312122,0.445232,1.4999999999999999e-05,0.1741628,0.23369720000000002,0.3331318,5.2632,1.0, \
        10.0,6.0,7.0,9.0,336.7906,71.0,487.0,418.0,459.4,484.24,493,296,580,350200,False,1,0,1000,12,,, \
        98.08625743412847,qwen3-8b,/data/w30031530/ModelEvalState/datasets/short_input_64_2k.jsonl,Simulator""",
        """1973.7498,0.2979828,0.0526314,1.0,7.7173,0.6064501,0.038523499999999995,0.41426260000000004, \
        0.4516434,0.5381849,0.6366436,1.2800000000000001e-05,0.0486303,0.0840513,0.17018960000000002, \
        3.5461,1.0,8.0,4.0,5.0,7.0,88.7778,27.0,123.0,107.0,115.0,122.2,613,228,430,344700,False,1,3,1000, \
        8,,,3.858610905842756,qwen3-8b,/data/w30031530/ModelEvalState/datasets/short_input_64_2k.jsonl,Simulator""",
        """1733.9595,0.2985424,0.0468999,1.0,6.7796,0.7885291,0.0356113,0.4122982,0.4491465,0.5274601999999999, \
        0.36779480000000003,1.31e-05,0.0435739,0.0623085,0.1526414,3.1881,1.0,7.0,4.0,5.0,6.0,73.7917,25.0,100.0, \
        86.25,92.9,99.29,613,228,430,344700,False,1,3,1000,7,,,4.219550109401511,qwen3-8b, \
        /data/w30031530/ModelEvalState/datasets/short_input_64_2k.jsonl,Simulator""",
        """2593.1693,0.3375048,0.1625388,1.0,10.1381,0.7262981,0.057505099999999996,0.4546796, \
        0.5310524,0.6384890000000001,0.5680927,1.35e-05,0.198526,0.2595833,0.36268700000000004, \
        5.3476,3.0,11.0,6.0,7.0,9.0,397.1239,61.0,591.0,499.5,555.4,587.62,613,228,430,344700, \
        False,1,3,1000,12,,,259.53835449706077,qwen3-8b, \
        /data/w30031530/ModelEvalState/datasets/short_input_64_2k.jsonl,Simulator""",
        """1950.1813,0.49612459999999997,0.0503404,1.0,7.6264,1.1575395000000002,0.0538583, \
        0.7074349999999999,0.7941196,0.903362,0.5111954,1.3e-05,0.0469313,0.0518141,0.196128, \
        6.0729,3.0,11.0,7.0,8.0,10.0,90.7671,30.0,121.0,106.0,113.8,120.28,969,139,650,670600, \
        False,0,3,1000,8,,,3.98983999087121,qwen3-8b, \
        /data/w30031530/ModelEvalState/datasets/short_input_64_2k.jsonl,Simulator""",
        """1745.8056,0.49327150000000003,0.0438012,1.0,6.8259,0.9928296,0.0596509,0.7072557, \
        0.787812,0.8864957000000001,0.3891669,1.32e-05,0.041196199999999995,0.045159,0.1718553, \
        5.4845,3.0,10.0,6.0,7.0,9.0,73.88,26.0,102.0,84.5,94.6,101.26,969,139,650,670600,False,0, \
        3,1000,7,,,4.283805244651319,qwen3-8b, \
        /data/w30031530/ModelEvalState/datasets/short_input_64_2k.jsonl,Simulator""",
        """2407.8866,0.5219253,0.07873909999999999,1.0,9.4147,1.0784282,0.0694427,0.7191335, \
        0.8201502,0.9298464,0.6178497,1.21e-05,0.0737019,0.097402,0.26629450000000005,7.5567, \
        2.0,13.0,8.0,10.0,12.0,168.3263,44.0,223.0,200.5,213.6,222.06,969,139,650,670600, \
        False,0,3,1000,10,,,4.754963590692261, \
        qwen3-8b,/data/w30031530/ModelEvalState/datasets/short_input_64_2k.jsonl,Simulator""",
        """1246.1593,0.4876233,0.0344552,1.0,4.8728,1.2842791,0.0470783,0.7252981, \
        0.7706660000000001,0.86063,0.5466711000000001,1.23e-05,0.0328953,0.0346846, \
        0.1309505,4.0541,1.0,8.0,5.0,6.0,7.0,42.0455,15.0,54.0,46.25,49.0,53.13,969,139, \
        650,670600,False,0,3,1000,5,,,5.706182639610976,qwen3-8b, \
        /data/w30031530/ModelEvalState/datasets/short_input_64_2k.jsonl,Simulator""",
        """1966.4359,0.4968034,0.0497617,1.0,7.6895,1.0411084000000002,0.0623178, \
        0.7018625,0.7927137,0.9036163,0.4520387,1.31e-05,0.0463816,0.051256100000000006, \
        0.1904178,6.0852,4.0,11.0,7.0,8.0,10.0,89.8289,33.0,122.0,104.0,114.5,121.25, \
        743,154,640,670600,False,0,1,1000,8,,,3.952181054954989,qwen3-8b, \
        /data/w30031530/ModelEvalState/datasets/short_input_64_2k.jsonl,Simulator""",
        """2157.4913,0.5037943,0.0586365,1.0,8.435,1.2262198,0.062042299999999995, \
        0.7029421,0.8011816,0.9243581000000001,0.5565085000000001,1.3e-05,0.0547994, \
        0.061973,0.2205356,6.7114,4.0,12.0,8.0,9.0,11.0,114.7286,31.0,155.0,136.75, \
        147.1,153.62,743,154,640,670600,False,0,1,1000,9,,,3.870976410628638, \
        qwen3-8b,/data/w30031530/ModelEvalState/datasets/short_input_64_2k.jsonl,Simulator""",
        """1503.396,0.4886381,0.0381119,1.0,5.8787,0.9642645,0.050871400000000004, \
        0.7220547,0.7747441,0.8654031,0.4425547,1.2800000000000001e-05,0.0362437, \
        0.0388145,0.1474528,4.7771,3.0,10.0,5.0,6.0,8.0,55.6761,24.0,71.0,61.0, \
        65.0,70.3,743,154,640,670600,False,0,1,1000,6,,,4.829008909894092,qwen3-8b, \
        /data/w30031530/ModelEvalState/datasets/short_input_64_2k.jsonl,Simulator"""
    ]
    with open(_config, "w", newline="") as f:
        for _row in datas:
            f.write(_row)
            f.write("\n")
    yield _config
    _config.unlink()


def test_computer_fitness():
    # 创建一个PSOOptimizer实例
    optimizer = PSOOptimizer(MagicMock(), target_field=default_support_field[:1])
    optimizer.minimum_algorithm = MagicMock(return_value=1.0)
    # 模拟load_history_data方法的行为
    optimizer.load_history_data = [
        {'generate_speed': 1, 'time_to_first_token': 2, 'time_per_output_token': 3, 'success_rate': 1.0,
         "max_batch_size": 150},
        {'generate_speed': 4, 'time_to_first_token': 5, 'time_per_output_token': 6, 'success_rate': 1.0,
         "max_batch_size": 160},
        {'generate_speed': 7, 'time_to_first_token': 8, 'time_per_output_token': 9, 'success_rate': 1.0,
         "max_batch_size": 170},
    ]
    # 调用computer_fitness方法
    positions, costs = optimizer.computer_fitness()

    # 检查结果
    assert np.array(positions).size == 3
    assert costs == [1.0, 1.0, 1.0]


def test_computer_fitness_with_key_error():
    # 创建一个PSOOptimizer实例
    optimizer = PSOOptimizer(MagicMock(), target_field=default_support_field)
    optimizer.minimum_algorithm = MagicMock(return_value=1.0)
    # 模拟load_history_data方法的行为
    optimizer.load_history_data = [
        {'generate_speed': 1, 'time_to_first_token': 2, 'time_per_output_token': 3, 'success_rate': 1.0,
         "max_batch_size": 3},
        {'generate_speed': 4, 'time_to_first_token': 5, 'time_per_output_token': 6, 'success_rate': 1.0,
         "max_batch_size": 4},
        {'generate_speed': 7, 'time_to_first_token': 8, 'time_per_output_token': 9, 'success_rate': 1.0,
         "max_batch_size": 5},
    ]

    # 调用computer_fitness方法
    positions, costs = optimizer.computer_fitness()

    # 检查结果，缺少字段的数据应该被忽略
    assert len(positions) == 0
    assert len(costs) == 0


# 测试数据
TEST_PARAMS = np.array([[1.0, 2.0], [3.0, 4.0]])
TEST_PARAMS_FIELD = ('field1', 'field2')


# 测试用例
def test_op_func_success():
    # 创建PSOOptimizer实例
    optimizer = PSOOptimizer(MagicMock(), target_field=default_support_field)
    # 模拟scheduler.run方法
    optimizer.scheduler.run_with_request_rate = MagicMock(return_value=PerformanceIndex(generate_speed=100,
                                                                                        time_to_first_token=0.1,
                                                                                        time_per_output_token=0.1,
                                                                                        success_rate=1))

    # 调用op_func方法
    result = optimizer.op_func(TEST_PARAMS)

    # 验证scheduler.run和minimum_algorithm方法被正确调用
    optimizer.scheduler.run_with_request_rate.assert_called()
    # 验证返回值
    assert result.size == 2


def test_op_func_exception():
    # 创建PSOOptimizer实例
    optimizer = PSOOptimizer(MagicMock(), target_field=default_support_field)

    # 模拟scheduler.run方法抛出异常
    optimizer.scheduler = MagicMock()
    optimizer.scheduler.run_with_request_rate = MagicMock(side_effect=Exception("Test Exception"))

    # 模拟minimum_algorithm方法

    # 调用op_func方法
    result = optimizer.op_func(TEST_PARAMS)

    # 验证scheduler.run和minimum_algorithm方法被正确调用
    optimizer.scheduler.run_with_request_rate.assert_called()

    # 验证返回值
    assert np.array_equal(result, np.array([float('inf'), float('inf')]))


class MockField:
    def __init__(self, min_value, max_value):
        self.min = min_value
        self.max = max_value


@patch("msserviceprofiler.modelevalstate.optimizer.optimizer.field_to_param", )
def test_refine_optimization_candidates(field_to_param_patch):
    field_to_param_patch.side_effect = [[3, 4], [7, 1]]
    pso = PSOOptimizer(MagicMock(), target_field=default_support_field)
    pso.default_res = PerformanceIndex()
    pso.default_fitness = 3.333
    pso.default_run_param = [5.5, 6.3]
    best_results = pd.DataFrame({"field1": [1, 2], "field2": [3, 4]})
    pso.scheduler = MagicMock()
    pso.scheduler.run = MagicMock(return_value=PerformanceIndex())
    pso.scheduler.save_result = MagicMock()
    pso.mindie_fine_tune = MagicMock()
    pso.mindie_fine_tune.mindie_fine_tune_with_cycle = MagicMock(side_effect=StopIteration)
    pso.minimum_algorithm = MagicMock(return_value=2)
    pso.get_target_field_from_case_data = MagicMock(return_value=default_support_field)
    pso.params_in_records = MagicMock(return_value=False)
    fitness, params, res = pso.refine_optimization_candidates(best_results)
    assert fitness == [3.333, 2, 2]
    assert params == [[5.5, 6.3], [3, 4], [7, 1]]
    assert len(res) == 3