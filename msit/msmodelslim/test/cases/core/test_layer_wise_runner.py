#  -*- coding: utf-8 -*-
#  Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import time
import unittest
from typing import Tuple, Optional
from unittest.mock import patch

import torch
import torch.nn as nn

from msmodelslim.core.base.processor import BaseProcessor
from msmodelslim.core.base.protocol import ProcessRequest, BatchProcessRequest
from msmodelslim.core.runner.generated_runner import GeneratedRunner
from msmodelslim.core.runner.layer_wise_forward import (
    generated_decoder_layer_visit_func,
    transformers_generated_forward_func
)


class DummyDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) \
            -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        return (x, None)


class ComplexModel(nn.Module):
    """复杂的测试模型，包含多个TransformerDecoder层"""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.decoder1 = DummyDecoderLayer()
        self.decoder2 = DummyDecoderLayer()
        self.decoder3 = DummyDecoderLayer()

    def forward(self, x):
        x = self.linear(x)
        x = self.decoder1(x)
        x = self.decoder2(x)
        x = self.decoder3(x)
        return x


class SimpleModel(nn.Module):
    """简单的测试模型，包含一个线性层和一个TransformerDecoder层"""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.decoder = nn.TransformerDecoderLayer(d_model=10, nhead=2)

    def forward(self, x):
        x = self.linear(x)
        x = self.decoder(x, x)
        return x


class DummyProcessor(BaseProcessor):
    """测试用的处理器，用于测试LayerWiseRunner的功能"""

    def __init__(self, model, name="default"):
        super().__init__(model)
        self.processed_modules = []
        self.name = name

    def process(self, request: BatchProcessRequest):
        # 记录处理时间和模块信息
        self.processed_modules.append((self.name, request.name, time.time()))
        time.sleep(100 / 1000)
        super().process(request)


class TestLayerWiseRunner(unittest.TestCase):
    """测试LayerWiseRunner的功能"""

    def setUp(self):
        """测试前的准备工作"""
        self.model = SimpleModel()
        self.complex_model = ComplexModel()
        self.runner = GeneratedRunner(self.model,
                                      generated_forward_func=transformers_generated_forward_func,
                                      generated_visit_func=generated_decoder_layer_visit_func
                                      )
        self.processor = DummyProcessor(self.model)

    @patch('torch.distributed.get_rank')
    @patch('torch.distributed.get_world_size')
    @patch('torch.distributed.barrier')
    def test_run(self, mock_barrier, mock_get_world_size, mock_get_rank):
        """测试运行功能"""
        # 设置模拟函数
        mock_get_rank.return_value = 0
        mock_get_world_size.return_value = 1

        # 添加处理器
        self.runner.add_processor(self.processor)

        # 运行
        self.runner.run()

    @patch('torch.distributed.get_rank')
    @patch('torch.distributed.get_world_size')
    @patch('torch.distributed.barrier')
    def test_transformers_generated_visit_func(self, mock_barrier, mock_get_world_size, mock_get_rank):
        """测试transformers_generated_visit_func函数"""
        # 设置模拟函数
        mock_get_rank.return_value = 0
        mock_get_world_size.return_value = 1

        # 获取生成器
        generator = generated_decoder_layer_visit_func(self.model)

        # 获取第一个请求
        request = next(generator)

        # 验证请求
        self.assertIsInstance(request, ProcessRequest)
        self.assertEqual(request.name, "decoder")
        self.assertEqual(request.module, self.model.decoder)

    @patch('torch.distributed.get_rank')
    @patch('torch.distributed.get_world_size')
    @patch('torch.distributed.barrier')
    def test_transformers_generated_forward_func(self, mock_barrier, mock_get_world_size, mock_get_rank):
        """测试_transformers_generated_forward_func函数"""
        # 设置模拟函数
        mock_get_rank.return_value = 0
        mock_get_world_size.return_value = 1

        # 创建输入数据
        input_data = torch.randn(5, 10)

        # 获取生成器
        generator = transformers_generated_forward_func(self.model, input_data)

        # 获取第一个请求
        request = next(generator)

        # 验证请求
        self.assertIsInstance(request, ProcessRequest)
        self.assertEqual(request.name, "decoder")
        self.assertEqual(request.module, self.model.decoder)

    @patch('torch.distributed.get_rank')
    @patch('torch.distributed.get_world_size')
    @patch('torch.distributed.barrier')
    def test_layer_schedule_with_datafree(self, mock_barrier, mock_get_world_size, mock_get_rank):
        """测试LayerWiseRunner在复杂模型上的交错调度特性"""
        # 设置模拟函数
        mock_get_rank.return_value = 0
        mock_get_world_size.return_value = 1

        # 创建一个新的runner，使用复杂模型
        complex_runner = GeneratedRunner(self.complex_model,
                                         generated_forward_func=transformers_generated_forward_func,
                                         generated_visit_func=generated_decoder_layer_visit_func
                                         )

        # 创建多个处理器
        processor1 = DummyProcessor(self.complex_model, "processor1")
        processor2 = DummyProcessor(self.complex_model, "processor2")
        processor3 = DummyProcessor(self.complex_model, "processor3")

        # 添加处理器到runner
        complex_runner.add_processor(processor1)
        complex_runner.add_processor(processor2)
        complex_runner.add_processor(processor3)

        # 运行
        complex_runner.run()

        # 验证处理器的执行顺序
        # 由于每个处理器处理相同的模型层，所以它们应该按照添加的顺序交错执行
        # 每个处理器都应该处理了三个decoder层
        self.assertEqual(len(processor1.processed_modules), 3)
        self.assertEqual(len(processor2.processed_modules), 3)
        self.assertEqual(len(processor3.processed_modules), 3)

        # 验证每个处理器都处理了三个decoder层
        for i in range(3):
            self.assertEqual(processor1.processed_modules[i][1], f"decoder{i + 1}")
            self.assertEqual(processor2.processed_modules[i][1], f"decoder{i + 1}")
            self.assertEqual(processor3.processed_modules[i][1], f"decoder{i + 1}")

        # 通过时间戳验证处理器的执行顺序
        # 对于每个decoder层，processor1应该先执行，然后是processor2，最后是processor3
        for i in range(3):
            # 获取每个处理器处理decoder{i+1}的时间戳
            time1 = processor1.processed_modules[i][2]
            time2 = processor2.processed_modules[i][2]
            time3 = processor3.processed_modules[i][2]

            # 验证时间戳的顺序
            self.assertLess(time1, time2, f"processor1应该先于processor2处理decoder{i + 1}")
            self.assertLess(time2, time3, f"processor2应该先于processor3处理decoder{i + 1}")

        # 验证不同decoder层的处理顺序
        # decoder1应该先于decoder2处理，decoder2应该先于decoder3处理
        for p in [processor1, processor2, processor3]:
            time1 = p.processed_modules[0][2]  # decoder1的时间戳
            time2 = p.processed_modules[1][2]  # decoder2的时间戳
            time3 = p.processed_modules[2][2]  # decoder3的时间戳

            self.assertLess(time1, time2, f"{p.name}应该先处理decoder1，再处理decoder2")
            self.assertLess(time2, time3, f"{p.name}应该先处理decoder2，再处理decoder3")

        # 验证decoder1被processor3处理的时间早于decoder2被processor1处理的时间
        decoder1_processor3_time = processor3.processed_modules[0][2]  # decoder1被processor3处理的时间
        decoder2_processor1_time = processor1.processed_modules[1][2]  # decoder2被processor1处理的时间

        self.assertLess(decoder1_processor3_time, decoder2_processor1_time,
                        "decoder1被processor3处理的时间必须早于decoder2被processor1处理的时间")

    @patch('torch.distributed.get_rank')
    @patch('torch.distributed.get_world_size')
    @patch('torch.distributed.barrier')
    def test_layer_schedule_with_input(self, mock_barrier, mock_get_world_size, mock_get_rank):
        """测试LayerWiseRunner在复杂模型上的带输入数据的逐层调度特性"""
        # 设置模拟函数
        mock_get_rank.return_value = 0
        mock_get_world_size.return_value = 1

        # 创建一个新的runner，使用复杂模型
        complex_runner = GeneratedRunner(self.complex_model,
                                         generated_forward_func=transformers_generated_forward_func,
                                         generated_visit_func=generated_decoder_layer_visit_func)

        # 创建多个处理器
        processor1 = DummyProcessor(self.complex_model, "processor1")
        processor2 = DummyProcessor(self.complex_model, "processor2")
        processor3 = DummyProcessor(self.complex_model, "processor3")

        # 创建输入数据
        input_data = torch.randn(5, 10)

        # 添加处理器到runner，并传入输入数据
        complex_runner.add_processor(processor1, [input_data])
        complex_runner.add_processor(processor2, [input_data])
        complex_runner.add_processor(processor3, [input_data])

        # 运行
        complex_runner.run()

        # 验证处理器的执行顺序
        # 由于每个处理器处理相同的模型层，所以它们应该按照添加的顺序交错执行
        # 每个处理器都应该处理了三个decoder层
        self.assertEqual(len(processor1.processed_modules), 3)
        self.assertEqual(len(processor2.processed_modules), 3)
        self.assertEqual(len(processor3.processed_modules), 3)

        # 验证每个处理器都处理了三个decoder层
        for i in range(3):
            self.assertEqual(processor1.processed_modules[i][1], f"decoder{i + 1}")
            self.assertEqual(processor2.processed_modules[i][1], f"decoder{i + 1}")
            self.assertEqual(processor3.processed_modules[i][1], f"decoder{i + 1}")

        # 通过时间戳验证处理器的执行顺序
        # 对于每个decoder层，processor1应该先执行，然后是processor2，最后是processor3
        for i in range(3):
            # 获取每个处理器处理decoder{i+1}的时间戳
            time1 = processor1.processed_modules[i][2]
            time2 = processor2.processed_modules[i][2]
            time3 = processor3.processed_modules[i][2]

            # 验证时间戳的顺序
            self.assertLess(time1, time2, f"processor1应该先于processor2处理decoder{i + 1}")
            self.assertLess(time2, time3, f"processor2应该先于processor3处理decoder{i + 1}")

        # 验证不同decoder层的处理顺序
        # decoder1应该先于decoder2处理，decoder2应该先于decoder3处理
        for p in [processor1, processor2, processor3]:
            time1 = p.processed_modules[0][2]  # decoder1的时间戳
            time2 = p.processed_modules[1][2]  # decoder2的时间戳
            time3 = p.processed_modules[2][2]  # decoder3的时间戳

            self.assertLess(time1, time2, f"{p.name}应该先处理decoder1，再处理decoder2")
            self.assertLess(time2, time3, f"{p.name}应该先处理decoder2，再处理decoder3")

        # 验证decoder1被processor3处理的时间早于decoder2被processor1处理的时间
        decoder1_processor3_time = processor3.processed_modules[0][2]  # decoder1被processor3处理的时间
        decoder2_processor1_time = processor1.processed_modules[1][2]  # decoder2被processor1处理的时间

        self.assertLess(decoder1_processor3_time, decoder2_processor1_time,
                        "decoder1被processor3处理的时间必须早于decoder2被processor1处理的时间")


if __name__ == '__main__':
    unittest.main()
