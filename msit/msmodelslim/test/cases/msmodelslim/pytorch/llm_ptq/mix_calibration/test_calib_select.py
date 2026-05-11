import pytest
import json
import os
import tempfile
import pandas as pd
from unittest.mock import MagicMock
from transformers import PreTrainedModel, AutoTokenizer, PreTrainedTokenizerBase
from msmodelslim.pytorch.llm_ptq.mix_calibration.calib_select import (
    DatasetFactory,
    CalibHandler,
    CalibrationData,
    SUPPORTED_DATASET_NAME
)


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock(spec=PreTrainedTokenizerBase)
    tokenizer.eos_token_id = "[EOS]"
    tokenizer.pad_token_id = "[PAD]"
    return tokenizer


@pytest.fixture
def mock_model():
    model = MagicMock(spec=PreTrainedModel)
    model.device = "cpu"
    return model


@pytest.fixture
def tmp_path():
    with tempfile.TemporaryDirectory() as tmp:
        """创建临时测试配置文件"""
        config = {
            "configurations": [
                {
                    "dataset_name": "boolq",
                    "dataset_path": str(os.path.join(tmp, "boolq", "boolq.jsonl"))
                },
                {
                    "dataset_name": "ceval_5_shot",
                    "dataset_path": str(os.path.join(tmp, "ceval_5_shot"))
                },
                {
                    "dataset_name": "gsm8k",
                    "dataset_path": str(os.path.join(tmp, "gsm8k", "GSM8K.jsonl"))
                },
                {
                    "dataset_name": "mmlu",
                    "dataset_path": str(os.path.join(tmp, "mmlu"))
                }
            ]
        }

        # 1. 保存原始 umask
        original_umask = os.umask(0)  # 临时设为 0 并获取原始值
        try:
            # 2. 设置目标 umask（0o027 对应权限 640）
            os.umask(0o027)
            with open(f"{tmp}/test_config.json", "w") as f:
                json.dump(config, f)

            # 创建临时数据目录
            os.mkdir(os.path.join(tmp, "boolq"))
            os.mkdir(os.path.join(tmp, "ceval_5_shot"))
            os.mkdir(os.path.join(tmp, "ceval_5_shot", "val"))
            os.mkdir(os.path.join(tmp, "ceval_5_shot", "dev"))
            os.open(os.path.join(tmp, "ceval_5_shot", "dev", "test_subject_dev.csv"), os.O_CREAT, 0o644)
            os.open(os.path.join(tmp, "ceval_5_shot", "val", "test_subject_val.csv"), os.O_CREAT, 0o644)
            os.mkdir(os.path.join(tmp, "gsm8k"))
            os.mkdir(os.path.join(tmp, "mmlu"))
            os.mkdir(os.path.join(tmp, "mmlu", "val"))
            os.mkdir(os.path.join(tmp, "mmlu", "dev"))

            # 创建示例数据文件
            with open(os.path.join(tmp, "boolq", "boolq.jsonl"), "w") as f:
                f.write(json.dumps({
                    "title": "Test Title",
                    "question": "Is this a test?",
                    "passage": "This is a test passage",
                    "answer": True
                }) + "\n")

            # 创建ceval测试数据
            subject_map = {}
            with open(os.path.join(tmp, "ceval_5_shot", "subject_mapping.json"), "w") as f:
                json.dump(subject_map, f)

            dev_data = [
                ["Q1", "A", "B", "C", "D", "A"],
                ["Q2", "A", "B", "C", "D", "A"]
            ]
            val_data = [
                ["Q1", "A", "B", "C", "D", "A"],
                ["Q2", "A", "B", "C", "D", "A"]
            ]

            pd.DataFrame(dev_data).to_csv(os.path.join(tmp, "ceval_5_shot", "dev", "test_subject_dev.csv"), header=False,
                                        index=False)
            pd.DataFrame(val_data).to_csv(os.path.join(tmp, "ceval_5_shot", "val", "test_subject_val.csv"), header=False,
                                        index=False)

            # 创建mmlu测试数据
            subject_map = {"mmlu_all_sets": []}
            with open(os.path.join(tmp, "mmlu", "subject_mapping.json"), "w") as f:
                json.dump(subject_map, f)

            # 创建gsm8k测试数据
            with open(os.path.join(tmp, "gsm8k", "GSM8K.jsonl"), "w") as f:
                f.write(json.dumps({
                    "question": "What is 2 + 2?",
                    "answer": "The answer is <<2+2=4>>4#### 4"
                }) + "\n")
        finally:
            # 4. 无论是否出错，都恢复原始 umask
            os.umask(original_umask)

        yield tmp


class TestDatasetFactory:
    @pytest.mark.parametrize("dataset_name", SUPPORTED_DATASET_NAME)
    def test_create_supported_processors(self, dataset_name, tmp_path):
        config = {
            "boolq": str(os.path.join(tmp_path, "boolq", "boolq.jsonl")),
            "ceval_5_shot": str(os.path.join(tmp_path, "ceval_5_shot")),
            "gsm8k": str(os.path.join(tmp_path, "gsm8k", "GSM8K.jsonl")),
            "mmlu": str(os.path.join(tmp_path, "mmlu"))
        }

        processor = DatasetFactory.create_dataset_processor(
            dataset_name=dataset_name,
            dataset_path=config[dataset_name],
            customized_dataset_processor={}
        )
        assert processor is not None
        assert processor.__class__.__name__ == f"{dataset_name.capitalize()}Processor" if dataset_name != "ceval_5_shot" else processor.__class__.__name__ == "CEval5ShotProcessor"

    def test_create_invalid_processor(self):
        processor = DatasetFactory.create_dataset_processor(
            dataset_name="invalid_dataset",
            dataset_path="",
            customized_dataset_processor={}
        )
        assert processor is None


class TestCalibHandler:
    def test_basic_sampling(self, mock_tokenizer, tmp_path):
        # 初始化boolq processor
        processor = DatasetFactory.create_dataset_processor(
            "boolq", str(os.path.join(tmp_path, "boolq", "boolq.jsonl")), {}, mock_tokenizer)

        handler = CalibHandler(
            dataset_name="boolq",
            dataset_processor=processor,
            shuffle_seed=42,
            batch_size=2
        )
        handler.set_sample_size(1)

        samples = handler.run()
        assert len(samples) == 1
        assert "prompt" in samples[0]
        assert "ans" in samples[0]

    def test_positive_prompt_verification(self, mock_tokenizer, mock_model, tmp_path):
        # 配置mock model返回值
        mock_model.generate.return_value = ["aaaaa"]
        mock_tokenizer.decode.return_value = "A"
        mock_model.device = "cpu"

        processor = DatasetFactory.create_dataset_processor(
            "gsm8k", str(os.path.join(tmp_path, "gsm8k", "GSM8K.jsonl")), {}, mock_tokenizer, mock_model)

        handler = CalibHandler(
            dataset_name="gsm8k",
            dataset_processor=processor,
            shuffle_seed=42,
            batch_size=2
        )
        handler.set_sample_size(1)

        samples = handler.run(need_positive_prompt=True)
        assert len(samples) <= 1  # 验证可能过滤部分样本


class TestCalibrationData:
    def test_config_loading(self, tmp_path, mock_tokenizer):
        calib = CalibrationData(
            config_path=str(os.path.join(tmp_path, "test_config.json")),
            tokenizer=mock_tokenizer
        )

        assert len(calib.handlers) == 4
        assert "boolq" in calib.handlers
        assert "ceval_5_shot" in calib.handlers

    def test_mixed_dataset_generation(self, tmp_path, mock_tokenizer):
        save_path = str(os.path.join(tmp_path, "output.json"))
        calib = CalibrationData(
            config_path=str(os.path.join(tmp_path, "test_config.json")),
            tokenizer=mock_tokenizer,
            save_path=save_path
        )
        calib.set_sample_size({"boolq": 1, "ceval_5_shot": 1})

        mixed_data = calib.process()
        assert len(mixed_data) == 1
        assert os.path.exists(save_path)

    def test_invalid_model_handling(self, tmp_path):
        with pytest.raises(TypeError):
            CalibrationData(
                config_path=str(os.path.join(tmp_path, "test_config.json")),
                tokenizer=MagicMock(),
                model="invalid_model_type"
            )


# 验证各processor的数据解析逻辑
@pytest.mark.parametrize("dataset_name", SUPPORTED_DATASET_NAME)
def test_dataset_processors(tmp_path, dataset_name):
    config = {
        "boolq": str(os.path.join(tmp_path, "boolq", "boolq.jsonl")),
        "ceval_5_shot": str(os.path.join(tmp_path, "ceval_5_shot")),
        "gsm8k": str(os.path.join(tmp_path, "gsm8k", "GSM8K.jsonl")),
        "mmlu": str(os.path.join(tmp_path, "mmlu"))
    }

    # 初始化processor
    processor = DatasetFactory.create_dataset_processor(
        dataset_name=dataset_name,
        dataset_path=config[dataset_name],
        customized_dataset_processor={}
    )

    # 验证基础属性
    assert processor.dataset_path == str(config[dataset_name])
