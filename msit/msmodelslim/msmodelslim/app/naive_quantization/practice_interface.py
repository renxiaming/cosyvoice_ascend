# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator

from msmodelslim.app.base.quant_config import BaseQuantConfig


class PracticeManagerInterface(ABC):
    @abstractmethod
    def get_config_by_id(self, model_pedigree, config_id: str) -> BaseQuantConfig:
        """Get configuration by ID"""
        raise NotImplementedError

    @abstractmethod
    def get_config_by_path(self, config_path: Path) -> BaseQuantConfig:
        """Get configuration by path"""
        raise NotImplementedError

    @abstractmethod
    def iter_config(self, model_pedigree) -> Generator[BaseQuantConfig, None, None]:
        """Iterate configurations by priority"""
        raise NotImplementedError
