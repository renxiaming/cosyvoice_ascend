#  Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from msmodelslim.app.quant_service.dataset_interface import DatasetLoaderInterface
from ..base.model import BaseModelAdapter
from ..base.quant_config import BaseQuantConfig


class BaseQuantService(ABC):
    backend_name: str = "Unknown"

    def __init__(self, dataset_loader: DatasetLoaderInterface):
        self.dataset_loader = dataset_loader

    @abstractmethod
    def quantize(
            self,
            model: BaseModelAdapter,
            quant_config: BaseQuantConfig,
            save_path: Optional[Path] = None
    ) -> None:
        raise NotImplementedError
