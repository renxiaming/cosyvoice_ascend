# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import torch

from ..quant_service.dataset_interface import DatasetLoaderInterface
from ..base.model import BaseModelAdapter


class BaseAnalysisService(ABC):
    """Base class for model analysis services"""
    
    def __init__(self, dataset_loader: DatasetLoaderInterface):
        self.dataset_loader = dataset_loader
    
    @abstractmethod
    def analyze(self, 
                model: BaseModelAdapter, 
                patterns: List[str], 
                analysis_config: Optional[Dict[str, Any]] = None):
        """
        Analyze model layers based on given patterns
        
        Args:
            model: The model to analyze
            patterns: List of layer name patterns to analyze (e.g., ['*linear*', 'attention.*'])
            analysis_config: Configuration for analysis method

        """
        raise NotImplementedError
    
    @abstractmethod
    def export_results(self, result: Any, top_k: int = 10):
        """
        export analysis results in service-specific format
        
        Args:
            result: AnalysisResult containing layer scores and metadata
            top_k: Number of top layers to display
        """
        raise NotImplementedError 