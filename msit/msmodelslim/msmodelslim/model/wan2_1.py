# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import logging
import os
import time
import argparse
import random
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Generator, List, Literal, cast
import torch
from torch import nn, distributed as dist
from tqdm import tqdm
from msmodelslim.app.base.const import DeviceType, PipelineType
from msmodelslim.app.quant_service.modelslim_v1.multimodal_sd_quant_inference import MultimodalSDQuantInference
from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.runner.generated_runner import GeneratedForwardFuncType, GeneratedVisitFuncType
from msmodelslim.core.runner.layer_wise_forward import _TransformersForwardBreak
from msmodelslim.core.runner.model_wise_forward import model_wise_forward_func, model_wise_visit_func
from msmodelslim.model.default_multimodal_sd import MultimodalSDModelAdapter
from msmodelslim.model.factory import ModelFactory
from msmodelslim.utils.cache import to_device
from msmodelslim.utils.exception import InvalidModelError, SchemaValidateError, UnsupportedError

MAX_RECURSION_DEPTH = 20

EXAMPLE_PROMPT = {
    "t2v-1.3B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted "
                  "stage.",
    },
    "t2v-14B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted "
                  "stage.",
    },
    "t2i-14B": {
        "prompt": "一个朴素端庄的美人",
    },
    "i2v-14B": {
        "prompt":
            "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred "
            "feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the "
            "background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. "
            "The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up "
            "shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image":
            "examples/i2v_input.JPG",
    },
}


@ModelFactory.register("Wan2_1")
class Wan2Point1Adapter(MultimodalSDModelAdapter, MultimodalSDQuantInference):
    def __init__(self,
                 model_type: str,
                 ori_path: Path,
                 device: DeviceType = DeviceType.NPU,
                 trust_remote_code: bool = False,
                 **kwargs):
        super().__init__(model_type, ori_path, device, trust_remote_code, **kwargs)

    @property
    def model(self):
        """
        The model of the model.
        """
        return self._model

    def set_model_args(self, override_model_config: object):
        # 覆盖默认参数配置
        self._set_model_args(override_model_config)

    def add_attentioncache_args(self, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(title="Attention Cache args")

        group.add_argument("--use_attentioncache", action='store_true')
        group.add_argument("--attentioncache_ratio", type=float, default=1.2)
        group.add_argument("--attentioncache_interval", type=int, default=4)
        group.add_argument("--start_step", type=int, default=12)
        group.add_argument("--end_step", type=int, default=37)

        return parser

    def setup_cache(self):
        """设置Cache机制"""
        try:
            from mindiesd import CacheConfig, CacheAgent
        except ImportError as e:
            # Concise import error message
            raise ImportError(
                "Failed to import required components from mindiesd. "
            ) from e

        if self.model_args.use_attentioncache:
            config = CacheConfig(
                method="attention_cache",
                blocks_count=len(self.transformer.blocks),
                steps_count=self.model_args.sample_steps,
                step_start=self.model_args.start_step,
                step_interval=self.model_args.attentioncache_interval,
                step_end=self.model_args.end_step
            )
        else:
            config = CacheConfig(
                method="attention_cache",
                blocks_count=len(self.transformer.blocks),
                steps_count=self.model_args.sample_steps
            )
        cache = CacheAgent(config)
        if self.model_args.dit_fsdp:
            for block in self.transformer._fsdp_wrapped_module.blocks:
                block._fsdp_wrapped_module.cache = cache
                block._fsdp_wrapped_module.args = self.model_args
        else:
            for block in self.transformer.blocks:
                block.cache = cache
                block.args = self.model_args

    def load_pipeline(self):
        self._load_pipeline()
        self.setup_cache()

    def run_calib_inference(self):
        """运行校准推理"""
        from wan.configs import SIZE_CONFIGS
        stream = torch.npu.Stream()
        args = self.model_args

        self.wan_t2v.model.to('npu')
        # Start sampling
        for _ in tqdm(range(1), desc='Dump calib data by float model inference'):
            # set seed
            torch.manual_seed(args.base_seed)
            torch.npu.manual_seed(args.base_seed)
            torch.npu.manual_seed_all(args.base_seed)

            begin = time.time()
            video = self.wan_t2v.generate(
                self.model_args.prompt,
                size=SIZE_CONFIGS[args.size],
                frame_num=args.frame_num,
                shift=args.sample_shift,
                sample_solver=args.sample_solver,
                sampling_steps=args.sample_steps,
                guide_scale=args.sample_guide_scale,
                seed=args.base_seed,
                offload_model=args.offload_model)
            stream.synchronize()
            end = time.time()
            logging.info(f"Generating video used time {end - begin: .4f}s")

    def transformers_generated_forward_func(self, model: torch.nn.Module,
                                            inputs: Any,
                                            ) -> Generator[ProcessRequest, Any, None]:
        transformer_blocks = [
            (name, module)
            for name, module in model.named_modules()
            if "attentionblock" in module.__class__.__name__.lower()
        ]

        # 存储第一个transformer block的输入
        first_block_input = None

        def break_hook(module: nn.Module, hook_args: Tuple[Any, ...], hook_kwargs: Dict[str, Any]):
            nonlocal first_block_input
            first_block_input = (hook_args, hook_kwargs,)
            raise _TransformersForwardBreak()

        hooks = [transformer_blocks[0][1].register_forward_pre_hook(break_hook, with_kwargs=True)]

        # 执行一次前向传播以获取输入
        try:
            if isinstance(inputs, list) or isinstance(inputs, tuple):
                model(*inputs)
            elif isinstance(inputs, dict):
                model(**inputs)
            else:
                model(inputs)
        except _TransformersForwardBreak:
            pass
        except Exception as e:
            raise e
        finally:
            for hook in hooks:
                hook.remove()

        if first_block_input is None:
            raise InvalidModelError("Can't get first block input.", action="Please check the model and input")

        # 循环处理每个transformer block
        first_block_input = to_device(first_block_input, 'cpu')
        current_inputs = first_block_input

        if dist.is_initialized():
            dist.barrier()

        for name, block in transformer_blocks:
            args, kwargs = current_inputs
            outputs = yield ProcessRequest(name, block, args, kwargs)
            hidden_states = outputs
            current_inputs = ((hidden_states,), current_inputs[1])

    def generated_decoder_layer_visit_func(self, model: torch.nn.Module,
                                           transformer_blocks: Optional[List[Tuple[str, torch.nn.Module]]] = None,
                                           ) -> Generator[ProcessRequest, Any, None]:
        if transformer_blocks is None:
            transformer_blocks = [
                (name, module)
                for name, module in model.named_modules()
                if "attentionblock" in module.__class__.__name__.lower()
            ]

        if dist.is_initialized():
            dist.barrier()

        for name, block in transformer_blocks:
            yield ProcessRequest(name, block, [], {})

    def get_pipeline_functions(self, pipeline: Literal[PipelineType.MODEL_WISE, PipelineType.LAYER_WISE]) -> Tuple[
        GeneratedForwardFuncType, GeneratedVisitFuncType]:
        """根据pipeline类型获取对应的forward和visit函数。

        Args:
            pipeline: pipeline类型，'model_wise' 或 'layer_wise'

        Returns:
            Tuple[GeneratedForwardFuncType, GeneratedVisitFuncType]: forward函数和visit函数的元组
        """
        if pipeline == PipelineType.MODEL_WISE:
            generated_forward_func = cast(GeneratedForwardFuncType, model_wise_forward_func)
            generated_visit_func = cast(GeneratedVisitFuncType, model_wise_visit_func)
        else:
            generated_forward_func = cast(GeneratedForwardFuncType, self.transformers_generated_forward_func)
            generated_visit_func = cast(GeneratedVisitFuncType, self.generated_decoder_layer_visit_func)

        return generated_forward_func, generated_visit_func

    def apply_quantization(self, process_model_func):
        from contextlib import contextmanager
        import torch.cuda.amp as amp

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self, 'no_sync', noop_no_sync)

        # 遍历所有子模块，将非blocks部分移至npu
        for name, module in self.transformer.named_modules():
            # 处理非blocks模块：确保在npu上
            if not name.startswith('blocks'):
                module.to('npu')
            # 处理blocks模块：确保在cpu上
            else:
                module.to('cpu')
        with amp.autocast(dtype=torch.bfloat16), torch.no_grad(), no_sync():
            process_model_func(model=self.transformer, adapter=self)

    def support_layer_wise_schedule(self) -> bool:
        return True

    def _set_model_args(self, override_model_config: object):
        """
        将 override_model_config 的属性更新到 model_args
        :param override_model_config: 来自 YAML 的配置对象
        """
        # 模型路径
        self.model_args.ckpt_dir = self.ori

        missing_attrs = []
        for key in override_model_config.keys():
            if not hasattr(self.model_args, key):
                missing_attrs.append(key)

        if missing_attrs:
            available = [a for a in dir(self.model_args)]
            raise SchemaValidateError(
                f"illegal config attributes: {missing_attrs}. \n"
                f"supported config attributes: {available}"
            )

        # 执行更新
        for key in override_model_config.keys():
            setattr(self.model_args, key, override_model_config[key])

        # 验证参数配置
        self._validate_args(self.model_args)

        parser = self._get_parser()
        # 1. 把 Namespace 还原为等价的命令行参数
        argv = []
        for key, val in vars(self.model_args).items():
            if val is None:  # 允许 None 的参数直接跳过
                continue
            # flag 型（bool/存根）特殊处理
            if key == "offload_model":  # 专门处理 offload_model，让它带值
                argv.extend(["--offload_model", str(val).lower()])
            elif isinstance(val, bool):
                if val:  # True -> --flag
                    argv.append(f"--{key}")
                # False 时忽略
            else:
                argv.extend([f"--{key}", str(val)])

        # 2. 重新解析，得到经过校验/类型转换的新 Namespace
        self.model_args = parser.parse_args(argv)

    def _initialize_torch_dtype(self):
        """初始化torch dtype，子类可覆盖实现"""
        return torch.bfloat16 if self._device is DeviceType.NPU else torch.float32

    def _get_model_pedigree(self) -> str:
        return 'wan2_1'

    def _load_model(self, device: DeviceType = DeviceType.NPU, trust_remote_code: bool = False):
        pass

    def _load_config(self):
        pass

    def _load_hook(self) -> None:
        pass

    def _persist_hook(self) -> None:
        pass

    def _load_tokenizer(self, trust_remote_code=False):
        pass

    def _check_import_dependency(self):
        try:
            import wan
            from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES
            from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
            from wan.utils.utils import cache_video, cache_image, str2bool
            from wan.distributed.parallel_mgr import ParallelConfig, init_parallel_env, finalize_parallel_env
            from wan.distributed.tp_applicator import TensorParallelApplicator
        except ImportError as e:
            # Concise import error message
            raise ImportError(
                "Failed to import required components from wan. "
                "Please install the Wan2.1 from Modelers, "
                "make sure you can run the original floating-point inference successfully, "
                "and add the Wan2.1 repository to the Python search path environment variable PYTHONPATH. "
                "e.g. export PYTHONPATH=/path/to/Wan2.1:$PYTHONPATH"
            ) from e

    def _validate_args(self, args):
        """Get default parameter configuration, integrating wan config parameters"""
        self._check_import_dependency()
        from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES

        # Basic check
        if args.ckpt_dir is None:
            raise InvalidModelError("Please specify the checkpoint directory.")
        if args.task not in WAN_CONFIGS:
            raise UnsupportedError(f"Unsupport task: {args.task}")
        if args.task not in EXAMPLE_PROMPT:
            raise UnsupportedError(f"Unsupport task: {args.task}")

        # The default sampling steps are 40 for image-to-video tasks and 50 for text-to-video tasks.
        if args.sample_steps is None:
            args.sample_steps = 40 if "i2v" in args.task else 50

        if args.sample_shift is None:
            args.sample_shift = 5.0
            if "i2v" in args.task and args.size in ["832*480", "480*832"]:
                args.sample_shift = 3.0

        # The default number of frames are 1 for text-to-image tasks and 81 for other tasks.
        if args.frame_num is None:
            args.frame_num = 1 if "t2i" in args.task else 81

        # T2I frame_num check
        if "t2i" in args.task and args.frame_num != 1:
            raise UnsupportedError(f"Unsupport frame_num {args.frame_num} for task {args.task}")

        args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
            0, sys.maxsize)
        # Size check
        if args.size not in SUPPORTED_SIZES[args.task]:
            raise UnsupportedError(
                f"Unsupport size {args.size} for task {args.task},"
                f" supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])} "
            )

    def _get_parser(self) -> Dict[str, Any]:
        """Get default parameter configuration, integrating wan config parameters"""
        self._check_import_dependency()
        from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES
        from wan.utils.utils import cache_video, cache_image, str2bool

        # Create argument parser and add all necessary configurations
        parser = argparse.ArgumentParser(
            description="Generate a image or video from a text prompt or image using Wan"
        )
        parser.add_argument(
            "--task",
            type=str,
            default="t2v-14B",
            choices=list(WAN_CONFIGS.keys()),
            help="The task to run.")
        parser.add_argument(
            "--size",
            type=str,
            default="1280*720",
            choices=list(SIZE_CONFIGS.keys()),
            help="The area (width*height) of the generated video. For the I2V task,"
                 "the aspect ratio of the output video will follow that of the input image."
        )
        parser.add_argument(
            "--frame_num",
            type=int,
            default=None,
            help="How many frames to sample from a image or video. The number should be 4n+1"
        )
        parser.add_argument(
            "--ckpt_dir",
            type=str,
            default=None,
            help="The path to the checkpoint directory.")
        parser.add_argument(
            "--offload_model",
            type=str2bool,
            default=None,
            help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
        )
        parser.add_argument(
            "--cfg_size",
            type=int,
            default=1,
            help="The size of the cfg parallelism in DiT.")
        parser.add_argument(
            "--ulysses_size",
            type=int,
            default=1,
            help="The size of the ulysses parallelism in DiT.")
        parser.add_argument(
            "--ring_size",
            type=int,
            default=1,
            help="The size of the ring attention parallelism in DiT.")
        parser.add_argument(
            "--tp_size",
            type=int,
            default=1,
            help="The size of the tensor parallelism in DiT.")
        parser.add_argument(
            "--vae_parallel",
            action="store_true",
            default=False,
            help="Whether to use parallel for vae.")
        parser.add_argument(
            "--t5_fsdp",
            action="store_true",
            default=False,
            help="Whether to use FSDP for T5.")
        parser.add_argument(
            "--t5_cpu",
            action="store_true",
            default=False,
            help="Whether to place T5 model on CPU.")
        parser.add_argument(
            "--dit_fsdp",
            action="store_true",
            default=False,
            help="Whether to use FSDP for DiT.")
        parser.add_argument(
            "--save_file",
            type=str,
            default=None,
            help="The file to save the generated image or video to.")
        parser.add_argument(
            "--prompt",
            type=str,
            default=None,
            help="The prompt to generate the image or video from.")
        parser.add_argument(
            "--use_prompt_extend",
            action="store_true",
            default=False,
            help="Whether to use prompt extend.")
        parser.add_argument(
            "--prompt_extend_method",
            type=str,
            default="local_qwen",
            choices=["dashscope", "local_qwen"],
            help="The prompt extend method to use.")
        parser.add_argument(
            "--prompt_extend_model",
            type=str,
            default=None,
            help="The prompt extend model to use.")
        parser.add_argument(
            "--prompt_extend_target_lang",
            type=str,
            default="zh",
            choices=["zh", "en"],
            help="The target language of prompt extend.")
        parser.add_argument(
            "--base_seed",
            type=int,
            default=-1,
            help="The seed to use for generating the image or video.")
        parser.add_argument(
            "--image",
            type=str,
            default=None,
            help="The image to generate the video from.")
        parser.add_argument(
            "--sample_solver",
            type=str,
            default='unipc',
            choices=['unipc', 'dpm++'],
            help="The solver used to sample.")
        parser.add_argument(
            "--sample_steps", type=int, default=None, help="The sampling steps.")
        parser.add_argument(
            "--sample_shift",
            type=float,
            default=None,
            help="Sampling shift factor for flow matching schedulers.")
        parser.add_argument(
            "--sample_guide_scale",
            type=float,
            default=5.0,
            help="Classifier free guidance scale.")
        parser = self.add_attentioncache_args(parser)
        return parser

    def _get_default_model_args(self):

        parser = self._get_parser()
        args = parser.parse_args([])
        self.model_args = args

    def _init_logging(self, rank):
        # logging
        if rank == 0:
            # set format
            logging.basicConfig(
                level=logging.INFO,
                format="[%(asctime)s] %(levelname)s: %(message)s",
                handlers=[logging.StreamHandler(stream=sys.stdout)])
        else:
            logging.basicConfig(level=logging.ERROR)

    def _load_pipeline(self):
        # 动态导入外部依赖
        self._check_import_dependency()

        import wan
        from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES

        rank = int(os.getenv("RANK", 0))
        world_size = int(os.getenv("WORLD_SIZE", 1))
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        device = local_rank
        self._init_logging(rank)

        args = self.model_args
        cfg = WAN_CONFIGS[args.task]
        if args.ulysses_size > 1:
            if cfg.num_heads % args.ulysses_size != 0:
                raise SchemaValidateError(f"`num_heads` must be divisible by `ulysses_size`.")
        logging.info(f"Generation job args: {args}")
        logging.info(f"Generation model config: {cfg}")

        logging.info("Creating WanT2V pipeline.")
        self.wan_t2v = wan.WanT2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
            t5_cpu=args.t5_cpu,
            use_vae_parallel=args.vae_parallel,
        )

        self.transformer = self.wan_t2v.model

    def _get_transformer(self):
        return self.transformer
