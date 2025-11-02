import ast
import asyncio
from datetime import timedelta
import gc
import importlib
import argparse
import math
import pathlib
import re
import os
import sys
import time
import random
from omegaconf import OmegaConf
import json
import toml
from multiprocessing import Value
from typing import Any, Dict, List, Optional
from tqdm import tqdm
import av
from einops import rearrange
from PIL import Image

import numpy as np
import torch
import torchvision
import huggingface_hub
import accelerate
from accelerate.utils import TorchDynamoPlugin, set_seed, DynamoBackend
from accelerate import (
    Accelerator, 
    InitProcessGroupKwargs, 
    DistributedDataParallelKwargs,
    PartialState
)   
from safetensors.torch import load_file as load_safetensors
import transformers
from diffusers.optimization import (
    SchedulerType as DiffusersSchedulerType,
    TYPE_TO_SCHEDULER_FUNCTION as DIFFUSERS_TYPE_TO_SCHEDULER_FUNCTION,
)
from transformers.optimization import SchedulerType, TYPE_TO_SCHEDULER_FUNCTION

from ovi.train.safetensors_utils import MemoryEfficientSafeOpen
from ovi.train.lr_schedulers import RexLR
from ovi.train import huggingface_utils, train_utils
from ovi.train.device_utils import synchronize_device
from ovi.train.image_utils import preprocess_image, resize_image_to_bucket
from ovi.train.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler

from ovi.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


SS_METADATA_KEY_BASE_MODEL_VERSION = "ss_base_model_version"
SS_METADATA_KEY_NETWORK_MODULE = "ss_network_module"
SS_METADATA_KEY_NETWORK_DIM = "ss_network_dim"
SS_METADATA_KEY_NETWORK_ALPHA = "ss_network_alpha"
SS_METADATA_KEY_NETWORK_ARGS = "ss_network_args"

SS_METADATA_MINIMUM_KEYS = [
    SS_METADATA_KEY_BASE_MODEL_VERSION,
    SS_METADATA_KEY_NETWORK_MODULE,
    SS_METADATA_KEY_NETWORK_DIM,
    SS_METADATA_KEY_NETWORK_ALPHA,
    SS_METADATA_KEY_NETWORK_ARGS,
]


def clean_memory_on_device(device: torch.device):
    r"""
    Clean memory on the specified device, will be called from training scripts.
    """
    gc.collect()

    # device may "cuda" or "cuda:0", so we need to check the type of device
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if device.type == "xpu":
        torch.xpu.empty_cache()
    if device.type == "mps":
        torch.mps.empty_cache()


class collator_class:
    def __init__(self, epoch, dataset):
        self.current_epoch = epoch
        self.dataset = dataset
    
    def __call__(self, examples):
        worker_info = torch.utils.data.get_worker_info()
        # worker_info is None for the main process
        if worker_info is None:
            dataset = worker_info.dataset
        else:
            dataset = self.dataset
        
        # set epoch for validation dataset
        dataset.set_current_epoch(self.current_epoch.value)
        return examples[0]    # batch size is always 1, so return the only element
    

def prepare_accelerator(cfgs) -> Accelerator:
    """
    DeepSpeed is not supported in this script currently.
    """
    if cfgs.logging_dir is None:
        logging_dir = None
    else:
        log_prefix = "" if cfgs.log_prefix is None else cfgs.log_prefix
        logging_dir = cfgs.logging_dir + "/" + log_prefix + time.strftime("%Y%m%d%H%M%S", time.localtime())

    if cfgs.log_with is None:
        if logging_dir is not None:
            log_with = "tensorboard"
        else:
            log_with = None
    else:
        log_with = cfgs.log_with
        if log_with in ["tensorboard", "all"]:
            if logging_dir is None:
                raise ValueError(
                    "logging_dir is required when log_with is tensorboard / logging_dir需要配置以使用Tensorboard"
                )
        if log_with in ["wandb", "all"]:
            try:
                import wandb
            except ImportError:
                raise ImportError("No wandb / 没有安装wandb，请使用`pip install wandb`安装")
            if logging_dir is not None:
                os.makedirs(logging_dir, exist_ok=True)
                os.environ["WANDB_DIR"] = logging_dir
            if cfgs.wandb_api_key is not None:
                wandb.login(key=cfgs.wandb_api_key)

    kwargs_handlers = [
        (
            InitProcessGroupKwargs(
                backend="gloo" if os.name == "nt" or not torch.cuda.is_available() else "nccl",
                init_method=(
                    "env://?use_libuv=False" if os.name == "nt" and Version(torch.__version__) >= Version("2.4.0") else None
                ),
                timeout=timedelta(minutes=cfgs.ddp_timeout) if cfgs.ddp_timeout else None,
            )
            if torch.cuda.device_count() > 1
            else None
        ),
        (
            DistributedDataParallelKwargs(
                gradient_as_bucket_view=cfgs.ddp_gradient_as_bucket_view, static_graph=cfgs.ddp_static_graph
            )
            if cfgs.ddp_gradient_as_bucket_view or cfgs.ddp_static_graph
            else None
        ),
    ]
    kwargs_handlers = [i for i in kwargs_handlers if i is not None]

    dynamo_plugin = None
    if cfgs.dynamo_backend.upper() != "NO":
        dynamo_plugin = TorchDynamoPlugin(
            backend=DynamoBackend(cfgs.dynamo_backend.upper()),
            mode=cfgs.dynamo_mode,
            fullgraph=cfgs.dynamo_fullgraph,
            dynamic=cfgs.dynamo_dynamic,
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=cfgs.gradient_accumulation_steps,
        mixed_precision=cfgs.mixed_precision if cfgs.mixed_precision else None,
        log_with=log_with,
        project_dir=logging_dir,
        dynamo_plugin=dynamo_plugin,
        kwargs_handlers=kwargs_handlers,
    )
    print("accelerator device:", accelerator.device)

    return accelerator


def line_to_prompt_dict(line: str) -> dict:
    # subset of gen_img_diffusers
    prompt_args = line.split(" --")
    prompt_dict = {}
    prompt_dict["prompt"] = prompt_args[0]

    for parg in prompt_args:
        try:
            m = re.match(r"w (\d+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["width"] = int(m.group(1))
                continue

            m = re.match(r"h (\d+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["height"] = int(m.group(1))
                continue

            m = re.match(r"f (\d+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["frame_count"] = int(m.group(1))
                continue

            m = re.match(r"d (\d+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["seed"] = int(m.group(1))
                continue

            m = re.match(r"s (\d+)", parg, re.IGNORECASE)
            if m:  # steps
                prompt_dict["sample_steps"] = max(1, min(1000, int(m.group(1))))
                continue

            m = re.match(r"g ([\d\.]+)", parg, re.IGNORECASE)
            if m:  # scale
                prompt_dict["guidance_scale"] = float(m.group(1))
                continue

            m = re.match(r"fs ([\d\.]+)", parg, re.IGNORECASE)
            if m:  # scale
                prompt_dict["discrete_flow_shift"] = float(m.group(1))
                continue

            m = re.match(r"l ([\d\.]+)", parg, re.IGNORECASE)
            if m:  # scale
                prompt_dict["cfg_scale"] = float(m.group(1))
                continue

            m = re.match(r"n (.+)", parg, re.IGNORECASE)
            if m:  # negative prompt
                prompt_dict["negative_prompt"] = m.group(1)
                continue

            m = re.match(r"i (.+)", parg, re.IGNORECASE)
            if m:  # image path
                prompt_dict["image_path"] = m.group(1).strip()
                continue

            m = re.match(r"ei (.+)", parg, re.IGNORECASE)
            if m:  # end image path
                prompt_dict["end_image_path"] = m.group(1).strip()
                continue

            m = re.match(r"cn (.+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["control_video_path"] = m.group(1).strip()
                continue

            m = re.match(r"ci (.+)", parg, re.IGNORECASE)
            if m:
                # can be multiple control images
                control_image_path = m.group(1).strip()
                if "control_image_path" not in prompt_dict:
                    prompt_dict["control_image_path"] = []
                prompt_dict["control_image_path"].append(control_image_path)
                continue

            m = re.match(r"of (.+)", parg, re.IGNORECASE)
            if m:  # output folder
                prompt_dict["one_frame"] = m.group(1).strip()
                continue

        except ValueError as ex:
            logger.error(f"Exception in parsing / 解析失败: {parg}")
            logger.error(ex)

    return prompt_dict


def load_prompts(prompt_file: str) -> list[Dict]:
    # read prompts
    if prompt_file.endswith(".txt"):
        with open(prompt_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        prompts = [line.strip() for line in lines if len(line.strip()) > 0 and line[0] != "#"]
    elif prompt_file.endswith(".toml"):
        with open(prompt_file, "r", encoding="utf-8") as f:
            data = toml.load(f)
        prompts = [dict(**data["prompt"], **subset) for subset in data["prompt"]["subset"]]
    elif prompt_file.endswith(".json"):
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompts = json.load(f)

    # preprocess prompts
    for i in range(len(prompts)):
        prompt_dict = prompts[i]
        if isinstance(prompt_dict, str):
            prompt_dict = line_to_prompt_dict(prompt_dict)
            prompts[i] = prompt_dict
        assert isinstance(prompt_dict, dict)

        # Adds an enumerator to the dict based on prompt position. Used later to name image files. Also cleanup of extra data in original prompt dict.
        prompt_dict["enum"] = i
        prompt_dict.pop("subset", None)
    
    return prompts


def compute_density_for_timestep_sampling(
    weighting_scheme: str, batch_size: int, logit_mean: float = None, logit_std: float = None, mode_scale: float = None
):
    """Compute the density for sampling the timesteps when doing SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "logit_normal":
        # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu")
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), device="cpu")
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        u = torch.rand(size=(batch_size,), device="cpu")
    return u


def get_sigmas(noise_scheduler, timesteps, device, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)

    # if sum([(schedule_timesteps == t) for t in timesteps]) < len(timesteps):
    if any([(schedule_timesteps == t).sum() == 0 for t in timesteps]):
        # round to nearest timestep
        logger.warning("Some timesteps are not in the schedule / 一部分timesteps不在调度表中，正在取最近的时间步")
        step_indices = [torch.argmin(torch.abs(schedule_timesteps - t)).item() for t in timesteps]
    else:
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def compute_loss_weighting_for_sd3(weighting_scheme: str, noise_scheduler, timesteps, device, dtype):
    """Computes loss weighting scheme for SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "sigma_sqrt" or weighting_scheme == "cosmap":
        sigmas = get_sigmas(noise_scheduler, timesteps, device, n_dim=5, dtype=dtype)
        if weighting_scheme == "sigma_sqrt":
            weighting = (sigmas**-2.0).float()
        else:
            bot = 1 - 2 * sigmas + 2 * sigmas**2
            weighting = 2 / (math.pi * bot)
    else:
        weighting = None  # torch.ones_like(sigmas)
    return weighting


def should_sample_images(cfgs, steps, epoch=None):
    if steps == 0:
        if not cfgs.sample_at_first:
            return False
    else:
        should_sample_by_steps = cfgs.sample_every_n_steps is not None and steps % cfgs.sample_every_n_steps == 0
        should_sample_by_epochs = (
            cfgs.sample_every_n_epochs is not None and epoch is not None and epoch % cfgs.sample_every_n_epochs == 0
        )
        if not should_sample_by_steps and not should_sample_by_epochs:
            return False
    return True


def detect_dit_dtype(path: str) -> torch.dtype:
    # get dtype from model weights
    with MemoryEfficientSafeOpen(path) as f:
        keys = set(f.keys())
        key1 = "model.diffusion_model.blocks.0.cross_attn.k.weight"  # 1.3B
        key2 = "blocks.0.cross_attn.k.weight"  # 14B
        if key1 in keys:
            dit_dtype = f.get_tensor(key1).dtype
        elif key2 in keys:
            dit_dtype = f.get_tensor(key2).dtype
        else:
            raise ValueError(f"Could not find the dtype in the model weights: {path}")
    logger.info(f"Detected DiT dtype: {dit_dtype}")
    return dit_dtype


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=1, fps=24):
    """save videos by video tensor
       copy from https://github.com/guoyww/AnimateDiff/blob/e92bd5671ba62c0d774a32951453e328018b7c5b/animatediff/utils/util.py#L61

    Args:
        videos (torch.Tensor): video tensor predicted by the model
        path (str): path to save video
        rescale (bool, optional): rescale the video tensor from [-1, 1] to  . Defaults to False.
        n_rows (int, optional): Defaults to 1.
        fps (int, optional): video save fps. Defaults to 8.
    """
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = torch.clamp(x, 0, 1)
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    # # save video with av
    # container = av.open(path, "w")
    # stream = container.add_stream("libx264", rate=fps)
    # for x in outputs:
    #     frame = av.VideoFrame.from_ndarray(x, format="rgb24")
    #     packet = stream.encode(frame)
    #     container.mux(packet)
    # packet = stream.encode(None)
    # container.mux(packet)
    # container.close()

    height, width, _ = outputs[0].shape

    # create output container
    container = av.open(path, mode="w")

    # create video stream
    codec = "libx264"
    pixel_format = "yuv420p"
    stream = container.add_stream(codec, rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = pixel_format
    stream.bit_rate = 4000000  # 4Mbit/s

    for frame_array in outputs:
        frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
        packets = stream.encode(frame)
        for packet in packets:
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)

    container.close()


def save_images_grid(
    videos: torch.Tensor, parent_dir: str, image_name: str, rescale: bool = False, n_rows: int = 1, create_subdir=True
) -> list[str]:
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = torch.clamp(x, 0, 1)
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    if create_subdir:
        output_dir = os.path.join(parent_dir, image_name)
    else:
        output_dir = parent_dir

    os.makedirs(output_dir, exist_ok=True)
    image_paths = []
    for i, x in enumerate(outputs):
        image_path = os.path.join(output_dir, f"{image_name}_{i:03d}.png")
        image_paths.append(image_path)
        image = Image.fromarray(x)
        image.save(image_path)

    return image_paths


class Trainer:
    def __init__(
        self,
        config: dict,
    ):
        self.blocks_to_swap = None
        self.timestep_range_pool = []
        self.num_timestep_buckets: Optional[int] = None

        # Initialize trainer with configuration
        self.config = config
        self.seed = config.get("seed", 42)
        self.optim_cfg = config.get("optimizer", {})
        self.model_cfg = config.get("model", {})
        self.data_cfg = config.get("data", {})
        self.train_cfg = config.get("train", {})
        self.eval_cfg = config.get("eval", {})
        self.log_cfg = config.get("log", {})


        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator or Accelerator()

        # Prepare everything with the accelerator
        (
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.eval_dataloader,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.eval_dataloader,
            self.lr_scheduler,
        )

    def get_optimizer(self, model: torch.nn.Module):
        # adamw, adamw8bit, adafactor
        trainable_params, lr_descriptions = model.prepare_optimizer_params(dit_lr=self.optim_cfg.learning_rate)
        optimizer_type = self.optim_cfg.optimizer_type.lower()
        
        # split optimizer_type and optimizer_args
        optimizer_kwargs = {}
        if self.optim_cfg.optimizer_args is not None and len(self.optim_cfg.optimizer_args) > 0:
            for arg in self.optim_cfg.optimizer_args:
                key, value = arg.split("=")
                value = ast.literal_eval(value)
                optimizer_kwargs[key] = value

        lr = self.optim_cfg.learning_rate
        optimizer = None
        optimizer_class = None

        if optimizer_type.endswith("8bit".lower()):
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError("No bitsandbytes / 缺少bitsandbytes，请使用`pip install bitsandbytes`安装")

            if optimizer_type == "AdamW8bit".lower():
                logger.info(f"use 8-bit AdamW optimizer | {optimizer_kwargs}")
                optimizer_class = bnb.optim.AdamW8bit
                optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

        elif optimizer_type == "Adafactor".lower():
            # Adafactor: check relative_step and warmup_init
            if "relative_step" not in optimizer_kwargs:
                optimizer_kwargs["relative_step"] = True  # default
            if not optimizer_kwargs["relative_step"] and optimizer_kwargs.get("warmup_init", False):
                logger.info(
                    "set relative_step to True because warmup_init is True / warmup_init设置为True，因此relative_step也设置为True"
                )
                optimizer_kwargs["relative_step"] = True
            logger.info(f"use Adafactor optimizer | {optimizer_kwargs}")

            if optimizer_kwargs["relative_step"]:
                logger.info("relative_step is true / relative_stepがtrueです")
                if lr != 0.0:
                    logger.warning("learning rate is used as initial_lr / learning rate被当做initial_lr使用")
                self.optim_cfg.learning_rate = None

                if self.optim_cfg.lr_scheduler != "adafactor":
                    logger.info("use adafactor_scheduler / 使用adafactor_scheduler")
                self.optim_cfg.lr_scheduler = f"adafactor:{lr}"  # ちょっと微妙だけど

                lr = None
            else:
                if self.optim_cfg.max_grad_norm != 0.0:
                    logger.warning(
                        "because max_grad_norm is set, clip_grad_norm is enabled. consider set to 0 / max_grad_norm被设定，启用clip_grad_norm。考虑设置为0"
                    )
                if self.optim_cfg.lr_scheduler != "constant_with_warmup":
                    logger.warning("constant_with_warmup will be good / 使用constant_with_warmup会更好")
                if optimizer_kwargs.get("clip_threshold", 1.0) != 1.0:
                    logger.warning("clip_threshold=1.0 will be good / 设置clip_threshold为1.0会更好")

            optimizer_class = transformers.optimization.Adafactor
            optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

        elif optimizer_type == "AdamW".lower():
            logger.info(f"use AdamW optimizer | {optimizer_kwargs}")
            optimizer_class = torch.optim.AdamW
            optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

        if optimizer is None:
            # 其他任意自定义optimizer
            case_sensitive_optimizer_type = self.optim_cfg.optimizer_type  # not lower
            logger.info(f"use {case_sensitive_optimizer_type} | {optimizer_kwargs}")

            if "." not in case_sensitive_optimizer_type:  # from torch.optim
                optimizer_module = torch.optim
            else:  # from other library
                values = case_sensitive_optimizer_type.split(".")
                optimizer_module = importlib.import_module(".".join(values[:-1]))
                case_sensitive_optimizer_type = values[-1]

            optimizer_class = getattr(optimizer_module, case_sensitive_optimizer_type)
            optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

        # for logging
        optimizer_name = optimizer_class.__module__ + "." + optimizer_class.__name__
        optimizer_args = ",".join([f"{k}={v}" for k, v in optimizer_kwargs.items()])

        # get train and eval functions
        if hasattr(optimizer, "train") and callable(optimizer.train):
            train_fn = optimizer.train
            eval_fn = optimizer.eval
        else:
            train_fn = lambda: None
            eval_fn = lambda: None

        return lr_descriptions, optimizer_name, optimizer_args, optimizer, train_fn, eval_fn
    
    def is_schedulefree_optimizer(self, optimizer: torch.optim.Optimizer) -> bool:
        return self.optim_cfg.optimizer_type.lower().endswith("schedulefree".lower())  # or self.optim_cfg.optimizer_schedulefree_wrapper
    
    def get_dummy_scheduler(self, optimizer: torch.optim.Optimizer) -> Any:
        # dummy scheduler for schedulefree optimizer. supports only empty step(), get_last_lr() and optimizers.
        # this scheduler is used for logging only.
        # this isn't be wrapped by accelerator because of this class is not a subclass of torch.optim.lr_scheduler._LRScheduler
        class DummyScheduler:
            def __init__(self, optimizer: torch.optim.Optimizer):
                self.optimizer = optimizer

            def step(self):
                pass

            def get_last_lr(self):
                return [group["lr"] for group in self.optimizer.param_groups]

        return DummyScheduler(optimizer)
    
    def get_lr_scheduler(self, optimizer: torch.optim.Optimizer, num_processes: int):
        """
        Unified API to get any scheduler from its name.
        """
        # if schedulefree optimizer, return dummy scheduler
        if self.is_schedulefree_optimizer(optimizer):
            return self.get_dummy_scheduler(optimizer)

        name = self.optim_cfg.lr_scheduler
        num_training_steps = self.optim_cfg.max_train_steps * num_processes  # * self.optim_cfg.gradient_accumulation_steps
        num_warmup_steps: Optional[int] = (
            int(self.optim_cfg.lr_warmup_steps * num_training_steps) if isinstance(self.optim_cfg.lr_warmup_steps, float) else self.optim_cfg.lr_warmup_steps
        )
        num_decay_steps: Optional[int] = (
            int(self.optim_cfg.lr_decay_steps * num_training_steps) if isinstance(self.optim_cfg.lr_decay_steps, float) else self.optim_cfg.lr_decay_steps
        )
        num_stable_steps = num_training_steps - num_warmup_steps - num_decay_steps
        num_cycles = self.optim_cfg.lr_scheduler_num_cycles
        power = self.optim_cfg.lr_scheduler_power
        timescale = self.optim_cfg.lr_scheduler_timescale
        min_lr_ratio = self.optim_cfg.lr_scheduler_min_lr_ratio

        lr_scheduler_kwargs = {}  # get custom lr_scheduler kwargs
        if self.optim_cfg.lr_scheduler_args is not None and len(self.optim_cfg.lr_scheduler_args) > 0:
            for arg in self.optim_cfg.lr_scheduler_args:
                key, value = arg.split("=")
                value = ast.literal_eval(value)
                lr_scheduler_kwargs[key] = value

        def wrap_check_needless_num_warmup_steps(return_vals):
            if num_warmup_steps is not None and num_warmup_steps != 0:
                raise ValueError(f"{name} does not require `num_warmup_steps`. Set None or 0.")
            return return_vals

        # using any lr_scheduler from other library
        if self.optim_cfg.lr_scheduler_type:
            lr_scheduler_type = self.optim_cfg.lr_scheduler_type
            logger.info(f"use {lr_scheduler_type} | {lr_scheduler_kwargs} as lr_scheduler")
            if "." not in lr_scheduler_type:  # default to use torch.optim
                lr_scheduler_module = torch.optim.lr_scheduler
            else:
                values = lr_scheduler_type.split(".")
                lr_scheduler_module = importlib.import_module(".".join(values[:-1]))
                lr_scheduler_type = values[-1]
            lr_scheduler_class = getattr(lr_scheduler_module, lr_scheduler_type)
            lr_scheduler = lr_scheduler_class(optimizer, **lr_scheduler_kwargs)
            return lr_scheduler

        if name.startswith("adafactor"):
            assert type(optimizer) == transformers.optimization.Adafactor, (
                "adafactor scheduler must be used with Adafactor optimizer / adafactor调度器必须与Adafactor优化器一起使用"
            )
            initial_lr = float(name.split(":")[1])
            # logger.info(f"adafactor scheduler init lr {initial_lr}")
            return wrap_check_needless_num_warmup_steps(transformers.optimization.AdafactorSchedule(optimizer, initial_lr))

        if name.lower() == "rex":
            return RexLR(
                optimizer,
                max_lr=self.optim_cfg.learning_rate,
                min_lr=(  # Will start and end with min_lr, use non-zero min_lr by default
                    self.optim_cfg.learning_rate * min_lr_ratio if min_lr_ratio is not None else self.optim_cfg.learning_rate * 0.01
                ),
                num_steps=num_training_steps,
                num_warmup_steps=num_warmup_steps,
                **lr_scheduler_kwargs,
            )

        if name == DiffusersSchedulerType.PIECEWISE_CONSTANT.value:
            name = DiffusersSchedulerType(name)
            schedule_func = DIFFUSERS_TYPE_TO_SCHEDULER_FUNCTION[name]
            return schedule_func(optimizer, **lr_scheduler_kwargs)  # step_rules and last_epoch are given as kwargs

        name = SchedulerType(name)
        schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]

        if name == SchedulerType.CONSTANT:
            return wrap_check_needless_num_warmup_steps(schedule_func(optimizer, **lr_scheduler_kwargs))

        # All other schedulers require `num_warmup_steps`
        if num_warmup_steps is None:
            raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

        if name == SchedulerType.CONSTANT_WITH_WARMUP:
            return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, **lr_scheduler_kwargs)

        if name == SchedulerType.INVERSE_SQRT:
            return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, timescale=timescale, **lr_scheduler_kwargs)

        # All other schedulers require `num_training_steps`
        if num_training_steps is None:
            raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

        if name == SchedulerType.COSINE_WITH_RESTARTS:
            return schedule_func(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                num_cycles=num_cycles,
                **lr_scheduler_kwargs,
            )

        if name == SchedulerType.POLYNOMIAL:
            return schedule_func(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                power=power,
                **lr_scheduler_kwargs,
            )

        if name == SchedulerType.COSINE_WITH_MIN_LR:
            return schedule_func(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                num_cycles=num_cycles / 2,
                min_lr_rate=min_lr_ratio,
                **lr_scheduler_kwargs,
            )

        # these schedulers do not require `num_decay_steps`
        if name == SchedulerType.LINEAR or name == SchedulerType.COSINE:
            return schedule_func(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                **lr_scheduler_kwargs,
            )

        # All other schedulers require `num_decay_steps`
        if num_decay_steps is None:
            raise ValueError(f"{name} requires `num_decay_steps`, please provide that argument.")
        if name == SchedulerType.WARMUP_STABLE_DECAY:
            return schedule_func(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_stable_steps=num_stable_steps,
                num_decay_steps=num_decay_steps,
                num_cycles=num_cycles / 2,
                min_lr_ratio=min_lr_ratio if min_lr_ratio is not None else 0.0,
                **lr_scheduler_kwargs,
            )

        return schedule_func(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_decay_steps=num_decay_steps,
            **lr_scheduler_kwargs,
        )

    def resume_from_local_or_hf_if_specified(self, accelerator: Accelerator) -> bool:
        if not self.train_cfg.resume:
            return False

        if not self.train_cfg.resume_from_huggingface:
            logger.info(f"resume training from local state: {self.train_cfg.resume}")
            accelerator.load_state(self.train_cfg.resume)
            return True

        logger.info(f"resume training from huggingface state: {self.train_cfg.resume}")
        repo_id = self.train_cfg.resume.split("/")[0] + "/" + self.train_cfg.resume.split("/")[1]
        path_in_repo = "/".join(self.train_cfg.resume.split("/")[2:])
        revision = None
        repo_type = None
        if ":" in path_in_repo:
            divided = path_in_repo.split(":")
            if len(divided) == 2:
                path_in_repo, revision = divided
                repo_type = "model"
            else:
                path_in_repo, revision, repo_type = divided
        logger.info(f"Downloading state from huggingface: {repo_id}/{path_in_repo}@{revision}")

        list_files = huggingface_utils.list_dir(
            repo_id=repo_id,
            subfolder=path_in_repo,
            revision=revision,
            token=self.train_cfg.huggingface_token,
            repo_type=repo_type,
        )

        async def download(filename) -> str:
            def task():
                return huggingface_hub.hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    revision=revision,
                    repo_type=repo_type,
                    token=self.train_cfg.huggingface_token,
                )

            return await asyncio.get_event_loop().run_in_executor(None, task)

        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(asyncio.gather(*[download(filename=filename.rfilename) for filename in list_files]))
        if len(results) == 0:
            raise ValueError(
                "No files found in the specified repo id/path/revision / 指定されたリポジトリID/パス/リビジョンにファイルが見つかりませんでした"
            )
        dirname = os.path.dirname(results[0])
        accelerator.load_state(dirname)

        return True

    def generate_step_logs(
        self,
        step_loss,
        avg_loss,
        lr_scheduler,
        lr_descriptions,
        optimizer = None,
        keys_scaled = None,
        mean_norm = None,
        maximum_norm = None,
    ):
        network_train_dit_only = True
        logs = {"loss/step": step_loss, "loss/average": avg_loss}

        if keys_scaled is not None:
            logs["max_norm/keys_scaled"] = keys_scaled
            logs["max_norm/average_key_norm"] = mean_norm
            logs["max_norm/max_key_norm"] = maximum_norm
        
        lrs = lr_scheduler.get_last_lr()

        for i, lr in enumerate(lrs):
            if lr_descriptions is not None:
                lr_desc = lr_descriptions[i]
            else:
                idx = i - (0 if network_train_dit_only else -1)
                if idx == -1:
                    lr_desc = "textencoder"
                else:
                    if len(lrs) > 2:
                        lr_desc = f"group{idx}"
                    else:
                        lr_desc = "unet"

            logs[f"lr/{lr_desc}"] = lr

            if self.optim_cfg.optimizer_type.lower().startswith("DAdapt".lower()) or self.optim_cfg.optimizer_type.lower().endswith("Prodigy".lower()):
                # tracking d*lr value
                logs[f"lr/d*lr/{lr_desc}"] = (
                    lr_scheduler.optimizers[-1].param_groups[i]["d"] * lr_scheduler.optimizers[-1].param_groups[i]["lr"]
                )
            if (
                self.optim_cfg.optimizer_type.lower().endswith("ProdigyPlusScheduleFree".lower()) and optimizer is not None
            ):  # tracking d*lr value of unet.
                logs["lr/d*lr"] = optimizer.param_groups[0]["d"] * optimizer.param_groups[0]["lr"]
        else:
            idx = 0
            if not network_train_dit_only:
                logs["lr/textencoder"] = float(lrs[0])
                idx = 1

            for i in range(idx, len(lrs)):
                logs[f"lr/group{i}"] = float(lrs[i])
                if self.optim_cfg.optimizer_type.lower().startswith("DAdapt".lower()) or self.optim_cfg.optimizer_type.lower().endswith(
                    "Prodigy".lower()
                ):
                    logs[f"lr/d*lr/group{i}"] = (
                        lr_scheduler.optimizers[-1].param_groups[i]["d"] * lr_scheduler.optimizers[-1].param_groups[i]["lr"]
                    )
                if self.optim_cfg.optimizer_type.lower().endswith("ProdigyPlusScheduleFree".lower()) and optimizer is not None:
                    logs[f"lr/d*lr/group{i}"] = optimizer.param_groups[i]["d"] * optimizer.param_groups[i]["lr"]

        return logs

    def get_bucketed_timestep(self) -> float:
        if self.num_timestep_buckets is None or self.num_timestep_buckets <= 1:
            return random.random()

        if len(self.timestep_range_pool) == 0:
            bucket_size = 1.0 / self.num_timestep_buckets
            for i in range(self.num_timestep_buckets):
                self.timestep_range_pool.append((i * bucket_size, (i + 1) * bucket_size))
            random.shuffle(self.timestep_range_pool)

        # print(f"timestep_range_pool: {self.timestep_range_pool}")
        a, b = self.timestep_range_pool.pop()
        return random.uniform(a, b)

    def show_timesteps(self):
        N_TRY = 100000
        BATCH_SIZE = 1000
        CONSOLE_WIDTH = 64
        N_TIMESTEPS_PER_LINE = 25

        noise_scheduler = FlowMatchDiscreteScheduler(shift=self.train_cfg.discrete_flow_shift, reverse=True, solver="euler")
        # print(f"Noise scheduler timesteps: {noise_scheduler.timesteps}")

        latents = torch.zeros(BATCH_SIZE, 1, 1, 1024 // 8, 1024 // 8, dtype=torch.float16)
        noise = torch.ones_like(latents)

        # sample timesteps
        sampled_timesteps = [0] * noise_scheduler.config.num_train_timesteps
        for i in tqdm(range(N_TRY // BATCH_SIZE)):
            bucketed_timesteps = None
            if self.train_cfg.num_timestep_buckets is not None and self.train_cfg.num_timestep_buckets > 1:
                self.num_timestep_buckets = self.train_cfg.num_timestep_buckets
                bucketed_timesteps = [self.get_bucketed_timestep() for _ in range(BATCH_SIZE)]

            # we use noise=1, so retured noisy_model_input is same as timestep, because `noisy_model_input = (1 - t) * latents + t * noise`
            actual_timesteps, _ = self.get_noisy_model_input_and_timesteps(
                self.train_cfg, noise, latents, bucketed_timesteps, noise_scheduler, "cpu", torch.float16
            )
            actual_timesteps = actual_timesteps[:, 0, 0, 0, 0] * 1000
            for t in actual_timesteps:
                t = int(t.item())
                sampled_timesteps[t] += 1

        # sample weighting
        sampled_weighting = [0] * noise_scheduler.config.num_train_timesteps
        for i in tqdm(range(len(sampled_weighting))):
            timesteps = torch.tensor([i + 1], device="cpu")
            weighting = compute_loss_weighting_for_sd3(self.train_cfg.weighting_scheme, noise_scheduler, timesteps, "cpu", torch.float16)
            if weighting is None:
                weighting = torch.tensor(1.0, device="cpu")
            elif torch.isinf(weighting).any():
                weighting = torch.tensor(1.0, device="cpu")
            sampled_weighting[i] = weighting.item()

        # show results
        if self.train_cfg.show_timesteps == "image":
            # show timesteps with matplotlib
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.bar(range(len(sampled_timesteps)), sampled_timesteps, width=1.0)
            plt.title("Sampled timesteps")
            plt.xlabel("Timestep")
            plt.ylabel("Count")

            plt.subplot(1, 2, 2)
            plt.bar(range(len(sampled_weighting)), sampled_weighting, width=1.0)
            plt.title("Sampled loss weighting")
            plt.xlabel("Timestep")
            plt.ylabel("Weighting")

            plt.tight_layout()
            plt.show()

        else:
            sampled_timesteps = np.array(sampled_timesteps)
            sampled_weighting = np.array(sampled_weighting)

            # average per line
            sampled_timesteps = sampled_timesteps.reshape(-1, N_TIMESTEPS_PER_LINE).mean(axis=1)
            sampled_weighting = sampled_weighting.reshape(-1, N_TIMESTEPS_PER_LINE).mean(axis=1)

            max_count = max(sampled_timesteps)
            print(f"Sampled timesteps: max count={max_count}")
            for i, t in enumerate(sampled_timesteps):
                line = f"{(i) * N_TIMESTEPS_PER_LINE:4d}-{(i + 1) * N_TIMESTEPS_PER_LINE - 1:4d}: "
                line += "#" * int(t / max_count * CONSOLE_WIDTH)
                print(line)

            max_weighting = max(sampled_weighting)
            print(f"Sampled loss weighting: max weighting={max_weighting}")
            for i, w in enumerate(sampled_weighting):
                line = f"{i * N_TIMESTEPS_PER_LINE:4d}-{(i + 1) * N_TIMESTEPS_PER_LINE - 1:4d}: {w:8.2f} "
                line += "#" * int(w / max_weighting * CONSOLE_WIDTH)
                print(line)

    def default_get_noisy_model_input_and_timesteps(
        self,
        args: argparse.Namespace,
        noise: torch.Tensor,
        latents: torch.Tensor,
        timesteps: Optional[List[float]],
        noise_scheduler: FlowMatchDiscreteScheduler,
        device: torch.device,
        dtype: torch.dtype,
    ):
        batch_size = noise.shape[0]

        if timesteps is not None:
            timesteps = torch.tensor(timesteps, device=device)

        # This function converts uniform distribution samples to logistic distribution samples.
        # The final distribution of the samples after shifting significantly differs from the original normal distribution.
        # So we cannot use this.
        # def uniform_to_normal(t_samples: torch.Tensor) -> torch.Tensor:
        #     # Clip small values to prevent log(0)
        #     eps = 1e-7
        #     t_samples = torch.clamp(t_samples, eps, 1.0 - eps)
        #     # Convert to logit space with inverse function
        #     x_samples = torch.log(t_samples / (1.0 - t_samples))
        #     return x_samples

        def uniform_to_normal_ppF(t_uniform: torch.Tensor) -> torch.Tensor:
            """Use `torch.erfinv` to compute the inverse CDF to generate values from a normal distribution."""
            # Clip small values to prevent inf in erfinv
            eps = 1e-7
            t_uniform = torch.clamp(t_uniform, eps, 1.0 - eps)

            # PPF of standard normal distribution: sqrt(2) * erfinv(2q - 1)
            term = 2.0 * t_uniform - 1.0
            x_normal = math.sqrt(2.0) * torch.erfinv(term)
            return x_normal

        def uniform_to_logsnr_ppF_pytorch(t_uniform: torch.Tensor, mean: float, std: float) -> torch.Tensor:
            """Use erfinv to compute the inverse CDF."""
            # Clip small values to prevent inf in erfinv
            eps = 1e-7
            t_uniform = torch.clamp(t_uniform, eps, 1.0 - eps)

            term = 2.0 * t_uniform - 1.0
            logsnr = mean + std * math.sqrt(2.0) * torch.erfinv(term)
            return logsnr

        if (
            args.timestep_sampling == "uniform"
            or args.timestep_sampling == "sigmoid"
            or args.timestep_sampling == "shift"
            or args.timestep_sampling == "flux_shift"
            or args.timestep_sampling == "qwen_shift"
            or args.timestep_sampling == "logsnr"
            or args.timestep_sampling == "qinglong_flux"
            or args.timestep_sampling == "qinglong_qwen"
        ):

            def compute_sampling_timesteps(org_timesteps: Optional[torch.Tensor]) -> torch.Tensor:
                def rand(bs: int, org_ts: Optional[torch.Tensor] = None) -> torch.Tensor:
                    nonlocal device
                    return torch.rand((bs,), device=device) if org_ts is None else org_ts

                def randn(bs: int, org_ts: Optional[torch.Tensor] = None) -> torch.Tensor:
                    nonlocal device
                    return uniform_to_normal_ppF(org_ts) if org_ts is not None else torch.randn((bs,), device=device)

                def rand_logsnr(bs: int, mean: float, std: float, org_ts: Optional[torch.Tensor] = None) -> torch.Tensor:
                    nonlocal device
                    logsnr = (
                        uniform_to_logsnr_ppF_pytorch(org_ts, mean, std)
                        if org_ts is not None
                        else torch.normal(mean=mean, std=std, size=(bs,), device=device)
                    )
                    return logsnr

                if args.timestep_sampling == "uniform" or args.timestep_sampling == "sigmoid":
                    # Simple random t-based noise sampling
                    if args.timestep_sampling == "sigmoid":
                        t = torch.sigmoid(args.sigmoid_scale * randn(batch_size, org_timesteps))
                    else:
                        t = rand(batch_size, org_timesteps)

                elif args.timestep_sampling.endswith("shift"):
                    if args.timestep_sampling == "shift":
                        shift = args.discrete_flow_shift
                    else:
                        h, w = latents.shape[-2:]
                        # we are pre-packed so must adjust for packed size
                        if args.timestep_sampling == "flux_shift":
                            mu = train_utils.get_lin_function(y1=0.5, y2=1.15)((h // 2) * (w // 2))
                        elif args.timestep_sampling == "qwen_shift":
                            mu = train_utils.get_lin_function(x1=256, y1=0.5, x2=8192, y2=0.9)((h // 2) * (w // 2))
                        # def time_shift(mu: float, sigma: float, t: torch.Tensor):
                        #     return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma) # sigma=1.0
                        shift = math.exp(mu)

                    logits_norm = randn(batch_size, org_timesteps)
                    logits_norm = logits_norm * args.sigmoid_scale  # larger scale for more uniform sampling
                    t = logits_norm.sigmoid()
                    t = (t * shift) / (1 + (shift - 1) * t)

                elif args.timestep_sampling == "logsnr":
                    # https://arxiv.org/abs/2411.14793v3
                    logsnr = rand_logsnr(batch_size, args.logit_mean, args.logit_std, org_timesteps)
                    t = torch.sigmoid(-logsnr / 2)

                elif args.timestep_sampling.startswith("qinglong"):
                    # Qinglong triple hybrid sampling: mid_shift:logsnr:logsnr2 = .80:.075:.125
                    # First decide which method to use for each sample independently
                    decision_t = torch.rand((batch_size,), device=device)

                    # Create masks based on decision_t: .80 for mid_shift, 0.075 for logsnr, and 0.125 for logsnr2
                    mid_mask = decision_t < 0.80  # 80% for mid_shift
                    logsnr_mask = (decision_t >= 0.80) & (decision_t < 0.875)  # 7.5% for logsnr
                    logsnr_mask2 = decision_t >= 0.875  # 12.5% for logsnr with -logit_mean

                    # Initialize output tensor
                    t = torch.zeros((batch_size,), device=device)

                    # Generate mid_shift samples for selected indices (80%)
                    if mid_mask.any():
                        mid_count = mid_mask.sum().item()
                        h, w = latents.shape[-2:]
                        if args.timestep_sampling == "qinglong_flux":
                            mu = train_utils.get_lin_function(y1=0.5, y2=1.15)((h // 2) * (w // 2))
                        elif args.timestep_sampling == "qinglong_qwen":
                            mu = train_utils.get_lin_function(x1=256, y1=0.5, x2=8192, y2=0.9)((h // 2) * (w // 2))
                        shift = math.exp(mu)
                        logits_norm_mid = randn(mid_count, org_timesteps[mid_mask] if org_timesteps is not None else None)
                        logits_norm_mid = logits_norm_mid * args.sigmoid_scale
                        t_mid = logits_norm_mid.sigmoid()
                        t_mid = (t_mid * shift) / (1 + (shift - 1) * t_mid)

                        t[mid_mask] = t_mid

                    # Generate logsnr samples for selected indices (7.5%)
                    if logsnr_mask.any():
                        logsnr_count = logsnr_mask.sum().item()
                        logsnr = rand_logsnr(
                            logsnr_count,
                            args.logit_mean,
                            args.logit_std,
                            org_timesteps[logsnr_mask] if org_timesteps is not None else None,
                        )
                        t_logsnr = torch.sigmoid(-logsnr / 2)

                        t[logsnr_mask] = t_logsnr

                    # Generate logsnr2 samples with -logit_mean for selected indices (12.5%)
                    if logsnr_mask2.any():
                        logsnr2_count = logsnr_mask2.sum().item()
                        logsnr2 = rand_logsnr(
                            logsnr2_count, 5.36, 1.0, org_timesteps[logsnr_mask2] if org_timesteps is not None else None
                        )
                        t_logsnr2 = torch.sigmoid(-logsnr2 / 2)

                        t[logsnr_mask2] = t_logsnr2

                return t  # 0 to 1

            t_min = args.min_timestep if args.min_timestep is not None else 0
            t_max = args.max_timestep if args.max_timestep is not None else 1000.0
            t_min /= 1000.0
            t_max /= 1000.0

            if not args.preserve_distribution_shape:
                t = compute_sampling_timesteps(timesteps)
                t = t * (t_max - t_min) + t_min  # scale to [t_min, t_max], default [0, 1]
            else:
                max_loops = 1000
                available_t = []
                for i in range(max_loops):
                    t = None
                    if self.num_timestep_buckets is not None:
                        t = torch.tensor([self.get_bucketed_timestep() for _ in range(batch_size)], device=device)
                    t = compute_sampling_timesteps(t)
                    for t_i in t:
                        if t_min <= t_i <= t_max:
                            available_t.append(t_i)
                        if len(available_t) == batch_size:
                            break
                    if len(available_t) == batch_size:
                        break
                if len(available_t) < batch_size:
                    logger.warning(
                        f"Could not sample {batch_size} valid timesteps in {max_loops} loops / {max_loops}ループで{batch_size}個の有効なタイムステップをサンプリングできませんでした"
                    )
                    available_t = compute_sampling_timesteps(timesteps)
                else:
                    t = torch.stack(available_t, dim=0)  # [batch_size, ]

            timesteps = t * 1000.0
            t = t.view(-1, 1, 1, 1, 1) if latents.ndim == 5 else t.view(-1, 1, 1, 1)
            noisy_model_input = (1 - t) * latents + t * noise

            timesteps += 1  # 1 to 1000
        else:
            # Sample a random timestep for each image
            # for weighting schemes where we sample timesteps non-uniformly
            u = compute_density_for_timestep_sampling(
                weighting_scheme=args.weighting_scheme,
                batch_size=batch_size,
                logit_mean=args.logit_mean,
                logit_std=args.logit_std,
                mode_scale=args.mode_scale,
            )
            # indices = (u * noise_scheduler.config.num_train_timesteps).long()
            t_min = args.min_timestep if args.min_timestep is not None else 0
            t_max = args.max_timestep if args.max_timestep is not None else 1000
            indices = (u * (t_max - t_min) + t_min).long()

            timesteps = noise_scheduler.timesteps[indices].to(device=device)  # 1 to 1000

            # Add noise according to flow matching.
            sigmas = get_sigmas(noise_scheduler, timesteps, device, n_dim=latents.ndim, dtype=dtype)
            noisy_model_input = sigmas * noise + (1.0 - sigmas) * latents

        # print(f"actual timesteps: {timesteps}")




        return noisy_model_input, timesteps

    def get_noisy_model_input_and_timesteps(
        self,
        args: argparse.Namespace,
        noise: torch.Tensor,
        latents: torch.Tensor,
        timesteps: Optional[List[float]],
        noise_scheduler: FlowMatchDiscreteScheduler,
        device: torch.device,
        dtype: torch.dtype,
    ):
        if not self.high_low_training:
            return self.default_get_noisy_model_input_and_timesteps(args, noise, latents, timesteps, noise_scheduler, device, dtype)

        # high-low training case
        # call super to get the noisy model input and timesteps, and sample only the first one, and choose the model we want based on the timestep
        noisy_model_input, sample_timesteps = super().get_noisy_model_input_and_timesteps(
            args, noise[0:1], latents[0:1], timesteps[0:1] if timesteps is not None else None, noise_scheduler, device, dtype
        )
        high_noise = sample_timesteps[0] / 1000.0 >= self.timestep_boundary
        self.next_model_is_high_noise = high_noise

        # choose each member of latents for high or low noise model. because we want to train all the latents
        num_max_calls = 100
        final_noisy_model_inputs = []
        final_timesteps_list = []
        bsize = latents.shape[0]
        for i in range(bsize):
            for _ in range(num_max_calls):
                ts_i = [self.get_bucketed_timestep()] if self.num_timestep_buckets is not None else None

                noisy_model_input, ts_i = super().get_noisy_model_input_and_timesteps(
                    args, noise[i : i + 1], latents[i : i + 1], ts_i, noise_scheduler, device, dtype
                )
                if (high_noise and ts_i[0] / 1000.0 >= self.timestep_boundary) or (
                    not high_noise and ts_i[0] / 1000.0 < self.timestep_boundary
                ):
                    final_noisy_model_inputs.append(noisy_model_input)
                    final_timesteps_list.append(ts_i)
                    break

        if len(final_noisy_model_inputs) < bsize:
            logger.warning(
                f"No valid noisy model inputs found for bsize={bsize}, high_noise={high_noise}, timestep_boundary={self.timestep_boundary}"
            )
            # fall back to the original method
            return super().get_noisy_model_input_and_timesteps(args, noise, latents, timesteps, noise_scheduler, device, dtype)

        # final noisy model input may have less than bsize elements, it will be fine for training
        final_noisy_model_input = torch.cat(final_noisy_model_inputs, dim=0)
        final_timesteps = torch.cat(final_timesteps_list, dim=0)

        return final_noisy_model_input, final_timesteps

    def show_timesteps(self, args):
        N_TRY = 100000
        BATCH_SIZE = 1000
        CONSOLE_WIDTH = 64
        N_TIMESTEPS_PER_LINE = 25

        noise_scheduler = FlowMatchDiscreteScheduler(shift=args.discrete_flow_shift, reverse=True, solver="euler")
        # print(f"Noise scheduler timesteps: {noise_scheduler.timesteps}")

        latents = torch.zeros(BATCH_SIZE, 1, 1, 1024 // 8, 1024 // 8, dtype=torch.float16)
        noise = torch.ones_like(latents)

        # sample timesteps
        sampled_timesteps = [0] * noise_scheduler.config.num_train_timesteps
        for i in tqdm(range(N_TRY // BATCH_SIZE)):
            bucketed_timesteps = None
            if args.num_timestep_buckets is not None and args.num_timestep_buckets > 1:
                self.num_timestep_buckets = args.num_timestep_buckets
                bucketed_timesteps = [self.get_bucketed_timestep() for _ in range(BATCH_SIZE)]

            # we use noise=1, so retured noisy_model_input is same as timestep, because `noisy_model_input = (1 - t) * latents + t * noise`
            actual_timesteps, _ = self.get_noisy_model_input_and_timesteps(
                args, noise, latents, bucketed_timesteps, noise_scheduler, "cpu", torch.float16
            )
            actual_timesteps = actual_timesteps[:, 0, 0, 0, 0] * 1000
            for t in actual_timesteps:
                t = int(t.item())
                sampled_timesteps[t] += 1

        # sample weighting
        sampled_weighting = [0] * noise_scheduler.config.num_train_timesteps
        for i in tqdm(range(len(sampled_weighting))):
            timesteps = torch.tensor([i + 1], device="cpu")
            weighting = compute_loss_weighting_for_sd3(args.weighting_scheme, noise_scheduler, timesteps, "cpu", torch.float16)
            if weighting is None:
                weighting = torch.tensor(1.0, device="cpu")
            elif torch.isinf(weighting).any():
                weighting = torch.tensor(1.0, device="cpu")
            sampled_weighting[i] = weighting.item()

        # show results
        if args.show_timesteps == "image":
            # show timesteps with matplotlib
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.bar(range(len(sampled_timesteps)), sampled_timesteps, width=1.0)
            plt.title("Sampled timesteps")
            plt.xlabel("Timestep")
            plt.ylabel("Count")

            plt.subplot(1, 2, 2)
            plt.bar(range(len(sampled_weighting)), sampled_weighting, width=1.0)
            plt.title("Sampled loss weighting")
            plt.xlabel("Timestep")
            plt.ylabel("Weighting")

            plt.tight_layout()
            plt.show()

        else:
            sampled_timesteps = np.array(sampled_timesteps)
            sampled_weighting = np.array(sampled_weighting)

            # average per line
            sampled_timesteps = sampled_timesteps.reshape(-1, N_TIMESTEPS_PER_LINE).mean(axis=1)
            sampled_weighting = sampled_weighting.reshape(-1, N_TIMESTEPS_PER_LINE).mean(axis=1)

            max_count = max(sampled_timesteps)
            print(f"Sampled timesteps: max count={max_count}")
            for i, t in enumerate(sampled_timesteps):
                line = f"{(i) * N_TIMESTEPS_PER_LINE:4d}-{(i + 1) * N_TIMESTEPS_PER_LINE - 1:4d}: "
                line += "#" * int(t / max_count * CONSOLE_WIDTH)
                print(line)

            max_weighting = max(sampled_weighting)
            print(f"Sampled loss weighting: max weighting={max_weighting}")
            for i, w in enumerate(sampled_weighting):
                line = f"{i * N_TIMESTEPS_PER_LINE:4d}-{(i + 1) * N_TIMESTEPS_PER_LINE - 1:4d}: {w:8.2f} "
                line += "#" * int(w / max_weighting * CONSOLE_WIDTH)
                print(line)

    def init_env(self):
        # Initialize environment settings if needed
        pass
    
    def init_model(self, model_path: str):
        # Load model from the given path
        pass
    
    def load_vae(self, args: argparse.Namespace, vae_dtype: torch.dtype, vae_path: str):
        vae_path = args.vae

        logger.info(f"Loading VAE model from {vae_path}")
        cache_device = torch.device("cpu") if args.vae_cache_cpu else None
        vae = WanVAE(vae_path=vae_path, device="cpu", dtype=vae_dtype, cache_device=cache_device)
        return vae

    def load_transformer(
        self,
        accelerator: Accelerator,
        args: argparse.Namespace,
        dit_path: str,
        attn_mode: str,
        split_attn: bool,
        loading_device: str,
        dit_weight_dtype: Optional[torch.dtype],
    ):
        model = load_wan_model(
            self.config,
            accelerator.device,
            dit_path,
            attn_mode,
            split_attn,
            loading_device,
            dit_weight_dtype,
            args.fp8_scaled,
            disable_numpy_memmap=args.disable_numpy_memmap,
        )
        if args.force_v2_1_time_embedding:
            model.set_time_embedding_v2_1(True)

        if self.high_low_training:
            # load high noise model
            logger.info(f"Loading high noise model from {self.dit_high_noise_path}")
            model_high_noise = load_wan_model(
                self.config,
                accelerator.device,
                self.dit_high_noise_path,
                attn_mode,
                split_attn,
                "cpu" if args.offload_inactive_dit else loading_device,
                dit_weight_dtype,
                args.fp8_scaled,
                disable_numpy_memmap=args.disable_numpy_memmap,
            )
            if args.force_v2_1_time_embedding:
                model_high_noise.set_time_embedding_v2_1(True)
            if self.blocks_to_swap > 0:
                # This moves the weights to the appropriate device
                logger.info(f"Prepare block swap for high noise model, blocks_to_swap={self.blocks_to_swap}")
                model_high_noise.enable_block_swap(self.blocks_to_swap, accelerator.device, supports_backward=True)
                model_high_noise.move_to_device_except_swap_blocks(accelerator.device)
                model_high_noise.prepare_block_swap_before_forward()

            self.dit_inactive_state_dict = model_high_noise.state_dict()

            self.current_model_is_high_noise = False
            self.next_model_is_high_noise = False
        else:
            self.dit_inactive_state_dict = None
            self.current_model_is_high_noise = False
            self.next_model_is_high_noise = False

        return model

    def load_dataset(self, dataset_path: str):
        # Load dataset from the given path
        pass

    def handle_model_specific_args(self, args):
        self.config = WAN_CONFIGS[args.task]
        # we cannot use config.i2v because Fun-Control T2V has i2v flag TODO refactor this
        self._i2v_training = "i2v" in args.task or "flf2v" in args.task
        self._control_training = self.config.is_fun_control

        self.dit_dtype = detect_wan_sd_dtype(args.dit)

        if self.dit_dtype == torch.float16:
            assert args.mixed_precision in ["fp16", "no"], "DiT weights are in fp16, mixed precision must be fp16 or no"
        elif self.dit_dtype == torch.bfloat16:
            assert args.mixed_precision in ["bf16", "no"], "DiT weights are in bf16, mixed precision must be bf16 or no"

        if args.fp8_scaled and self.dit_dtype.itemsize == 1:
            raise ValueError(
                "DiT weights is already in fp8 format, cannot scale to fp8. Please use fp16/bf16 weights / DiTの重みはすでにfp8形式です。fp8にスケーリングできません。fp16/bf16の重みを使用してください"
            )

        # dit_dtype cannot be fp8, so we select the appropriate dtype
        if self.dit_dtype.itemsize == 1:
            self.dit_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16

        args.dit_dtype = model_utils.dtype_to_str(self.dit_dtype)

        # Wan2.2: Store timestep boundary
        self.dit_high_noise_path = args.dit_high_noise
        self.high_low_training = self.dit_high_noise_path is not None

        if self.high_low_training:
            if args.blocks_to_swap is not None and args.blocks_to_swap > 0:
                assert not args.offload_inactive_dit, (
                    "Block swap is not supported with offloading inactive DiT / 非アクティブDiTをオフロードする設定ではブロックスワップはサポートされていません"
                )
            if args.num_timestep_buckets is not None:
                logger.warning(
                    "num_timestep_buckets is not working well with high and low models training / high and lowモデルのトレーニングではnum_timestep_bucketsがうまく機能しません"
                )

        self.timestep_boundary = (
            args.timestep_boundary if args.timestep_boundary is not None else self.config.boundary
        )  # may be None
        if self.timestep_boundary is None and self.high_low_training:
            raise ValueError(
                "timestep_boundary is not specified for high noise model"
                + " / high noiseモデルを使用する場合は、timestep_boundaryを指定する必要があります。"
            )
        if self.timestep_boundary is not None:
            if self.timestep_boundary > 1:
                self.timestep_boundary /= 1000.0  # convert to 0 to 1 range
            logger.info(f"Converted timestep_boundary to 0 to 1 range: {self.timestep_boundary}")

        self.default_guidance_scale = 1.0  # not used
    
    def process_sample_prompts(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        sample_prompts: str,
    ):
        config = self.config
        device = accelerator.device
        t5_path, clip_path, fp8_t5 = args.t5, args.clip, args.fp8_t5

        logger.info(f"cache Text Encoder outputs for sample prompt: {sample_prompts}")
        prompts = load_prompts(sample_prompts)

        def encode_for_text_encoder(text_encoder):
            sample_prompts_te_outputs = {}  # (prompt) -> (embeds, mask)
            # with accelerator.autocast(), torch.no_grad(): # this causes NaN if dit_dtype is fp16
            t5_dtype = config.t5_dtype
            with torch.amp.autocast(device_type=device.type, dtype=t5_dtype), torch.no_grad():
                for prompt_dict in prompts:
                    if "negative_prompt" not in prompt_dict:
                        prompt_dict["negative_prompt"] = self.config["sample_neg_prompt"]
                    for p in [prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", None)]:
                        if p is None:
                            continue
                        if p not in sample_prompts_te_outputs:
                            logger.info(f"cache Text Encoder outputs for prompt: {p}")

                            prompt_outputs = text_encoder([p], device)
                            sample_prompts_te_outputs[p] = prompt_outputs

            return sample_prompts_te_outputs

        # Load Text Encoder 1 and encode
        logger.info(f"loading T5: {t5_path}")
        t5 = T5EncoderModel(text_len=config.text_len, dtype=config.t5_dtype, device=device, weight_path=t5_path, fp8=fp8_t5)

        logger.info("encoding with Text Encoder 1")
        te_outputs_1 = encode_for_text_encoder(t5)
        del t5

        # prepare sample parameters
        sample_parameters = []
        for prompt_dict in prompts:
            prompt_dict_copy = prompt_dict.copy()

            p = prompt_dict.get("prompt", "")
            prompt_dict_copy["t5_embeds"] = te_outputs_1[p][0]

            p = prompt_dict.get("negative_prompt", None)
            if p is not None:
                prompt_dict_copy["negative_t5_embeds"] = te_outputs_1[p][0]

            p = prompt_dict.get("image_path", None)
            
            sample_parameters.append(prompt_dict_copy)

        clean_memory_on_device(accelerator.device)

        return sample_parameters

    def swap_high_low_weights(self, args: argparse.Namespace, accelerator: Accelerator, model: WanModel):
        if self.current_model_is_high_noise != self.next_model_is_high_noise:
            if self.blocks_to_swap == 0:
                # If offloading inactive DiT, move the model to CPU first
                if args.offload_inactive_dit:
                    model.to("cpu", non_blocking=True)
                    synchronize_device(accelerator.device)  # wait for the CPU to finish
                    clean_memory_on_device(accelerator.device)

                state_dict = model.state_dict()  # CPU or accelerator.device

                info = model.load_state_dict(self.dit_inactive_state_dict, strict=True, assign=True)
                assert len(info.missing_keys) == 0, f"Missing keys: {info.missing_keys}"
                assert len(info.unexpected_keys) == 0, f"Unexpected keys: {info.unexpected_keys}"

                if args.offload_inactive_dit:
                    model.to(accelerator.device, non_blocking=True)
                    synchronize_device(accelerator.device)

                self.dit_inactive_state_dict = state_dict  # swap the state dict
            else:
                # If block swap is enabled, we cannot use offloading inactive DiT, because weights are partially on CPU
                state_dict = model.state_dict()  # CPU or accelerator.device

                info = model.load_state_dict(self.dit_inactive_state_dict, strict=True, assign=True)
                assert len(info.missing_keys) == 0, f"Missing keys: {info.missing_keys}"
                assert len(info.unexpected_keys) == 0, f"Unexpected keys: {info.unexpected_keys}"

                self.dit_inactive_state_dict = state_dict  # swap the state dict

            self.current_model_is_high_noise = self.next_model_is_high_noise

    def call_dit(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer,
        latents: torch.Tensor,
        batch: dict[str, torch.Tensor],
        noise: torch.Tensor,
        noisy_model_input: torch.Tensor,
        timesteps: torch.Tensor,
        network_dtype: torch.dtype,
    ):
        if self.high_low_training:
            # high-low training case
            self.swap_high_low_weights(args, accelerator, transformer)

        # Call the DiT model
        return self._call_dit(args, accelerator, transformer, latents, batch, noise, noisy_model_input, timesteps, network_dtype)
    
    def _call_dit(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer,
        latents: torch.Tensor,
        batch: dict[str, torch.Tensor],
        noise: torch.Tensor,
        noisy_model_input: torch.Tensor,
        timesteps: torch.Tensor,
        network_dtype: torch.dtype,
    ):
        model: WanModel = transformer

        # I2V training and Control training
        image_latents = None
        clip_fea = None
        if self.i2v_training:
            image_latents = batch["latents_image"]
            image_latents = image_latents.to(device=accelerator.device, dtype=network_dtype)

            if not self.config.v2_2:
                clip_fea = batch["clip"]
                clip_fea = clip_fea.to(device=accelerator.device, dtype=network_dtype)

                # clip_fea is [B, N, D] (normal) or [B, 1, N, D] (one frame) for I2V, and [B, 2, N, D] for FLF2V, we need to reshape it to [B, N, D] for I2V and [B*2, N, D] for FLF2V
                if clip_fea.shape[1] == 1:
                    clip_fea = clip_fea.squeeze(1)
                elif clip_fea.shape[1] == 2:
                    clip_fea = clip_fea.view(-1, clip_fea.shape[2], clip_fea.shape[3])

        if self.control_training:
            control_latents = batch["latents_control"]
            control_latents = control_latents.to(device=accelerator.device, dtype=network_dtype)
            if image_latents is not None:
                image_latents = image_latents[:, 4:]  # remove mask for Wan2.1-Fun-Control
                image_latents[:, :, 1:] = 0  # remove except the first frame
            else:
                image_latents = torch.zeros_like(control_latents)  # B, C, F, H, W
            image_latents = torch.concat([control_latents, image_latents], dim=1)  # B, C, F, H, W
            control_latents = None

        context = [t.to(device=accelerator.device, dtype=network_dtype) for t in batch["t5"]]

        # ensure the hidden state will require grad
        if args.gradient_checkpointing:
            noisy_model_input.requires_grad_(True)
            for t in context:
                t.requires_grad_(True)
            if image_latents is not None:
                image_latents.requires_grad_(True)
            if clip_fea is not None:
                clip_fea.requires_grad_(True)

        # call DiT
        lat_f, lat_h, lat_w = latents.shape[2:5]
        seq_len = lat_f * lat_h * lat_w // (self.config.patch_size[0] * self.config.patch_size[1] * self.config.patch_size[2])
        latents = latents.to(device=accelerator.device, dtype=network_dtype)
        noisy_model_input = noisy_model_input.to(device=accelerator.device, dtype=network_dtype)
        with accelerator.autocast():
            model_pred = model(noisy_model_input, t=timesteps, context=context, clip_fea=clip_fea, seq_len=seq_len, y=image_latents)
        model_pred = torch.stack(model_pred, dim=0)  # list to tensor

        # flow matching loss
        target = noise - latents

        return model_pred, target

    def train(self, args):
        # check required arguments
        if args.dataset_config is None:
            raise ValueError("dataset_config is required / dataset_configが必要です")
        if args.dit is None:
            raise ValueError("path to DiT model is required / DiTモデルのパスが必要です")
        assert not args.fp8_scaled or args.fp8_base, "fp8_scaled requires fp8_base / fp8_scaledはfp8_baseが必要です"

        if args.sage_attn:
            raise ValueError(
                "SageAttention doesn't support training currently. Please use `--sdpa` or `--xformers` etc. instead."
                " / SageAttentionは現在学習をサポートしていないようです。`--sdpa`や`--xformers`などの他のオプションを使ってください"
            )

        if args.disable_numpy_memmap:
            logger.info(
                "Disabling numpy memory mapping for model loading (for Wan, FramePack and Qwen-Image). This may lead to higher memory usage but can speed up loading in some cases."
                " / モデル読み込み時のnumpyメモリマッピングを無効にします（Wan、FramePack、Qwen-Imageでのみ有効）。これによりメモリ使用量が増える可能性がありますが、場合によっては読み込みが高速化されることがあります"
            )

        # check model specific arguments
        self.handle_model_specific_args(args)

        # show timesteps for debugging
        if args.show_timesteps:
            self.show_timesteps(args)
            return

        session_id = random.randint(0, 2**32)
        training_started_at = time.time()
        # setup_logging(args, reset=True)

        if args.seed is None:
            args.seed = random.randint(0, 2**32)
        set_seed(args.seed)

        # Load dataset config
        if args.num_timestep_buckets is not None:
            logger.info(f"Using timestep bucketing. Number of buckets: {args.num_timestep_buckets}")
        self.num_timestep_buckets = args.num_timestep_buckets  # None or int, None makes all the behavior same as before

        current_epoch = Value("i", 0)  # shared between processes

        blueprint_generator = BlueprintGenerator(ConfigSanitizer())
        logger.info(f"Load dataset config from {args.dataset_config}")
        user_config = config_utils.load_user_config(args.dataset_config)
        blueprint = blueprint_generator.generate(user_config, args, architecture=self.architecture)
        train_dataset_group = config_utils.generate_dataset_group_by_blueprint(
            blueprint.dataset_group, training=True, num_timestep_buckets=self.num_timestep_buckets, shared_epoch=current_epoch
        )

        if train_dataset_group.num_train_items == 0:
            raise ValueError(
                "No training items found in the dataset. Please ensure that the latent/Text Encoder cache has been created beforehand."
                " / データセットに学習データがありません。latent/Text Encoderキャッシュを事前に作成したか確認してください"
            )

        ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
        collator = collator_class(current_epoch, ds_for_collator)

        # prepare accelerator
        logger.info("preparing accelerator")
        accelerator = prepare_accelerator(args)
        if args.mixed_precision is None:
            args.mixed_precision = accelerator.mixed_precision
            logger.info(f"mixed precision set to {args.mixed_precision} / mixed precisionを{args.mixed_precision}に設定")
        is_main_process = accelerator.is_main_process

        # prepare dtype
        weight_dtype = torch.float32
        if args.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif args.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        # HunyuanVideo: bfloat16 or float16, Wan2.1: bfloat16
        dit_dtype = torch.bfloat16 if args.dit_dtype is None else model_utils.str_to_dtype(args.dit_dtype)
        dit_weight_dtype = (None if args.fp8_scaled else torch.float8_e4m3fn) if args.fp8_base else dit_dtype
        logger.info(f"DiT precision: {dit_dtype}, weight precision: {dit_weight_dtype}")

        # get embedding for sampling images
        vae_dtype = torch.float16 if args.vae_dtype is None else model_utils.str_to_dtype(args.vae_dtype)
        sample_parameters = None
        vae = None
        if args.sample_prompts:
            sample_parameters = self.process_sample_prompts(args, accelerator, args.sample_prompts)

            # Load VAE model for sampling images: VAE is loaded to cpu to save gpu memory
            vae = self.load_vae(args, vae_dtype=vae_dtype, vae_path=args.vae)
            vae.requires_grad_(False)
            vae.eval()

        # load DiT model
        blocks_to_swap = args.blocks_to_swap if args.blocks_to_swap else 0
        self.blocks_to_swap = blocks_to_swap
        loading_device = "cpu" if blocks_to_swap > 0 else accelerator.device

        logger.info(f"Loading DiT model from {args.dit}")
        if args.sdpa:
            attn_mode = "torch"
        elif args.flash_attn:
            attn_mode = "flash"
        elif args.sage_attn:
            attn_mode = "sageattn"
        elif args.xformers:
            attn_mode = "xformers"
        elif args.flash3:
            attn_mode = "flash3"
        else:
            raise ValueError(
                "either --sdpa, --flash-attn, --flash3, --sage-attn or --xformers must be specified / --sdpa, --flash-attn, --flash3, --sage-attn, --xformersのいずれかを指定してください"
            )
        transformer = self.load_transformer(
            accelerator, args, args.dit, attn_mode, args.split_attn, loading_device, dit_weight_dtype
        )
        transformer.eval()
        transformer.requires_grad_(False)

        if blocks_to_swap > 0:
            logger.info(f"enable swap {blocks_to_swap} blocks to CPU from device: {accelerator.device}")
            transformer.enable_block_swap(blocks_to_swap, accelerator.device, supports_backward=True)
            transformer.move_to_device_except_swap_blocks(accelerator.device)

        # load network model for differential training
        sys.path.append(os.path.dirname(__file__))
        accelerator.print("import network module:", args.network_module)
        network_module: lora_module = importlib.import_module(args.network_module)  # actual module may be different

        if args.base_weights is not None:
            # if base_weights is specified, merge the weights to DiT model
            for i, weight_path in enumerate(args.base_weights):
                if args.base_weights_multiplier is None or len(args.base_weights_multiplier) <= i:
                    multiplier = 1.0
                else:
                    multiplier = args.base_weights_multiplier[i]

                accelerator.print(f"merging module: {weight_path} with multiplier {multiplier}")

                weights_sd = load_file(weight_path)
                module = network_module.create_arch_network_from_weights(
                    multiplier, weights_sd, unet=transformer, for_inference=True
                )
                module.merge_to(None, transformer, weights_sd, weight_dtype, "cpu")

            accelerator.print(f"all weights merged: {', '.join(args.base_weights)}")

        # prepare network
        net_kwargs = {}
        if args.network_args is not None:
            for net_arg in args.network_args:
                key, value = net_arg.split("=")
                net_kwargs[key] = value

        if args.dim_from_weights:
            logger.info(f"Loading network from weights: {args.dim_from_weights}")
            weights_sd = load_file(args.dim_from_weights)
            network, _ = network_module.create_arch_network_from_weights(1, weights_sd, unet=transformer)
        else:
            # We use the name create_arch_network for compatibility with LyCORIS
            if hasattr(network_module, "create_arch_network"):
                network = network_module.create_arch_network(
                    1.0,
                    args.network_dim,
                    args.network_alpha,
                    vae,
                    None,
                    transformer,
                    neuron_dropout=args.network_dropout,
                    **net_kwargs,
                )
            else:
                # LyCORIS compatibility
                network = network_module.create_network(
                    1.0,
                    args.network_dim,
                    args.network_alpha,
                    vae,
                    None,
                    transformer,
                    **net_kwargs,
                )
        if network is None:
            return

        if hasattr(network_module, "prepare_network"):
            network.prepare_network(args)

        # apply network to DiT
        network.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)

        if args.network_weights is not None:
            # FIXME consider alpha of weights: this assumes that the alpha is not changed
            info = network.load_weights(args.network_weights)
            accelerator.print(f"load network weights from {args.network_weights}: {info}")

        if args.gradient_checkpointing:
            transformer.enable_gradient_checkpointing(args.gradient_checkpointing_cpu_offload)
            network.enable_gradient_checkpointing()  # may have no effect

        # prepare optimizer, data loader etc.
        accelerator.print("prepare optimizer, data loader etc.")

        trainable_params, lr_descriptions = network.prepare_optimizer_params(unet_lr=args.learning_rate)
        optimizer_name, optimizer_args, optimizer, optimizer_train_fn, optimizer_eval_fn = self.get_optimizer(
            args, trainable_params
        )

        # prepare dataloader

        # num workers for data loader: if 0, persistent_workers is not available
        n_workers = min(args.max_data_loader_n_workers, os.cpu_count())  # cpu_count or max_data_loader_n_workers

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset_group,
            batch_size=1,
            shuffle=True,
            collate_fn=collator,
            num_workers=n_workers,
            persistent_workers=args.persistent_data_loader_workers,
        )

        # calculate max_train_steps
        if args.max_train_epochs is not None:
            args.max_train_steps = args.max_train_epochs * math.ceil(
                len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps
            )
            accelerator.print(
                f"override steps. steps for {args.max_train_epochs} epochs is / 指定エポックまでのステップ数: {args.max_train_steps}"
            )

        # send max_train_steps to train_dataset_group
        train_dataset_group.set_max_train_steps(args.max_train_steps)

        # prepare lr_scheduler
        lr_scheduler = self.get_lr_scheduler(args, optimizer, accelerator.num_processes)

        # prepare training model. accelerator does some magic here

        # experimental feature: train the model with gradients in fp16/bf16
        network_dtype = torch.float32
        args.full_fp16 = args.full_bf16 = False  # temporary disabled because stochastic rounding is not supported yet
        if args.full_fp16:
            assert args.mixed_precision == "fp16", (
                "full_fp16 requires mixed precision='fp16' / full_fp16を使う場合はmixed_precision='fp16'を指定してください。"
            )
            accelerator.print("enable full fp16 training.")
            network_dtype = weight_dtype
            network.to(network_dtype)
        elif args.full_bf16:
            assert args.mixed_precision == "bf16", (
                "full_bf16 requires mixed precision='bf16' / full_bf16を使う場合はmixed_precision='bf16'を指定してください。"
            )
            accelerator.print("enable full bf16 training.")
            network_dtype = weight_dtype
            network.to(network_dtype)

        if dit_weight_dtype != dit_dtype and dit_weight_dtype is not None:
            logger.info(f"casting model to {dit_weight_dtype}")
            transformer.to(dit_weight_dtype)

        if blocks_to_swap > 0:
            transformer = accelerator.prepare(transformer, device_placement=[not blocks_to_swap > 0])
            accelerator.unwrap_model(transformer).move_to_device_except_swap_blocks(accelerator.device)  # reduce peak memory usage
            accelerator.unwrap_model(transformer).prepare_block_swap_before_forward()
        else:
            transformer = accelerator.prepare(transformer)

        network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(network, optimizer, train_dataloader, lr_scheduler)
        training_model = network

        if args.gradient_checkpointing:
            transformer.train()
        else:
            transformer.eval()

        accelerator.unwrap_model(network).prepare_grad_etc(transformer)

        if args.full_fp16:
            # patch accelerator for fp16 training
            # def patch_accelerator_for_fp16_training(accelerator):
            org_unscale_grads = accelerator.scaler._unscale_grads_

            def _unscale_grads_replacer(optimizer, inv_scale, found_inf, allow_fp16):
                return org_unscale_grads(optimizer, inv_scale, found_inf, True)

            accelerator.scaler._unscale_grads_ = _unscale_grads_replacer

        # before resuming make hook for saving/loading to save/load the network weights only
        def save_model_hook(models, weights, output_dir):
            # pop weights of other models than network to save only network weights
            # only main process or deepspeed https://github.com/huggingface/diffusers/issues/2606
            if accelerator.is_main_process:  # or args.deepspeed:
                remove_indices = []
                for i, model in enumerate(models):
                    if not isinstance(model, type(accelerator.unwrap_model(network))):
                        remove_indices.append(i)
                for i in reversed(remove_indices):
                    if len(weights) > i:
                        weights.pop(i)
                # print(f"save model hook: {len(weights)} weights will be saved")

        def load_model_hook(models, input_dir):
            # remove models except network
            remove_indices = []
            for i, model in enumerate(models):
                if not isinstance(model, type(accelerator.unwrap_model(network))):
                    remove_indices.append(i)
            for i in reversed(remove_indices):
                models.pop(i)
            # print(f"load model hook: {len(models)} models will be loaded")

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

        # resume from local or huggingface. accelerator.step is set
        self.resume_from_local_or_hf_if_specified(accelerator, args)  # accelerator.load_state(args.resume)

        # epoch数を計算する
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        # 学習する
        # total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        accelerator.print("running training / 学習開始")
        accelerator.print(f"  num train items / 学習画像、動画数: {train_dataset_group.num_train_items}")
        accelerator.print(f"  num batches per epoch / 1epochのバッチ数: {len(train_dataloader)}")
        accelerator.print(f"  num epochs / epoch数: {num_train_epochs}")
        accelerator.print(
            f"  batch size per device / バッチサイズ: {', '.join([str(d.batch_size) for d in train_dataset_group.datasets])}"
        )
        # accelerator.print(f"  total train batch size (with parallel & distributed & accumulation) / 総バッチサイズ（並列学習、勾配合計含む）: {total_batch_size}")
        accelerator.print(f"  gradient accumulation steps / 勾配を合計するステップ数 = {args.gradient_accumulation_steps}")
        accelerator.print(f"  total optimization steps / 学習ステップ数: {args.max_train_steps}")

        # TODO refactor metadata creation and move to util
        metadata = {
            "ss_session_id": session_id,  # random integer indicating which group of epochs the model came from
            "ss_training_started_at": training_started_at,  # unix timestamp
            "ss_output_name": args.output_name,
            "ss_learning_rate": args.learning_rate,
            "ss_num_train_items": train_dataset_group.num_train_items,
            "ss_num_batches_per_epoch": len(train_dataloader),
            "ss_num_epochs": num_train_epochs,
            "ss_gradient_checkpointing": args.gradient_checkpointing,
            "ss_gradient_checkpointing_cpu_offload": args.gradient_checkpointing_cpu_offload,
            "ss_gradient_accumulation_steps": args.gradient_accumulation_steps,
            "ss_max_train_steps": args.max_train_steps,
            "ss_lr_warmup_steps": args.lr_warmup_steps,
            "ss_lr_scheduler": args.lr_scheduler,
            SS_METADATA_KEY_BASE_MODEL_VERSION: self.architecture_full_name,
            # "ss_network_module": args.network_module,
            # "ss_network_dim": args.network_dim,  # None means default because another network than LoRA may have another default dim
            # "ss_network_alpha": args.network_alpha,  # some networks may not have alpha
            SS_METADATA_KEY_NETWORK_MODULE: args.network_module,
            SS_METADATA_KEY_NETWORK_DIM: args.network_dim,
            SS_METADATA_KEY_NETWORK_ALPHA: args.network_alpha,
            "ss_network_dropout": args.network_dropout,  # some networks may not have dropout
            "ss_mixed_precision": args.mixed_precision,
            "ss_seed": args.seed,
            "ss_training_comment": args.training_comment,  # will not be updated after training
            # "ss_sd_scripts_commit_hash": train_util.get_git_revision_hash(),
            "ss_optimizer": optimizer_name + (f"({optimizer_args})" if len(optimizer_args) > 0 else ""),
            "ss_max_grad_norm": args.max_grad_norm,
            "ss_fp8_base": bool(args.fp8_base),
            # "ss_fp8_llm": bool(args.fp8_llm), # remove this because this is only for HuanyuanVideo TODO set architecure dependent metadata
            "ss_full_fp16": bool(args.full_fp16),
            "ss_full_bf16": bool(args.full_bf16),
            "ss_weighting_scheme": args.weighting_scheme,
            "ss_logit_mean": args.logit_mean,
            "ss_logit_std": args.logit_std,
            "ss_mode_scale": args.mode_scale,
            "ss_guidance_scale": args.guidance_scale,
            "ss_timestep_sampling": args.timestep_sampling,
            "ss_sigmoid_scale": args.sigmoid_scale,
            "ss_discrete_flow_shift": args.discrete_flow_shift,
        }

        datasets_metadata = []
        # tag_frequency = {}  # merge tag frequency for metadata editor # TODO support tag frequency
        for dataset in train_dataset_group.datasets:
            dataset_metadata = dataset.get_metadata()
            datasets_metadata.append(dataset_metadata)

        metadata["ss_datasets"] = json.dumps(datasets_metadata)

        # add extra args
        if args.network_args:
            # metadata["ss_network_args"] = json.dumps(net_kwargs)
            metadata[SS_METADATA_KEY_NETWORK_ARGS] = json.dumps(net_kwargs)

        # model name and hash
        # calculate hash takes time, so we omit it for now
        if args.dit is not None:
            # logger.info(f"calculate hash for DiT model: {args.dit}")
            logger.info(f"set DiT model name for metadata: {args.dit}")
            sd_model_name = args.dit
            if os.path.exists(sd_model_name):
                # metadata["ss_sd_model_hash"] = model_utils.model_hash(sd_model_name)
                # metadata["ss_new_sd_model_hash"] = model_utils.calculate_sha256(sd_model_name)
                sd_model_name = os.path.basename(sd_model_name)
            metadata["ss_sd_model_name"] = sd_model_name

        if args.vae is not None:
            # logger.info(f"calculate hash for VAE model: {args.vae}")
            logger.info(f"set VAE model name for metadata: {args.vae}")
            vae_name = args.vae
            if os.path.exists(vae_name):
                # metadata["ss_vae_hash"] = model_utils.model_hash(vae_name)
                # metadata["ss_new_vae_hash"] = model_utils.calculate_sha256(vae_name)
                vae_name = os.path.basename(vae_name)
            metadata["ss_vae_name"] = vae_name

        metadata = {k: str(v) for k, v in metadata.items()}

        # make minimum metadata for filtering
        minimum_metadata = {}
        for key in SS_METADATA_MINIMUM_KEYS:
            if key in metadata:
                minimum_metadata[key] = metadata[key]

        if accelerator.is_main_process:
            init_kwargs = {}
            if args.wandb_run_name:
                init_kwargs["wandb"] = {"name": args.wandb_run_name}
            if args.log_tracker_config is not None:
                init_kwargs = toml.load(args.log_tracker_config)
            accelerator.init_trackers(
                "network_train" if args.log_tracker_name is None else args.log_tracker_name,
                config=train_utils.get_sanitized_config_or_none(args),
                init_kwargs=init_kwargs,
            )

        # TODO skip until initial step
        progress_bar = tqdm(range(args.max_train_steps), smoothing=0, disable=not accelerator.is_local_main_process, desc="steps")

        epoch_to_start = 0
        global_step = 0
        noise_scheduler = FlowMatchDiscreteScheduler(shift=args.discrete_flow_shift, reverse=True, solver="euler")

        loss_recorder = train_utils.LossRecorder()
        del train_dataset_group

        # function for saving/removing
        save_dtype = dit_dtype

        def save_model(ckpt_name: str, unwrapped_nw, steps, epoch_no, force_sync_upload=False):
            os.makedirs(args.output_dir, exist_ok=True)
            ckpt_file = os.path.join(args.output_dir, ckpt_name)

            accelerator.print(f"\nsaving checkpoint: {ckpt_file}")
            metadata["ss_training_finished_at"] = str(time.time())
            metadata["ss_steps"] = str(steps)
            metadata["ss_epoch"] = str(epoch_no)

            metadata_to_save = minimum_metadata if args.no_metadata else metadata

            title = args.metadata_title if args.metadata_title is not None else args.output_name
            if args.min_timestep is not None or args.max_timestep is not None:
                min_time_step = args.min_timestep if args.min_timestep is not None else 0
                max_time_step = args.max_timestep if args.max_timestep is not None else 1000
                md_timesteps = (min_time_step, max_time_step)
            else:
                md_timesteps = None

            sai_metadata = sai_model_spec.build_metadata(
                None,
                self.architecture,
                time.time(),
                title,
                args.metadata_reso,
                args.metadata_author,
                args.metadata_description,
                args.metadata_license,
                args.metadata_tags,
                timesteps=md_timesteps,
                custom_arch=args.metadata_arch,
            )

            metadata_to_save.update(sai_metadata)

            unwrapped_nw.save_weights(ckpt_file, save_dtype, metadata_to_save)
            if args.huggingface_repo_id is not None:
                huggingface_utils.upload(args, ckpt_file, "/" + ckpt_name, force_sync_upload=force_sync_upload)

        def remove_model(old_ckpt_name):
            old_ckpt_file = os.path.join(args.output_dir, old_ckpt_name)
            if os.path.exists(old_ckpt_file):
                accelerator.print(f"removing old checkpoint: {old_ckpt_file}")
                os.remove(old_ckpt_file)

        # For --sample_at_first
        if should_sample_images(args, global_step, epoch=0):
            optimizer_eval_fn()
            self.sample_images(accelerator, args, 0, global_step, vae, transformer, sample_parameters, dit_dtype)
            optimizer_train_fn()
        if len(accelerator.trackers) > 0:
            # log empty object to commit the sample images to wandb
            accelerator.log({}, step=0)

        # training loop

        # log device and dtype for each model
        logger.info(f"DiT dtype: {transformer.dtype}, device: {transformer.device}")

        clean_memory_on_device(accelerator.device)

        optimizer_train_fn()  # Set training mode

        for epoch in range(epoch_to_start, num_train_epochs):
            accelerator.print(f"\nepoch {epoch + 1}/{num_train_epochs}")
            current_epoch.value = epoch + 1

            metadata["ss_epoch"] = str(epoch + 1)

            accelerator.unwrap_model(network).on_epoch_start(transformer)

            for step, batch in enumerate(train_dataloader):
                latents = batch["latents"]

                with accelerator.accumulate(training_model):
                    accelerator.unwrap_model(network).on_step_start()

                    latents = self.scale_shift_latents(latents)

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)

                    # calculate model input and timesteps
                    noisy_model_input, timesteps = self.get_noisy_model_input_and_timesteps(
                        args, noise, latents, batch["timesteps"], noise_scheduler, accelerator.device, dit_dtype
                    )

                    weighting = compute_loss_weighting_for_sd3(
                        args.weighting_scheme, noise_scheduler, timesteps, accelerator.device, dit_dtype
                    )

                    model_pred, target = self.call_dit(
                        args, accelerator, transformer, latents, batch, noise, noisy_model_input, timesteps, network_dtype
                    )
                    loss = torch.nn.functional.mse_loss(model_pred.to(network_dtype), target, reduction="none")

                    if weighting is not None:
                        loss = loss * weighting
                    # loss = loss.mean([1, 2, 3])
                    # # min snr gamma, scale v pred loss like noise pred, v pred like loss, debiased estimation etc.
                    # loss = self.post_process_loss(loss, args, timesteps, noise_scheduler)

                    loss = loss.mean()  # mean loss over all elements in batch

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        # self.all_reduce_network(accelerator, network)  # sync DDP grad manually
                        state = accelerate.PartialState()
                        if state.distributed_type != accelerate.DistributedType.NO:
                            for param in network.parameters():
                                if param.grad is not None:
                                    param.grad = accelerator.reduce(param.grad, reduction="mean")

                        if args.max_grad_norm != 0.0:
                            params_to_clip = accelerator.unwrap_model(network).get_trainable_params()
                            accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                if args.scale_weight_norms:
                    keys_scaled, mean_norm, maximum_norm = accelerator.unwrap_model(network).apply_max_norm_regularization(
                        args.scale_weight_norms, accelerator.device
                    )
                    max_mean_logs = {"Keys Scaled": keys_scaled, "Average key norm": mean_norm}
                else:
                    keys_scaled, mean_norm, maximum_norm = None, None, None

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    # to avoid calling optimizer_eval_fn() too frequently, we call it only when we need to sample images or save the model
                    should_sampling = should_sample_images(args, global_step, epoch=None)
                    should_saving = args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0

                    if should_sampling or should_saving:
                        optimizer_eval_fn()
                        if should_sampling:
                            self.sample_images(accelerator, args, None, global_step, vae, transformer, sample_parameters, dit_dtype)

                        if should_saving:
                            accelerator.wait_for_everyone()
                            if accelerator.is_main_process:
                                ckpt_name = train_utils.get_step_ckpt_name(args.output_name, global_step)
                                save_model(ckpt_name, accelerator.unwrap_model(network), global_step, epoch)

                                if args.save_state:
                                    train_utils.save_and_remove_state_stepwise(args, accelerator, global_step)

                                remove_step_no = train_utils.get_remove_step_no(args, global_step)
                                if remove_step_no is not None:
                                    remove_ckpt_name = train_utils.get_step_ckpt_name(args.output_name, remove_step_no)
                                    remove_model(remove_ckpt_name)
                        optimizer_train_fn()

                current_loss = loss.detach().item()
                loss_recorder.add(epoch=epoch, step=step, loss=current_loss)
                avr_loss: float = loss_recorder.moving_average
                logs = {"avr_loss": avr_loss}  # , "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

                if args.scale_weight_norms:
                    progress_bar.set_postfix(**{**max_mean_logs, **logs})

                if len(accelerator.trackers) > 0:
                    logs = self.generate_step_logs(
                        args, current_loss, avr_loss, lr_scheduler, lr_descriptions, optimizer, keys_scaled, mean_norm, maximum_norm
                    )
                    accelerator.log(logs, step=global_step)

                if global_step >= args.max_train_steps:
                    break

            if len(accelerator.trackers) > 0:
                logs = {"loss/epoch": loss_recorder.moving_average}
                accelerator.log(logs, step=epoch + 1)

            accelerator.wait_for_everyone()

            # save model at the end of epoch if needed
            optimizer_eval_fn()
            if args.save_every_n_epochs is not None:
                saving = (epoch + 1) % args.save_every_n_epochs == 0 and (epoch + 1) < num_train_epochs
                if is_main_process and saving:
                    ckpt_name = train_utils.get_epoch_ckpt_name(args.output_name, epoch + 1)
                    save_model(ckpt_name, accelerator.unwrap_model(network), global_step, epoch + 1)

                    remove_epoch_no = train_utils.get_remove_epoch_no(args, epoch + 1)
                    if remove_epoch_no is not None:
                        remove_ckpt_name = train_utils.get_epoch_ckpt_name(args.output_name, remove_epoch_no)
                        remove_model(remove_ckpt_name)

                    if args.save_state:
                        train_utils.save_and_remove_state_on_epoch_end(args, accelerator, epoch + 1)

            self.sample_images(accelerator, args, epoch + 1, global_step, vae, transformer, sample_parameters, dit_dtype)
            optimizer_train_fn()

            # end of epoch

        # metadata["ss_epoch"] = str(num_train_epochs)
        metadata["ss_training_finished_at"] = str(time.time())

        if is_main_process:
            network = accelerator.unwrap_model(network)

        accelerator.end_training()
        optimizer_eval_fn()

        if is_main_process and (args.save_state or args.save_state_on_train_end):
            train_utils.save_state_on_train_end(args, accelerator)

        if is_main_process:
            ckpt_name = train_utils.get_last_ckpt_name(args.output_name)
            save_model(ckpt_name, network, global_step, num_train_epochs, force_sync_upload=True)

            logger.info("model saved.")

    def sample_images(self, accelerator, args, epoch, steps, vae, transformer, sample_parameters, dit_dtype):
        """architecture independent sample images"""
        if not should_sample_images(args, steps, epoch):
            return

        logger.info("")
        logger.info(f"generating sample images at step / サンプル画像生成 ステップ: {steps}")
        if sample_parameters is None:
            logger.error(f"No prompt file / プロンプトファイルがありません: {args.sample_prompts}")
            return

        distributed_state = PartialState()  # for multi gpu distributed inference. this is a singleton, so it's safe to use it here

        # Use the unwrapped model
        transformer = accelerator.unwrap_model(transformer)
        transformer.switch_block_swap_for_inference()

        # Create a directory to save the samples
        save_dir = os.path.join(args.output_dir, "sample")
        os.makedirs(save_dir, exist_ok=True)

        # save random state to restore later
        rng_state = torch.get_rng_state()
        cuda_rng_state = None
        try:
            cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        except Exception:
            pass

        if distributed_state.num_processes <= 1:
            # If only one device is available, just use the original prompt list. We don't need to care about the distribution of prompts.
            with torch.no_grad(), accelerator.autocast():
                for sample_parameter in sample_parameters:
                    self.sample_image_inference(
                        accelerator, args, transformer, dit_dtype, vae, save_dir, sample_parameter, epoch, steps
                    )
                    clean_memory_on_device(accelerator.device)
        else:
            # Creating list with N elements, where each element is a list of prompt_dicts, and N is the number of processes available (number of devices available)
            # prompt_dicts are assigned to lists based on order of processes, to attempt to time the image creation time to match enum order. Probably only works when steps and sampler are identical.
            per_process_params = []  # list of lists
            for i in range(distributed_state.num_processes):
                per_process_params.append(sample_parameters[i :: distributed_state.num_processes])

            with torch.no_grad():
                with distributed_state.split_between_processes(per_process_params) as sample_parameter_lists:
                    for sample_parameter in sample_parameter_lists[0]:
                        self.sample_image_inference(
                            accelerator, args, transformer, dit_dtype, vae, save_dir, sample_parameter, epoch, steps
                        )
                        clean_memory_on_device(accelerator.device)

        torch.set_rng_state(rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state)

        transformer.switch_block_swap_for_training()
        clean_memory_on_device(accelerator.device)

    def sample_image_inference(self, accelerator, args, transformer, dit_dtype, vae, save_dir, sample_parameter, epoch, steps):
        """architecture independent sample images"""
        sample_steps = sample_parameter.get("sample_steps", 20)
        width = sample_parameter.get("width", 256)  # make smaller for faster and memory saving inference
        height = sample_parameter.get("height", 256)
        frame_count = sample_parameter.get("frame_count", 1)
        guidance_scale = sample_parameter.get("guidance_scale", self.default_guidance_scale)
        discrete_flow_shift = sample_parameter.get("discrete_flow_shift", 14.5)
        seed = sample_parameter.get("seed")
        prompt: str = sample_parameter.get("prompt", "")
        cfg_scale = sample_parameter.get("cfg_scale", None)  # None for architecture default
        negative_prompt = sample_parameter.get("negative_prompt", None)

        # round width and height to multiples of 8
        width = (width // 8) * 8
        height = (height // 8) * 8

        frame_count = (frame_count - 1) // 4 * 4 + 1  # 1, 5, 9, 13, ... For HunyuanVideo and Wan2.1

        if self.i2v_training:
            image_path = sample_parameter.get("image_path", None)
            if image_path is None:
                logger.error("No image_path for i2v model / i2vモデルのサンプル画像生成にはimage_pathが必要です")
                return
        else:
            image_path = None

        if self.control_training:
            control_video_path = sample_parameter.get("control_video_path", None)
            if control_video_path is None:
                logger.error(
                    "No control_video_path for control model / controlモデルのサンプル画像生成にはcontrol_video_pathが必要です"
                )
                return
        else:
            control_video_path = None

        device = accelerator.device
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            generator = torch.Generator(device=device).manual_seed(seed)
        else:
            # True random sample image generation
            torch.seed()
            torch.cuda.seed()
            generator = torch.Generator(device=device).manual_seed(torch.initial_seed())

        logger.info(f"prompt: {prompt}")
        logger.info(f"height: {height}")
        logger.info(f"width: {width}")
        logger.info(f"frame count: {frame_count}")
        logger.info(f"sample steps: {sample_steps}")
        logger.info(f"guidance scale: {guidance_scale}")
        logger.info(f"discrete flow shift: {discrete_flow_shift}")
        if seed is not None:
            logger.info(f"seed: {seed}")

        do_classifier_free_guidance = False
        if negative_prompt is not None:
            do_classifier_free_guidance = True
            logger.info(f"negative prompt: {negative_prompt}")
            logger.info(f"cfg scale: {cfg_scale}")

        if self.i2v_training:
            logger.info(f"image path: {image_path}")
        if self.control_training:
            logger.info(f"control video path: {control_video_path}")

        # inference: architecture dependent
        video = self.do_inference(
            accelerator,
            args,
            sample_parameter,
            vae,
            dit_dtype,
            transformer,
            discrete_flow_shift,
            sample_steps,
            width,
            height,
            frame_count,
            generator,
            do_classifier_free_guidance,
            guidance_scale,
            cfg_scale,
            image_path=image_path,
            control_video_path=control_video_path,
        )

        # Save video
        if video is None:
            logger.error("No video generated / 生成された動画がありません")
            return

        ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
        num_suffix = f"e{epoch:06d}" if epoch is not None else f"{steps:06d}"
        seed_suffix = "" if seed is None else f"_{seed}"
        prompt_idx = sample_parameter.get("enum", 0)
        save_path = (
            f"{'' if args.output_name is None else args.output_name + '_'}{num_suffix}_{prompt_idx:02d}_{ts_str}{seed_suffix}"
        )

        wandb_tracker = None
        try:
            wandb_tracker = accelerator.get_tracker("wandb")  # raises ValueError if wandb is not initialized
            try:
                import wandb
            except ImportError:
                raise ImportError("No wandb / wandb がインストールされていないようです")
        except:  # wandb 無効時
            wandb = None

        if video.shape[2] == 1:
            image_paths = save_images_grid(video, save_dir, save_path, create_subdir=False)
            if wandb_tracker is not None and wandb is not None:
                for image_path in image_paths:
                    wandb_tracker.log({f"sample_{prompt_idx}": wandb.Image(image_path)}, step=steps)
        else:
            video_path = os.path.join(save_dir, save_path) + ".mp4"
            save_videos_grid(video, video_path)
            if wandb_tracker is not None and wandb is not None:
                wandb_tracker.log({f"sample_{prompt_idx}": wandb.Video(video_path)}, step=steps)

        # Move models back to initial state
        vae.to("cpu")
        clean_memory_on_device(device)

    def do_inference(
        self,
        accelerator,
        args,
        sample_parameter,
        vae,
        dit_dtype,
        transformer,
        discrete_flow_shift,
        sample_steps,
        width,
        height,
        frame_count,
        generator,
        do_classifier_free_guidance,
        guidance_scale,
        cfg_scale,
        image_path=None,
        control_video_path=None,
    ):
        """architecture dependent inference"""
        model: WanModel = transformer
        device = accelerator.device

        if self.high_low_training:
            self.next_model_is_high_noise = False  # We use low noise model to sample the video
            self.swap_high_low_weights(args, accelerator, model)

        # TODO support different cfg_scale for low and high noise models
        if cfg_scale is None:
            cfg_scale = self.config.sample_guide_scale[0]  # use low noise guide scale by default
        do_classifier_free_guidance = do_classifier_free_guidance and cfg_scale != 1.0

        # prepare parameters
        one_frame_mode = args.one_frame
        if one_frame_mode:
            target_index, control_indices, f_indices, one_frame_inference_index = parse_one_frame_inference_args(
                sample_parameter["one_frame"]
            )
            latent_video_length = len(f_indices)  # number of frames in the video
        else:
            target_index, control_indices, f_indices, one_frame_inference_index = None, None, None, None

            # Calculate latent video length based on VAE version
            latent_video_length = (frame_count - 1) // self.config["vae_stride"][0] + 1

        # Get embeddings
        context = sample_parameter["t5_embeds"].to(device=device)
        if do_classifier_free_guidance:
            context_null = sample_parameter["negative_t5_embeds"].to(device=device)
        else:
            context_null = None

        num_channels_latents = 16  # model.in_dim
        vae_scale_factor = self.config["vae_stride"][1]

        # Initialize latents
        lat_h = height // vae_scale_factor
        lat_w = width // vae_scale_factor
        shape_or_frame = (1, num_channels_latents, 1, lat_h, lat_w)
        latents = []
        for _ in range(latent_video_length):
            latents.append(torch.randn(shape_or_frame, generator=generator, device=device, dtype=torch.float32))
        latents = torch.cat(latents, dim=2)

        image_latents = None

        if one_frame_mode:
            # One frame inference mode
            logger.info(
                f"One frame inference mode: target_index={target_index}, control_indices={control_indices}, f_indices={f_indices}"
            )
            vae.to(device)
            vae.eval()

            # prepare start and control latent
            def encode_image(path):
                image = Image.open(path)
                if image.mode == "RGBA":
                    alpha = image.split()[-1]
                    image = image.convert("RGB")
                else:
                    alpha = None
                image = resize_image_to_bucket(image, (width, height))  # returns a numpy array
                image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(1).unsqueeze(0).float()  # 1, C, 1, H, W
                image = image / 127.5 - 1  # -1 to 1
                with torch.amp.autocast(device_type=device.type, dtype=vae.dtype), torch.no_grad():
                    image = image.to(device=device)
                    latent = vae.encode(image)[0]
                return latent, alpha

            control_latents = []
            control_alphas = []
            if "control_image_path" in sample_parameter:
                for control_image_path in sample_parameter["control_image_path"]:
                    control_latent, control_alpha = encode_image(control_image_path)
                    control_latents.append(control_latent)
                    control_alphas.append(control_alpha)

            with torch.amp.autocast(device_type=device.type, dtype=vae.dtype), torch.no_grad():
                black_image_latent = vae.encode([torch.zeros((3, 1, height, width), dtype=torch.float32, device=device)])[0]

            # Create latent and mask for the required number of frames
            image_latents = torch.zeros(4 + 16, len(f_indices), lat_h, lat_w, dtype=torch.float32, device=device)
            ci = 0
            for j, index in enumerate(f_indices):
                if index == target_index:
                    image_latents[4:, j : j + 1, :, :] = black_image_latent  # set black latent for the target frame
                else:
                    image_latents[:4, j, :, :] = 1.0  # set mask to 1.0 for the clean latent frames
                    image_latents[4:, j : j + 1, :, :] = control_latents[ci]  # set control latent
                    ci += 1
            image_latents = image_latents.unsqueeze(0)  # add batch dim

            vae.to("cpu")
            clean_memory_on_device(device)

        elif self.i2v_training or self.control_training:
            # Move VAE to the appropriate device for sampling: consider to cache image latents in CPU in advance
            vae.to(device)
            vae.eval()

            if self.i2v_training:
                image = Image.open(image_path)
                image = resize_image_to_bucket(image, (width, height))  # returns a numpy array
                image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(1).float()  # C, 1, H, W
                image = image / 127.5 - 1  # -1 to 1

                # Create mask for the required number of frames
                msk = torch.ones(1, frame_count, lat_h, lat_w, device=device)
                msk[:, 1:] = 0
                msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
                msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
                msk = msk.transpose(1, 2)  # B, C, T, H, W

                with torch.amp.autocast(device_type=device.type, dtype=vae.dtype), torch.no_grad():
                    # Zero padding for the required number of frames only
                    padding_frames = frame_count - 1  # The first frame is the input image
                    image = torch.concat([image, torch.zeros(3, padding_frames, height, width)], dim=1).to(device=device)
                    y = vae.encode([image])[0]

                y = y[:, :latent_video_length]  # may be not needed
                y = y.unsqueeze(0)  # add batch dim
                image_latents = torch.concat([msk, y], dim=1)

            if self.control_training:
                # Control video
                video = load_video(control_video_path, 0, frame_count, bucket_reso=(width, height))  # list of frames
                video = np.stack(video, axis=0)  # F, H, W, C
                video = torch.from_numpy(video).permute(3, 0, 1, 2).float()  # C, F, H, W
                video = video / 127.5 - 1  # -1 to 1
                video = video.to(device=device)

                with torch.amp.autocast(device_type=device.type, dtype=vae.dtype), torch.no_grad():
                    control_latents = vae.encode([video])[0]
                    control_latents = control_latents[:, :latent_video_length]
                    control_latents = control_latents.unsqueeze(0)  # add batch dim

                # We supports Wan2.1-Fun-Control only
                if image_latents is not None:
                    image_latents = image_latents[:, 4:]  # remove mask for Wan2.1-Fun-Control
                    image_latents[:, :, 1:] = 0  # remove except the first frame
                else:
                    image_latents = torch.zeros_like(control_latents)  # B, C, F, H, W

                image_latents = torch.concat([control_latents, image_latents], dim=1)  # B, C, F, H, W

            vae.to("cpu")
            clean_memory_on_device(device)

        # use the default value for num_train_timesteps (1000)
        scheduler = FlowUniPCMultistepScheduler(shift=1, use_dynamic_shifting=False)
        scheduler.set_timesteps(sample_steps, device=device, shift=discrete_flow_shift)
        timesteps = scheduler.timesteps

        # Generate noise for the required number of frames only
        noise = torch.randn(16, latent_video_length, lat_h, lat_w, dtype=torch.float32, generator=generator, device=device).to(
            "cpu"
        )

        # prepare the model input
        max_seq_len = latent_video_length * lat_h * lat_w // (self.config.patch_size[1] * self.config.patch_size[2])
        arg_c = {"context": [context], "seq_len": max_seq_len}
        arg_null = {"context": [context_null], "seq_len": max_seq_len}

        if self.i2v_training and not one_frame_mode:
            if not self.config.v2_2:
                arg_c["clip_fea"] = sample_parameter["clip_embeds"].to(device=device, dtype=dit_dtype)
                arg_null["clip_fea"] = arg_c["clip_fea"]

        if one_frame_mode:
            if not self.config.v2_2:
                if "end_image_clip_embeds" in sample_parameter:
                    arg_c["clip_fea"] = torch.cat(
                        [sample_parameter["clip_embeds"], sample_parameter["end_image_clip_embeds"]], dim=0
                    ).to(device=device, dtype=dit_dtype)
                else:
                    arg_c["clip_fea"] = sample_parameter["clip_embeds"].to(device=device, dtype=dit_dtype)
                arg_null["clip_fea"] = arg_c["clip_fea"]

            arg_c["f_indices"] = [f_indices]
            arg_null["f_indices"] = arg_c["f_indices"]
            # print(f"One arg_c: {arg_c}, arg_null: {arg_null}")

        if self.i2v_training or self.control_training:
            arg_c["y"] = image_latents
            arg_null["y"] = image_latents

        # Wrap the inner loop with tqdm to track progress over timesteps
        prompt_idx = sample_parameter.get("enum", 0)
        latent = noise
        with torch.no_grad():
            for i, t in enumerate(tqdm(timesteps, desc=f"Sampling timesteps for prompt {prompt_idx + 1}")):
                latent_model_input = [latent.to(device=device)]
                timestep = t.unsqueeze(0)

                with accelerator.autocast():
                    noise_pred_cond = model(latent_model_input, t=timestep, **arg_c)[0].to("cpu")
                    if do_classifier_free_guidance:
                        noise_pred_uncond = model(latent_model_input, t=timestep, **arg_null)[0].to("cpu")
                    else:
                        noise_pred_uncond = None

                if do_classifier_free_guidance:
                    noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
                else:
                    noise_pred = noise_pred_cond

                temp_x0 = scheduler.step(noise_pred.unsqueeze(0), t, latent.unsqueeze(0), return_dict=False, generator=generator)[0]
                latent = temp_x0.squeeze(0)

        # Move VAE to the appropriate device for sampling
        vae.to(device)
        vae.eval()

        # Decode latents to video
        logger.info(f"Decoding video from latents: {latent.shape}")
        latent = latent.unsqueeze(0)  # add batch dim
        latent = latent.to(device=device)

        if one_frame_mode:
            latent = latent[:, :, one_frame_inference_index : one_frame_inference_index + 1, :, :]  # select the one frame
        with torch.amp.autocast(device_type=device.type, dtype=vae.dtype), torch.no_grad():
            video = vae.decode(latent)[0]  # vae returns list
        video = video.unsqueeze(0)  # add batch dim
        del latent

        logger.info("Decoding complete")
        video = video.to(torch.float32).cpu()
        video = (video / 2 + 0.5).clamp(0, 1)  # -1 to 1 -> 0 to 1

        vae.to("cpu")
        clean_memory_on_device(device)

        return video