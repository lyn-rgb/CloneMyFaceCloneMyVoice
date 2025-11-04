import torch, types
import numpy as np
from PIL import Image
from einops import repeat
from typing import Optional, Union
from einops import rearrange
import numpy as np
from tqdm import tqdm
from typing import Optional
from typing_extensions import Literal
import imageio
import os
from typing import List, Tuple
import PIL
from utils.model_loading_utils import (
    init_fusion_score_model_ovi, 
    init_text_model, 
    init_mmaudio_vae, 
    init_wan_vae_2_2, 
    load_fusion_checkpoint
)
from modules.utils import load_state_dict
from modules.fusion import FusionModel
from modules.model import WanModel
from modules.model import WanRMSNorm, sinusoidal_embedding_1d
from modules.t5 import T5EncoderModel as WanTextEncoder
from modules.t5 import (
    T5RelativeEmbedding,
    T5LayerNorm,
)
from modules.vae2_2 import WanVideoVAE, RMS_norm, CausalConv3d, Upsample
from modules.clip import CLIPModel as WanImageEncoder
from schedulers.flow_match import FlowMatchScheduler
from vram_management import (
    enable_vram_management,
    AutoWrappedModule,
    AutoWrappedLinear,
    WanAutoCastLayerNorm,
)
from distributed_comms.parallel_states import initialize_sequence_parallel_state
from lora import GeneralLoRALoader


class WanVideoPipeline(torch.nn.Module):
    def __init__(self, config, meta_init=True, device="cuda", torch_dtype=torch.bfloat16):
        super().__init__()
        self.device = device
        self.torch_dtype = torch_dtype
        # The following parameters are used for shape check.
        self.height_division_factor = config.get("height_division_factor", 16)
        self.width_division_factor = config.get("width_division_factor", 16)
        self.time_division_factor = config.get("time_division_factor", 4)
        self.time_division_remainder = config.get("time_division_remainder", 1)
        self.vram_management_enabled = False

        self.config = config
        self.cpu_offload = config.get("cpu_offload", False)

        # init env before construct models
        if config.use_sp:
            self.initialize_usp()

        # init model
        model, video_config, audio_config = init_fusion_score_model_ovi(rank=device, meta_init=meta_init)
        fp8 = config.get("fp8", False)
        if not meta_init:
            if not fp8:
                model = model.to(dtype=self.torch_dtype)
            model = model.to(device=device if not self.cpu_offload else "cpu")
        
        # load vaes
        vae_model_video = init_wan_vae_2_2(config.ckpt_dir, rank=device)
        vae_model_video.model.requires_grad_(False).eval()
        self.video_vae = vae_model_video.bfloat16()

        vae_model_audio = init_mmaudio_vae(config.ckpt_dir, rank=device)
        vae_model_audio.requires_grad_(False).eval()
        self.audio_vae = vae_model_audio.bfloat16()

        # load text encoder
        self.text_model = init_text_model(config.ckpt_dir, rank=device, cpu_offload=self.cpu_offload)
        if config.get("shard_text_model", False):
            raise NotImplementedError("Sharding text model is not implemented yet.")
        if self.cpu_offload:
            self.offload_to_cpu(self.text_model.model)

        # load image encoder (clip)
        self.image_encoder: WanImageEncoder = None

        # load fusion checkpoint
        checkpoint_path = os.path.join(
            config.ckpt_dir,
            "Ovi",
            "model.safetensors" if not fp8 else "model_fp8_e4m3fn.safetensors",
        )
        if not os.path.exists(checkpoint_path):
            raise RuntimeError(f"No fusion checkpoint found in {config.ckpt_dir}")
        load_fusion_checkpoint(model, checkpoint_path=checkpoint_path, from_meta=meta_init)

        if meta_init:
            if not fp8:
                model = model.to(dtype=self.torch_dtype)
            model = model.to(device=device if not self.cpu_offload else "cpu")
            model.set_rope_params()
        
        model.model.init_lora()
        self.model = model
        
        # init scheduler
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)

        # other params
        self._patch_size_h, self._patch_size_w = self.model.video_model.patch_size[1], self.model.video_model.patch_size[2]

        # forward function
        self.model_fn = model_fn

    def switch_to_train(self):
        # set trainable params
        self.model.train()
        self.scheduler.set_timesteps(1000, training=True)
        self.freeze_except(self.model, [] if self.config.get("trainable_models", None) is None else self.config.get("trainable_models", None).split(","))

    def offload_to_cpu(self, model):
        model = model.cpu()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        return model
    
    def get_vram(self):
        return torch.cuda.mem_get_info(self.device)[1] / (1024**3)

    def freeze_except(self, model, model_names):
        for name, sub_model in model.named_children():
            if name in model_names:
                sub_model.train()
                sub_model.requires_grad_(True)
            else:
                sub_model.eval()
                sub_model.requires_grad_(False)
                self.freeze_except(sub_model, model_names)
    
    @torch.no_grad()
    def preprocess_inputs(self, inputs: dict) -> dict:
        outputs = {}
        # 1. process input video
        outputs["batch_size"] = video.shape[0]
        video = inputs["video"].to(dtype=torch.bfloat16, device=self.device)    # B, 3, T, H, W, [-1, 1]
        video_input_latents = self.video_vae.wrapped_encode(video).to(dtype=self.torch_dtype, device=self.device)
        outputs["video_input_latents"] = video_input_latents    # B, C, T', H', W'
        
        # 2. process input audio
        audio = inputs["audio"].to(dtype=torch.bfloat16, device=self.device)    # B, L [-1, 1]
        audio_input_latents = self.audio_vae.wrapped_encode(audio).to(dtype=self.torch_dtype, device=self.device)
        outputs["audio_input_latents"] = audio_input_latents # B, L', D

        # 3. sample noises
        video_noise = torch.randn_like(video_input_latents).to(dtype=self.torch_dtype, device=self.device)
        audio_noise = torch.randn_like(audio_input_latents).to(dtype=self.torch_dtype, device=self.device)
        outputs["video_noise"] = video_noise
        outputs["audio_noise"] = audio_noise

        # 4. process prompts
        prompts = inputs["prompts"]   # list of str
        context = self.text_model(prompts, device=self.device)
        outputs["audio_context"] = context
        outputs["video_context"] = context
        
        outputs["video_seq_len"] = video_noise.shape[2] * video_noise.shape[3] * video_noise.shape[4] // (self._patch_size_h * self._patch_size_w)
        outputs["audio_seq_len"] = audio_noise.shape[1]
        outputs["first_frame_is_clean"] = False   # not i2v
        
        # 5. process ip image
        if "ip_image" in inputs:
            ip_image = inputs["ip_image"].to(dtype=torch.bfloat16, device=self.device)    # B, 3, 1, H, W
            ip_image_latents = self.video_vae.wrapped_encode(ip_image).to(dtype=self.torch_dtype, device=self.device)
            outputs["ip_image_latents"] = ip_image_latents

        # 6. process ip audio
        if "ip_audio" in inputs:
            ip_audio = inputs["ip_audio"].to(dtype=torch.bfloat16, device=self.device)    # B, L
            ip_audio_latents = self.audio_vae.wrapped_encode(ip_audio).to(dtype=self.torch_dtype, device=self.device)
            outputs["ip_audio_latents"] = ip_audio_latents

        return outputs

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(
            *args, **kwargs
        )
        if device is not None:
            self.device = device
        if dtype is not None:
            self.torch_dtype = dtype
        super().to(*args, **kwargs)
        return self

    def load_lora(self, module, path, alpha=1):
        loader = GeneralLoRALoader(torch_dtype=self.torch_dtype, device=self.device)
        lora = load_state_dict(path, torch_dtype=self.torch_dtype, device=self.device)
        loader.load(module, lora, alpha=alpha)

    def training_loss(self, **inputs):
        batch_size = inputs["batch_size"]
        max_timestep_boundary = int(
            inputs.get("max_timestep_boundary", 1) * self.scheduler.num_train_timesteps
        )
        min_timestep_boundary = int(
            inputs.get("min_timestep_boundary", 0) * self.scheduler.num_train_timesteps
        )
        timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (batch_size,))
        timestep = self.scheduler.timesteps[timestep_id].to(
            dtype=self.torch_dtype, device=self.device
        )

        inputs["video_latents"] = self.scheduler.add_noise(
            inputs["video_input_latents"], inputs["video_noise"], timestep
        )
        training_video_target = self.scheduler.training_target(
            inputs["video_input_latents"], inputs["video_noise"], timestep
        )
        inputs["audio_latents"] = self.scheduler.add_noise(
            inputs["audio_input_latents"], inputs["audio_noise"], timestep
        )
        training_audio_target = self.scheduler.training_target(
            inputs["audio_input_latents"], inputs["audio_noise"], timestep
        )

        noise_pred_vid, noise_pred_aud = self.model_fn(**inputs, timestep=timestep)

        loss_video = torch.nn.functional.mse_loss(noise_pred_vid.float(), training_video_target.float())
        loss_audio = torch.nn.functional.mse_loss(noise_pred_aud.float(), training_audio_target.float())
        loss_video = loss_video * self.scheduler.training_weight(timestep)
        loss_audio = loss_audio * self.scheduler.training_weight(timestep)
        loss = loss_video + loss_audio
        losses = {"loss": loss, "loss_video": loss_video, "loss_audio": loss_audio}
        return losses

    def enable_vram_management(
        self, num_persistent_param_in_dit=None, vram_limit=None, vram_buffer=0.5
    ):
        self.vram_management_enabled = True
        if num_persistent_param_in_dit is not None:
            vram_limit = None
        else:
            if vram_limit is None:
                vram_limit = self.get_vram()
            vram_limit = vram_limit - vram_buffer
        if self.text_model is not None:
            dtype = next(iter(self.text_model.parameters())).dtype
            enable_vram_management(
                self.text_model,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Embedding: AutoWrappedModule,
                    T5RelativeEmbedding: AutoWrappedModule,
                    T5LayerNorm: AutoWrappedModule,
                },
                module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.model is not None:
            dtype = next(iter(self.model.parameters())).dtype
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.model,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: WanAutoCastLayerNorm,
                    WanRMSNorm: AutoWrappedModule,
                    torch.nn.Conv2d: AutoWrappedModule,
                },
                module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                max_num_param=num_persistent_param_in_dit,
                overflow_module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.audio_vae is not None:
            dtype = next(iter(self.vae.parameters())).dtype
            enable_vram_management(
                self.vae,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    RMS_norm: AutoWrappedModule,
                    CausalConv3d: AutoWrappedModule,
                    Upsample: AutoWrappedModule,
                    torch.nn.SiLU: AutoWrappedModule,
                    torch.nn.Dropout: AutoWrappedModule,
                },
                module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=self.device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
            )
        if self.video_vae is not None:
            dtype = next(iter(self.vae.parameters())).dtype
            enable_vram_management(
                self.vae,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    RMS_norm: AutoWrappedModule,
                    CausalConv3d: AutoWrappedModule,
                    Upsample: AutoWrappedModule,
                    torch.nn.SiLU: AutoWrappedModule,
                    torch.nn.Dropout: AutoWrappedModule,
                },
                module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=self.device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
            )
        
    def initialize_usp(self):
        initialize_sequence_parallel_state(self.config.get("sp_size", 1))


def model_fn(
    model: FusionModel,
    video_latents: torch.Tensor = None,
    audio_latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    video_context: torch.Tensor = None,
    audio_context: torch.Tensor = None,
    video_seq_len: int = None,
    audio_seq_len: int = None,
    ip_image_latents: Optional[torch.Tensor] = None,
    ip_audio_latents: Optional[torch.Tensor] = None,
    **kwargs,
):
    return model(
        vid=video_latents,
        audio=audio_latents,
        t=timestep,
        vid_context=video_context,
        audio_context=audio_context,
        vid_seq_len=video_seq_len,
        audio_seq_len=audio_seq_len,
        clip_fea=None,
        clip_fea_audio=None,
        y=None,
        first_frame_is_clean=False,
        slg_layer=False,
        vid_ip=ip_image_latents,
        audio_ip=ip_audio_latents,
        **kwargs,
    )
    
