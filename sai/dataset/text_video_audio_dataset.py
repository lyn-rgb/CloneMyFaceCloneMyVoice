import gc
import os
import os.path as osp

import math
import numpy as np
import cv2
import pandas as pd
import librosa
import random

import torch  # NOTE, import torch before decord to avoid bug occurs in decord
import decord
decord.bridge.set_bridge("torch")    # Read to torch.Tensor directly

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from . import video_transforms


class TextAudioVideoDataset(Dataset):
    """ NOTE, only supports batch size = 1 yet.
    """
    def __init__(self, data_root, meta_dir, audio_sr=16000, normalize_audio=True, target_fps=24, height=480, width=864, height_div=32, width_div=32, num_frames=81):
        super().__init__()
        self.data_root = data_root
        self.data = self._load_data(meta_dir)

        self.audio_sr = audio_sr
        self.normalize_audio = normalize_audio
        self.target_fps = target_fps
        self.num_pixels = height * width
        self.height_div = height_div
        self.width_div = width_div
        self.num_frames = num_frames


    def _load_data(self, meta_dir):
        meta_names = [meta_name for meta_name in os.listdir(meta_dir) if meta_name.endswith("csv")]
        data = []
        for meta_name in meta_names:
            meta_path = osp.join(meta_dir, meta_name)
            pd_data = pd.read_csv(meta_path)
            data.extend([pd_data.iloc[i].to_dict() for i in range(len(pd_data))])
        return data
    
    def load_video(self, video_path, bbox=None):
        video_reader = decord.VideoReader(video_path)
        source_fps = video_reader.get_avg_fps()
        frame_index_delta = source_fps / self.target_fps    # 48 / 24 = 2
        video_length = len(video_reader)                    # 240

        # random start frame index
        total_target_frames = math.floor(video_length / frame_index_delta)    # 120
        target_start_idx = np.random.randint(0, max(0, total_target_frames - self.num_frames))
        source_start_idx = int(target_start_idx * frame_index_delta)
        # TODO, padding if source video are not long enough
        
        # calculate frame indices
        frame_ids = [source_start_idx]
        for i in range(1, self.num_frames):
            frame_id = source_start_idx + int(i * frame_index_delta)
            frame_ids.append(frame_id)
        
        frame_ids = np.array(frame_ids, dtype=np.int32)
        video = video_reader.get_batch(frame_ids).permute(0, 3, 1, 2)   # T,C,H,W

        # random a frame for ip_image
        ref_id = np.random.randint(0, video_length)
        ref_id = np.array([ref_id], dtype=np.int32)
        ref_image = video_reader.get_batch(ref_id).premute(0, 3, 1, 2)  # 1,C,H,W
        # TODO, crop ref_frame from bbox
        if bbox is not None:
            pass

        # prcess video
        _, _, oh, ow = video.shape
        ratio = math.sqrt(self.num_pixels / oh * ow)
        rh, rw = int(oh * ratio), int(ow * ratio)    # resize to
        ch = (rh // self.height_div) * self.height_div   # crop to
        cw = (rw // self.width_div) * self.width_div

        video_transform = transforms.Compose(
            [
                video_transforms.ToTensorVideo(),  # -> T,C,H,W, [0, 1]
                video_transforms.ResizeVideo((rh, rw), interpolation_mode="bilinear"),
                video_transforms.CenterCropVideo((ch, cw)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ]
        )
        video = video_transform(video)

        # process reference image
        _, _, ref_oh, ref_ow = ref_image.shape
        ref_rh, ref_rw = int(ref_oh * ratio), int(ref_ow * ratio)    # resize to
        if ref_rh * ref_rw < 256 * 256:                              # make sure reference image is large enougth
            ratio = math.sqrt(256 * 256 / ref_oh * ref_ow)
            ref_rh, ref_rw = int(ref_oh * ratio), int(ref_ow * ratio)    # resize to
    
        ref_ch = (ref_rh // self.height_div) * self.height_div   # crop to
        ref_cw = (ref_rw // self.width_div) * self.width_div

        image_transform = transforms.Compose(
            [
                video_transforms.ToTensorVideo(),  # -> T,C,H,W, [0, 1]
                video_transforms.ResizeVideo((ref_rh, ref_rw), interpolation_mode="bilinear"),
                video_transforms.CenterCropVideo((ref_ch, ref_cw)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ]
        )
        ref_image = image_transform(ref_image)

        return video, ref_image, source_start_idx, frame_index_delta

    def load_audio(self, audio_path):
        wave_data, sample_rate = librosa.load(audio_path, sr=self.audio_sr, mono=True)
        if self.normalize_audio:
            wave_data = wave_data / (np.max(np.abs(wave_data)) * 0.95 + 1e-6)  # align with mmaudio/data/extraction.wav_dataset.py line 89
        return torch.from_numpy(wave_data) 

    def sample_data(self, sample):
        video_path = sample["video_path"]
        audio_path = sample["audio_path"]
        bbox = sample["bbox"]

        audio = self.load_audio(audio_path)
        video_reader, frame_index_delta, video_length = self.load_video(video_path)

        


        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        return len(self.data)
