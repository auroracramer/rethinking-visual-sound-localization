import glob
import math
import os
import random
import json

import ffmpeg
import h5py
import librosa
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from functools import partial
from PIL import Image
from torch.utils.data import IterableDataset
from torchaudio.io import StreamReader
from torchaudio.utils import ffmpeg_utils
from torchaudio.transforms import Spectrogram
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    Resize,
    ToTensor,
)
from ..audio_utils import (
    get_silence_ratio,
    get_silence_ratio_spectrogram,
    get_stream,
    SpectrogramGcc,
)


try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose(
        [
            Resize(n_px, interpolation=BICUBIC),
            CenterCrop(n_px),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


def spectrogram(audio, n_fft=512, hop_length=353):
    x = Spectrogram(audio.unsqueeze(1), n_fft, hop_length)
    x = torch.log(x + 1e-7)
    x = (x - torch.mean(x)) / (torch.std(x) + 1e-9)
    return x


class AudioVisualDataset(IterableDataset):
    """
        Dataset for loading VGGSound
    """

    def __init__(
            self,
            data_root,
            split: str = "train",
            duration: int = 5,
            sample_rate: int = 16000,
    ):
        super(AudioVisualDataset).__init__()
        self.sample_rate = sample_rate
        self.duration = duration
        self.fps = 30
        self.transform = _transform(224)
        self.preprocess = spectrogram
        self.data_root = data_root

        if split in ("train", "valid"):
            self.split = "train"
            files = self.get_overlapping_files(self.split)
            if split == "train":
                files = files[:-500]
                random.shuffle(files)
            elif split == "valid":
                files = files[-500:]
        elif split == "test":
            self.split = split
            files = self.get_overlapping_files(self.split)
        else:
            assert False
        self.files = files

    def get_overlapping_files(self, split):
        audio_files = glob.glob("{}/{}/audio/*.flac".format(self.data_root, split))
        video_files = glob.glob("{}/{}/video/*.mp4".format(self.data_root, split))
        files = sorted(
            list(
                set([f.split("/")[-1].split(".")[0] for f in audio_files])
                & set([f.split("/")[-1].split(".")[0] for f in video_files])
            )
        )
        return files

    def __iter__(self):
        import skvideo.io
        for f in self.files:
            # TODO: filter stereo audio here
            audio, _ = librosa.load(
                "{}/{}/audio/{}.flac".format(self.data_root, self.split, f),
                sr=self.sample_rate,
            )
            video = skvideo.io.vread(
                "{}/{}/video/{}.mp4".format(self.data_root, self.split, f)
            )
            num_audio_samples = self.duration * self.sample_rate
            num_video_samples = self.duration * self.fps
            if self.duration < 10:
                if (
                        audio.shape[0] >= num_audio_samples  # TODO: change the dimension indicator
                        and video.shape[0] >= num_video_samples  # TODO: why [0] for 3 channels
                ):
                    audio_index = random.randint(0, audio.shape[0] - num_audio_samples)
                    video_index = int(
                        np.floor((audio_index / self.sample_rate) * self.fps)
                    )
                    audio_slice = slice(audio_index, audio_index + num_audio_samples)
                    video_slice = slice(
                        video_index + num_video_samples // 2,
                        video_index + num_video_samples // 2 + 1,
                    )
                    if (
                            audio[audio_slice].shape[0] == num_audio_samples
                            and video[video_slice, :].shape[0] == 1
                    ):
                        yield self.preprocess(audio[audio_slice]), self.transform(
                            Image.fromarray(video[video_slice, :, :, :][0])
                        )
            elif self.duration == 10:
                if (
                        audio.shape[0] == num_audio_samples
                        and video.shape[0] == num_video_samples
                ):
                    yield self.preprocess(audio), self.transform(
                        Image.fromarray(video[video.shape[0] // 2, :, :, :])
                    )
            else:
                assert False


class Ego4DHdf5Dataset(IterableDataset):
    """
        Dataset for loading EGO4D with stereo audio (from HDF5)
    """

    def __init__(
            self,
            data_root,
            split: str = "train",
            duration: int = 5,
            sample_rate: int = 16000,
            chunk_duration: Optional[int] = 10,
            num_channels: int = 3,
            random_seed: int = 1337,
            valid_ratio: float = 0.1,
            silence_threshold: float = 0.1,
    ):
        super(Ego4DHdf5Dataset).__init__()
        # TODO: Revisit a good value of `duration`, and if the embedding
        #       is actually compatible the SAVi AudioCNN. Though we could
        #       do a maxpooling/flatten type of post-processing like with L3
        assert chunk_duration >= duration
        assert num_channels in (1, 2, 3)
        self.sample_rate = sample_rate
        self.duration = duration
        self.fps = 30
        self.num_channels = num_channels
        self.data_root = Path(data_root)
        self.chunk_duration = chunk_duration
        self.split = split
        self.silence_threshold = silence_threshold

        files = self.get_video_files()
        # Set up splits
        random.seed(random_seed)
        random.shuffle(files)
        num_files = len(files)
        num_valid = int(num_files * valid_ratio)
        num_train = num_files - num_valid
        if split == "train":
            start_idx = 0
            end_idx = start_idx + num_train
        elif split == "valid":
            start_idx = num_train
            end_idx = start_idx + num_valid
        else:
            assert False
        self.files = files[start_idx:end_idx]

    def get_video_filepath(self, video_name):
        return os.path.join(self.data_root, f"{video_name}.h5")

    def get_video_files(self):
        return sorted([fpath.stem for fpath in self.data_root.glob("*.h5")])


    def __iter__(self):
        for f in self.files:
            fpath = self.get_video_filepath(f)

            # Check attrs
            try:
                with h5py.File(fpath, 'r') as h5:
                    assert h5.attrs["sample_rate"] == self.sample_rate, f
                    assert h5.attrs["fps"] == self.fps, f
                    assert h5.attrs["audio_nchan"] == self.num_channels, f
                    
                    num_chunks = h5.attrs["num_chunks"]
                    audio_frame_hop = h5.attrs["audio_frame_hop"]
                    num_audio_frames = int(self.duration / audio_frame_hop)
                    num_video_samples = int(self.duration * self.fps)
                    full_duration = h5.attrs["duration"]
                    silent = h5.attrs["silent"]
                    identical_audio_channels = h5.attrs["identical_audio_channels"]
                    silent_chunks = h5.attrs["silent_chunks"]
                    identical_audio_channels_chunks = h5.attrs["identical_audio_channels_chunks"]
                    too_short_chunks = h5.attrs["too_short_chunks"]
                    missing_chunks = h5.attrs["missing_chunks"]
                    num_silent_chunks = len(silent_chunks),
                    num_identical_audio_channels_chunks=len(identical_audio_channels_chunks),
                    num_too_short_chunks=len(too_short_chunks),
                    num_missing_chunks=len(missing_chunks),

                    if (
                        silent
                        or identical_audio_channels
                        or (num_silent_chunks == num_chunks)
                        or (num_identical_audio_channels_chunks == num_chunks)
                        or (num_missing_chunks == num_chunks)
                        or (num_too_short_chunks == num_chunks)
                    ):
                        return


                    if self.duration < full_duration:
                        sample_uniform = np.random.default_rng().uniform
                        # split video into chunks and sample a window from each
                        for chunk_idx in range(num_chunks):
                            start_ts = chunk_idx * self.chunk_duration
                            # sample a random window within the chunk
                            if start_ts + self.chunk_duration <= full_duration:
                                audio_offset = sample_uniform(0.0, self.chunk_duration - self.duration)
                            else:
                                leftover_duration = full_duration - start_ts
                                if leftover_duration >= self.duration:
                                    # if we have at least a window's worth of samples,
                                    # we can still sample a window
                                    audio_offset = sample_uniform(0.0, leftover_duration - self.duration)
                                else:
                                    # ignore windows that are too short
                                    continue

                            audio_ts = start_ts + audio_offset

                            audio_index = int(audio_ts / audio_frame_hop)
                            video_index = int(audio_ts * self.fps) + (num_video_samples // 2)

                            # Read from hdf5
                            audio = torch.tensor(h5["audio"][audio_index:audio_index+num_audio_frames]).clone()
                            video = torch.tensor(h5["video"][video_index]).clone()

                            # audio.shape (Ta, F, C) -> (C, F, Ta)
                            audio = audio.permute(2, 1, 0)
                            # video.shape (D, D, C) -> (C, D, D)
                            video = video.permute(2, 0, 1) 

                            if (
                                (chunk_idx in silent_chunks)
                                or (chunk_idx in too_short_chunks)
                                or (chunk_idx in identical_audio_channels_chunks)
                                or (chunk_idx in missing_chunks)
                            ):
                                continue

                            yield audio, video

                    elif self.duration == full_duration:
                        audio = torch.tensor(h5["audio"][:]).clone()
                        video = torch.tensor(h5["video"][num_video_samples // 2]).clone()
                        # audio.shape (Ta, F, C) -> (C, F, Ta)
                        audio = audio.permute(2, 1, 0)
                        # video.shape (Tv, D, D, C) -> (Tv, C, D, D)
                        video = video.permute(2, 0, 1) 

                        if (
                            silent
                            or (num_silent_chunks == num_chunks)
                            or (num_identical_audio_channels_chunks == num_chunks)
                            or (num_missing_chunks == num_chunks)
                            or (num_too_short_chunks == num_chunks)
                        ):
                            continue

                        yield audio, video

                    else:
                        assert False
            except OSError as e:
                print(f"{f}: {e}")
                raise e

def _video_to_float_tensor(x):
    if torch.is_floating_point(x):
        return x
    else:
        return x / 255.0


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    files = dataset.files
    # configure the dataset to only process the split workload
    per_worker = int(math.ceil((len(files)) / float(worker_info.num_workers)))
    worker_id = worker_info.id
    if getattr(dataset, "file_stats", None):
        fs = dataset.file_stats
        dataset.files = sorted(
            dataset.files, key=lambda x: fs[x]["num_valid_chunks"] / fs[x]["num_chunks"]
        )[worker_id::per_worker]
    else:
        dataset.files = files[
            worker_id * per_worker : min(worker_id * per_worker + per_worker, len(files))
        ]
