import glob
import math
import os
import random

import h5py
import librosa
import numpy as np
import skvideo.io
import torch
from pathlib import Path
from typing import Optional
from operator import itemgetter
from itertools import groupby
from torch.utils.data import IterableDataset
from torchaudio import Spectrogram
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    Resize,
    ToTensor,
    ToPILImage,
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
            ToPILImage(mode="RGB"),
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
                            video[video_slice, :, :, :][0]
                        )
            elif self.duration == 10:
                if (
                        audio.shape[0] == num_audio_samples
                        and video.shape[0] == num_video_samples
                ):
                    yield self.preprocess(audio), self.transform(
                        video[video.shape[0] // 2, :, :, :]
                    )
            else:
                assert False


class Ego4DDataset(IterableDataset):
    """
        Dataset for loading EGO4D with stereo audio
    """

    def __init__(
            self,
            data_root,
            split: str = "train",
            duration: int = 5,
            sample_rate: int = 16000,
            chunk_duration: Optional[int] = 10,
            num_channels: int = 2,
            random_seed: int = 1337,
            valid_ratio: float = 0.1,
            silence_threshold: float = 0.1,
    ):
        super(Ego4DDataset).__init__()
        # TODO: Revisit a good value of `duration`, and if the embedding
        #       is actually compatible the SAVi AudioCNN. Though we could
        #       do a maxpooling/flatten type of post-processing like with L3
        assert chunk_duration >= duration
        assert num_channels in (1, 2)
        self.sample_rate = sample_rate
        self.duration = duration
        self.fps = 30
        self.transform = _transform(224)
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

    @staticmethod
    def get_silence_ratio(signal):
        # Get overall silence ratio (this is faster, but not exactly what we want)
        # return (signal == 0).to(dtype=torch.float32).mean()

        # Get the ratio of longest contiguous silence to the whole signal
        # For simplicity, we're only considering digital silence
        # https://stackoverflow.com/a/58920786
        

        # audio.shape (F, Ta)
        nfreq, ntime = signal.shape
        signal = signal.permute(1, 0)
        return max(
            (
                len([i for i, _ in group])
                for is_zero, group in groupby(
                    enumerate(
                        # Since we're using spectrogram, check that the decibels
                        # are at the low end of the dynamic range
                        ((signal <= -80.0).sum(axis=-1) == nfreq).tolist()
                    ),
                    key=itemgetter(1),
                )
                if not is_zero
            ),
            default=0,
        ) / float(ntime)

    def __iter__(self):
        for f in self.files:
            fpath = self.get_video_filepath(f)
            num_audio_samples = self.duration * self.sample_rate
            num_video_samples = self.duration * self.fps

            # Check attrs
            with h5py.open(fpath, 'r') as h5:
                assert h5.attrs["sample_rate"] == self.sample_rate
                assert h5.attrs["fps"] == self.fps
                assert h5.attrs["audio_nchan"] == self.num_channels
                audio_frame_hop = h5.attrs["audio_frame_hop"]
                full_duration = h5.attrs["duration"]
                num_audio_frames = int(num_audio_samples / audio_frame_hop) + 1

                if self.duration < full_duration:
                    sample_uniform = np.random.default_rng().uniform
                    # split video into chunks and sample a window from each
                    for start_ts in np.arange(0.0, full_duration, step=self.chunk_duration):
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
                        video_index = audio_index + (num_video_samples // 2)

                        # Read from hdf5
                        audio = torch.tensor(h5["audio"][audio_index:audio_index+num_audio_frames])
                        video = torch.tensor(h5["audio"][video_index])

                        # audio.shape (Ta, F, C) -> (C, F, Ta)
                        audio = audio.permute(2, 1, 0)
                        # video.shape (Tv, D, D, C) -> (Tv, C, D, D)
                        video = video.permute(0, 3, 1, 2) 

                        # If both channels are the same, skip
                        if torch.allclose(audio[0], audio[1]):
                            continue

                        # If either channel has too much digital silence, skip
                        silence_ratio_0 = self.get_silence_ratio(audio[0])
                        silence_ratio_1 = self.get_silence_ratio(audio[1])
                        # If the longest silence is more than the threshold, skip
                        if max(silence_ratio_0, silence_ratio_1) >= self.silence_threshold:
                            continue

                        yield audio, video

                elif self.duration == full_duration:
                    audio = h5["audio"][:]
                    video = h5["video"][num_video_samples // 2]
                    # audio.shape (Ta, F, C) -> (C, F, Ta)
                    audio = audio.permute(2, 1, 0)
                    # video.shape (Tv, D, D, C) -> (Tv, C, D, D)
                    video = video.permute(0, 3, 1, 2) 

                    # If both channels are the same, skip
                    if torch.allclose(audio[0], audio[1]):
                        return

                    # If either channel has too much digital silence, skip
                    silence_ratio_0 = self.get_silence_ratio(audio[0])
                    silence_ratio_1 = self.get_silence_ratio(audio[1])
                    # If the longest silence is more than the threshold, skip
                    if max(silence_ratio_0, silence_ratio_1) >= self.silence_threshold:
                        return

                    yield audio, video

                else:
                    assert False


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    files = dataset.files
    # configure the dataset to only process the split workload
    per_worker = int(math.ceil((len(files)) / float(worker_info.num_workers)))
    worker_id = worker_info.id
    dataset.files = files[
                    worker_id * per_worker: min(worker_id * per_worker + per_worker, len(files))
                    ]
