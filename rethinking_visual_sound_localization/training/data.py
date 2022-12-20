import glob
import math
import random

import librosa
import numpy as np
import skvideo.io
import torch
from PIL import Image
from torch.utils.data import IterableDataset
from torchaudio.transforms import Spectrogram
from torchvision.transforms import CenterCrop
from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor

from ..audio_utils import SpectrogramGcc

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


class Ego4dDataset(IterableDataset):
    """
        Dataset for loading EGO4D with stereo audio
    """

    # TODO: make a similar one to vgg
    def __init__(
            self,
            config,
            data_root,
            # split: str = "train",
            duration: int = 5,
            sample_rate: int = 16000,
    ):
        super(AudioVisualDataset).__init__()
        self.config = config
        self.sample_rate = sample_rate
        self.duration = duration
        self.fps = 30
        self.transform = _transform(224)
        self.preprocess = spectrogram if not self.config.STEREO else SpectrogramGcc(self.config)
        self.data_root = data_root

        # TODO: load files
        files = self.get_overlapping_files(self.split)
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
                        and video.shape[0] >= num_video_samples
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
