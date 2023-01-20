import glob
import math
import os
import random

import librosa
import numpy as np
import skvideo.io
import torch
import ffmpeg
from pathlib import Path
from typing import Optional
from PIL import Image
from operator import itemgetter
from itertools import groupby
from torch.utils.data import IterableDataset
from torchaudio.io import StreamReader
from torchaudio.transforms import Spectrogram
from torchvision.transforms import CenterCrop
from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor

from ..audio_utils import SpectrogramGcc, read_mp4_audio_ffmpeg, read_mp4_video_ffmpeg

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
        self.preprocess = (
            SpectrogramGcc(self.sample_rate, self.duration)
            if (self.num_channels == 2) else spectrogram
        )
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
        return os.path.join(self.data_root, f"{video_name}.mp4")

    def get_video_files(self):
        file_list = []
        for fpath in self.data_root.glob("*.mp4"):
            metadata = ffmpeg.probe(str(fpath))
            num_channels = None
            for stream in metadata["streams"]:
                if stream["codec_type"] == "audio":
                    if num_channels is not None:
                        raise ValueError(
                            f"Found more than one audio stream for {str(fpath)}"
            )
                    num_channels = int(stream["channels"])
            if num_channels == self.num_channels:
                file_list.append(fpath.stem)
        # Return in sorted order for reproducibility
        return sorted(file_list)

    @staticmethod
    def get_silence_ratio(signal):
        # Get overall silence ratio (this is faster, but not exactly what we want)
        # return (signal == 0).to(dtype=torch.float32).mean()

        # Get the ratio of longest contiguous silence to the whole signal
        # For simplicity, we're only considering digital silence
        # https://stackoverflow.com/a/58920786
        return max(
            (
                len([i for i, _ in group])
                for is_zero, group in groupby(
                    enumerate((signal == 0.0).tolist()),
                    key=itemgetter(1),
                )
                if not is_zero
            ),
            default=0,
        ) / float(signal.shape[-1])

    def __iter__(self):
        for f in self.files:
            fpath = self.get_video_filepath(f)
            probe = ffmpeg.probe(fpath)

            num_audio_samples = self.duration * self.sample_rate
            num_video_samples = self.duration * self.fps
            # Take minimum duration of streams
            full_duration = min(float(stream['duration']) for stream in probe['streams'])

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

                    # Stream file to only load the relevant chunk at at time
                    # NOTE: We're encountering lots of OOMs despite streaming,
                    #       so it may be the case that it has to keep the 
                    #       decoder state in memory which maybe is a lot.
                    #       So instead of keeping the file open and seeking
                    #       in order, we'll just reopen the file and reseek for
                    #       each time. It's slower but uses less memory at least
                    with open(fpath, 'rb') as vfile:
                        streamer = StreamReader(vfile)
                        num_channels = streamer.get_src_stream_info(
                            streamer.default_audio_stream
                        ).num_channels
                        assert num_channels == 2, (
                            f"{fpath} must have two audio channels"
                        )

                        # add output streams
                        streamer.add_basic_audio_stream(
                            frames_per_chunk=num_audio_samples,
                            sample_rate=self.sample_rate,
                            format='fltp',
                            decoder_option={
                                "threads": "1",
                            }
                        )
                        streamer.add_basic_video_stream(
                            frames_per_chunk=num_video_samples,
                            frame_rate=self.fps,
                            format="rgb24",
                            decoder_option={
                                "threads": "1",
                            }
                        )

                        ## Extract the sampled window/frame for each time
                        # Seek to start of audio window
                        streamer.seek(audio_ts)
                        # Get corresponding audio and video window
                        stream = streamer.stream()
                        # audio.shape = (frames, channels)
                        # video.shape = (frames, channels, height, width)
                        audio, video = next(stream)
                        # Reseek to start to avoid ffmpeg decoding in the background
                        # https://github.com/dmlc/decord/issues/208#issuecomment-1157632702
                        streamer.seek(0, mode="key")

                    streamer = stream = None # help out GC with clearing iterator
                    assert audio.shape == (num_audio_samples, 2)
                    assert video.shape[:2] == (num_video_samples, 3)

                    # If both channels are the same, skip
                    if not torch.allclose(audio[0], audio[1]):
                        continue

                    # If either channel has too much digital silence, skip
                    silence_ratio_0 = self.get_silence_ratio(audio[0])
                    silence_ratio_1 = self.get_silence_ratio(audio[1])
                    # If the longest silence is more than the threshold, skip
                    if max(silence_ratio_0, silence_ratio_1) >= self.silence_threshold:
                        continue

                    # Get center frame of video
                    video = video[video.shape[0] // 2]

                    yield self.preprocess(audio), self.transform(video)

            elif self.duration == full_duration:
                # We can just load the whole ding dang thing
                # NOTE: Right now I'm assuming that we probably won't hit this case
                #       and if we do, that loading 5 seconds of audio and then
                #       5 seconds of video won't be a big deal. But if that
                #       isn't true, you'll need to replace this with stream
                #       version
                video = read_mp4_video_ffmpeg(fpath, probe, frame_rate=self.fps)
                audio = read_mp4_audio_ffmpeg(fpath, probe, sample_rate=self.sample_rate)
                assert audio.shape == (num_audio_samples, 2)
                assert video.shape[:2] == (num_video_samples, 3)
                # Get center frame of video
                video = video[video.shape[0] // 2, :, :, :]

                yield self.preprocess(audio), self.transform(video)
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
