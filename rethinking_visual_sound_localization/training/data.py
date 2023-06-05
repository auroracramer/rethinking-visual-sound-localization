import glob
import math
import os
import random
import json

import ffmpeg
import h5py
import librosa
import numpy as np
import skvideo.io
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
            num_channels: int = 2,
            random_seed: int = 1337,
            valid_ratio: float = 0.1,
            silence_threshold: float = 0.1,
    ):
        super(Ego4DHdf5Dataset).__init__()
        # TODO: Revisit a good value of `duration`, and if the embedding
        #       is actually compatible the SAVi AudioCNN. Though we could
        #       do a maxpooling/flatten type of post-processing like with L3
        assert chunk_duration >= duration
        assert num_channels in (1, 2)
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
                        silence_ratio_0 = get_silence_ratio_spectrogram(audio[0])
                        silence_ratio_1 = get_silence_ratio_spectrogram(audio[1])
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
                    silence_ratio_0 = get_silence_ratio_spectrogram(audio[0])
                    silence_ratio_1 = get_silence_ratio_spectrogram(audio[1])
                    # If the longest silence is more than the threshold, skip
                    if max(silence_ratio_0, silence_ratio_1) >= self.silence_threshold:
                        return

                    yield audio, video

                else:
                    assert False


class Ego4DDataset(IterableDataset):
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
            num_channels: int = 2,
            random_seed: int = 1337,
            valid_ratio: float = 0.1,
            silence_threshold: float = 0.1,
            files: Optional[List[str]] = None,
            file_stats: Optional[Dict[str, Dict[str, Any]]] = None,
            job_idx=None,
            num_jobs=None,
            project_root: Optional[str] = None,
    ):
        super(Ego4DDataset).__init__()
        # TODO: Revisit a good value of `duration`, and if the embedding
        #       is actually compatible the SAVi AudioCNN. Though we could
        #       do a maxpooling/flatten type of post-processing like with L3
        assert chunk_duration >= duration
        assert num_channels == 2
        self.sample_rate = sample_rate
        self.duration = duration
        self.fps = 30
        self.image_dim = 128
        self.video_transform = Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        )
        self.image_feature_shape = (3, self.image_dim, self.image_dim)
        self.spec_tf = SpectrogramGcc(self.sample_rate, self.duration)
        self.audio_transform = partial(self.spec_tf.forward, center=True, time_first=False)
        self.num_channels = num_channels
        self.data_root = Path(data_root)
        self.chunk_duration = chunk_duration
        self.split = split
        self.silence_threshold = silence_threshold
        self.device = 'cpu'
        self.num_retry_silence = 3
        assert self.chunk_duration >= self.duration
        self.ignore_files = set()
        self.ignore_segments = {}
        if file_stats:
            for fname, info in file_stats.items():
                if info["too_short"] or info["silent"] or info["duplicate_channels"]:
                    self.ignore_files.add(fname)
                self.ignore_segments[fname] = set(
                    info["chunk_idxs_silent"]
                    + info["chunk_idxs_duplicate_channels"] 
                    + info["chunk_idxs_too_short"]
                    + info["chunk_idxs_failed"]
                )
        if project_root is not None:
            self.project_root = Path(project_root)
            self.project_root.joinpath("video_info").mkdir(parents=True, exist_ok=True)
        else:
            self.project_root = None

        if split != "full":
            files = files or self.get_video_files()
            # Set up splits
            self.rng = np.random.default_rng(random_seed)
            self.rng.shuffle(files)
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
            self.scan = False
        elif split == "full" and not files:
            self.scan = True
            assert self.project_root is not None
            self.files = self.get_video_files()
            self.rng = np.random.default_rng(random_seed)
            self.rng.shuffle(self.files)
        else:
            self.files = files
            self.scan = False

        if num_jobs:
            assert job_idx is not None
            files_per_job = len(self.files) // num_jobs
            start_idx = files_per_job * job_idx
            end_idx = min(start_idx + files_per_job, len(self.files))
            self.files = self.files[start_idx:end_idx]

        if not self.ignore_segments:
            self.ignore_segments = {fname: set() for fname in self.files}
        else:
            # Only include ignore_segments defined in files
            self.ignore_segments = {fname: self.ignore_segments[fname] for fname in self.files}
        if self.ignore_files:
            self.ignore_files = set(fname for fname in self.files if fname in self.ignore_files)

    def get_video_filepath(self, video_name):
        return os.path.join(self.data_root, f"{video_name}.mp4")

    def get_video_files(self):
        file_list = []
        for fpath in self.data_root.glob("*.mp4"):
            if fpath.stem in self.ignore_files:
                continue
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

    def create_stream_reader(
        self, vfile, num_chunk_audio_samples,
        num_chunk_video_frames, video_width, video_height, video_codec,
    ):
        streamer = StreamReader(vfile)
        num_channels = streamer.get_src_stream_info(
            streamer.default_audio_stream
        ).num_channels
        assert num_channels == self.num_channels, (
            f"{vfile.name} must have two audio channels"
        )

        # add output streams
        audio_filter_desc = ",".join(
            [
                f"aresample={self.sample_rate}",
                f"aformat=sample_fmts=fltp",
            ]
        )
        #print(f"audio_filter_desc: {audio_filter_desc}")
        streamer.add_audio_stream(
            frames_per_chunk=num_chunk_audio_samples,
            buffer_chunk_size=5,
            decoder_option={
                "threads": "1",
            },
            filter_desc=audio_filter_desc,
        )
        if torch.cuda.is_available():
            video_decoder = f"{video_codec}_cuvid"
            video_hw_accel = f"cuda:{torch.cuda.current_device()}"
        else:
            video_decoder = video_codec
            video_hw_accel = None
        # Determine height in Python to avoid quoted sections
        # in filter descriptions
        if video_width > video_height:
            new_video_width = -1
            new_video_height = int(self.image_dim)
        else:
            new_video_width = int(self.image_dim)
            new_video_height = -1

        video_filter_desc = ",".join(
            [
                f"scale=width={new_video_width}:height={new_video_height}",
                f"crop={self.image_dim}:{self.image_dim}:exact=1",
                f"fps={self.fps}",
                f"format=pix_fmts=rgb24",
            ]
        )
        #print(f"video_decoder: {video_decoder}")
        #print(f"video_hw_accel: {video_hw_accel}")
        #print(f"video_filter_desc: {video_filter_desc}")
        streamer.add_video_stream(
            frames_per_chunk=num_chunk_video_frames,
            buffer_chunk_size=5,
            decoder=video_decoder,
            hw_accel=video_hw_accel,
            filter_desc=video_filter_desc,
        )
        # Seek to start to avoid ffmpeg decoding in the background
        # https://github.com/dmlc/decord/issues/208#issuecomment-1157632702
        streamer.seek(0)
        return streamer

    def sample_offset_ts(self, start_ts, end_ts):
        duration = (end_ts - start_ts) - self.duration
        return self.rng.uniform(0.0, duration)

    def __iter__(self):
        for f in self.files:
            if f in self.ignore_files:
                continue
            fpath = self.get_video_filepath(f)
            probe = ffmpeg.probe(fpath)
            audio_probe = get_stream(probe, "audio")
            video_probe = get_stream(probe, "video")
            audio_duration = float(audio_probe["duration"])
            video_duration = float(video_probe["duration"])
            video_width = int(video_probe["width"])
            video_height = int(video_probe["height"])
            video_codec = video_probe["codec_name"]
            full_duration = max(audio_duration, video_duration)
            if self.duration > full_duration:
                # Don't need to add if we're scanning ahead of time
                self.ignore_files.add(f)
                print(
                    f"WARNING: "
                    f"video '{Path(fpath).name}' (duration: {full_duration} "
                    f"seconds) is too short (less than {self.duration} seconds). "
                    f"Video will be skipped for sampling."
                )
                if self.scan:
                    with self.project_root.joinpath("video_info", f"{f}.json").open("w") as fh:
                        json.dump({
                            "too_short": True,
                            "silent": True,
                            "duplicate_channels": True,
                            "audio_duration": audio_duration,
                            "video_duration": video_duration,
                            "full_duration": full_duration,
                            "num_chunks": 0,
                            "num_valid_chunks": 0,
                            "num_chunks_missing": 0,
                            "num_chunks_silent": 0,
                            "num_chunks_duplicate_channels": 0,
                            "num_chunks_ignore": 0,
                            "num_chunks_short": 0,
                            "num_chunks_failed": 0,
                            "chunk_idxs_silent": [],
                            "chunk_idxs_duplicate_channels": [],
                            "chunk_idxs_too_short": [],
                            "chunk_idxs_failed": [],
                        }, fh)
                continue

            num_audio_samples = self.duration * self.sample_rate
            num_video_samples = self.duration * self.fps
            num_chunk_audio_samples = self.chunk_duration * self.sample_rate
            num_chunk_video_frames = self.chunk_duration * self.fps

            num_chunks = 0
            num_missing_chunks = 0
            num_silent_chunks = 0
            num_dupe_chunks = 0
            num_ignore_chunks = 0
            num_short_chunks = 0
            num_failed_chunks = 0
            num_valid_chunks = 0

            with open(fpath, 'rb') as vfile:
                streamer = self.create_stream_reader(
                    vfile,
                    num_chunk_audio_samples,
                    num_chunk_video_frames,
                    video_width,
                    video_height,
                    video_codec,
                )
                silent = True
                dupe_channels = True
                silent_chunks = set()
                dupe_chunks = set()
                too_short_chunks = set()
                failed_chunks = set()
                # split video into chunks and sample a window from each
                for chunk_idx, (audio, video) in enumerate(streamer.stream()):
                    num_chunks += 1
                    if audio is None or video is None:
                        num_missing_chunks += 1
                        continue
                    if chunk_idx in self.ignore_segments[f]:
                        num_ignore_chunks += 1
                        continue

                    # audio.shape = (C, T)
                    audio = audio.to(device=self.device).transpose(0, 1) # put channels first

                    # Get chunk boundaries
                    start_ts = float(self.chunk_duration * chunk_idx)
                    end_ts = float(min(start_ts + self.chunk_duration, full_duration))
                    end_ts_audio = start_ts + audio.shape[-1] / self.sample_rate

                    # Shape sanity checks
                    assert audio.shape[0] == self.num_channels

                    # Skip chunk if channels are the same
                    if torch.allclose(audio[0], audio[1]):
                        num_dupe_chunks += 1
                        dupe_chunks.add(chunk_idx)
                        print(
                            f"WARNING: "
                            f"video '{Path(fpath).name}': "
                            f"[{float(start_ts)} - {end_ts}] has duplicate "
                            f"audio channels. Skipping..."
                        )
                        dupe_channels = dupe_channels and True
                        continue
                    else:
                        dupe_channels = False

                    chunk_silence_ratio = max(get_silence_ratio(ch) for ch in audio)
                    # Skip chunk if it is silent
                    if np.isclose(chunk_silence_ratio, 1):
                        num_silent_chunks += 1
                        silent_chunks.add(chunk_idx)
                        print(
                            f"WARNING: "
                            f"video '{Path(fpath).name}': "
                            f"[{float(start_ts)} - {end_ts}] is silent. "
                            f"Skipping..."
                        )
                        silent = silent and True
                        continue
                    else:
                        silent = False
                    # Warn if above threshold
                    if chunk_silence_ratio >= self.silence_threshold:
                        print(
                            f"WARNING: "
                            f"video '{Path(fpath).name}': "
                            f"[{float(start_ts)} - {end_ts}] contains more than "
                            f"{int(self.silence_threshold * 100)}% digital silence"
                        )

                    if self.duration > (min(end_ts, end_ts_audio) - start_ts):
                        too_short_chunks.add(chunk_idx)
                        # We don't have enough for a full window, so skip
                        num_short_chunks += 1
                        continue

                    # sample a random window within the chunk
                    for _ in range(self.num_retry_silence):
                        # Sample a start time relative to start of the chunk (w.r.t audio)
                        offset_ts = self.sample_offset_ts(start_ts, end_ts_audio)

                        # Get corresponding indices for audio and video data
                        audio_index = int(offset_ts * self.sample_rate)
                        video_index = int(offset_ts * self.fps) + (num_video_samples // 2)

                        # Make sure we don't exceed length of chunk audio/video
                        if audio_index + num_audio_samples > audio.shape[-1]:
                            continue
                        if video_index >= video.shape[0]:
                            continue

                        silence_ratio = max(
                            get_silence_ratio(
                                ch[audio_index:audio_index+num_audio_samples]
                            )
                            for ch in audio
                        )


                        # Check to make sure sampled slice is not silent
                        if not np.isclose(silence_ratio, 1):
                            audio = audio[:, audio_index:audio_index+num_audio_samples]
                            break
                    else:
                        failed_chunks.add(chunk_idx)
                        num_failed_chunks += 1
                        print(
                            f"WARNING: "
                            f"video '{Path(fpath).name}': "
                            f"[{float(start_ts)} - {end_ts}] could not be sampled from "
                            f"in {self.num_retry_silence} attempts. Skipping chunk..."
                        )
                        continue

                    # audio.shape (C, F, Ta)
                    audio = self.audio_transform(audio)
                    # video.shape (C, D, D)
                    video = self.video_transform(video[video_index].permute(1, 2, 0))
                    num_valid_chunks += 1
                    yield audio, video

            # Mark files that have issues throughout the whole file as
            # to be ignored 
            if silent:
                print(
                    f"WARNING: "
                    f"video '{Path(fpath).name}' is silent. "
                    f"Video will be skipped for sampling."
                )
            if dupe_channels:
                print(
                    f"WARNING: "
                    f"video '{Path(fpath).name}' has duplicate channels. "
                    f"Video will be skipped for sampling."
                )
            
            if self.scan:
                print(
                    f"File Chunk Summary: {Path(fpath).name} "
                    f"| total: {num_chunks} "
                    f"| valid: {num_valid_chunks} "
                    f"| missing: {num_missing_chunks} "
                    f"| silent: {num_silent_chunks} "
                    f"| dupe: {num_dupe_chunks} "
                    f"| ignore: {num_ignore_chunks} "
                    f"| short: {num_short_chunks} "
                    f"| failed: {num_failed_chunks} "
                )
                with self.project_root.joinpath("video_info", f"{f}.json").open("w") as fh:
                    json.dump({
                        "too_short": False,
                        "silent": silent,
                        "duplicate_channels": dupe_channels,
                        "audio_duration": audio_duration,
                        "video_duration": video_duration,
                        "full_duration": full_duration,
                        "num_chunks": num_chunks,
                        "num_valid_chunks": num_valid_chunks,
                        "num_chunks_missing": num_missing_chunks,
                        "num_chunks_silent": num_silent_chunks,
                        "num_chunks_duplicate_channels": num_dupe_chunks,
                        "num_chunks_ignore": num_ignore_chunks,
                        "num_chunks_short": num_short_chunks,
                        "num_chunks_failed": num_failed_chunks,
                        "chunk_idxs_silent": list(sorted(silent_chunks)),
                        "chunk_idxs_duplicate_channels": list(sorted(dupe_chunks)),
                        "chunk_idxs_too_short": list(sorted(too_short_chunks)),
                        "chunk_idxs_failed": list(sorted(failed_chunks)),
                    }, fh)


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
