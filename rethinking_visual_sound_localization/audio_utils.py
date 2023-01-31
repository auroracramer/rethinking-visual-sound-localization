"""

Reused and modified from https://github.com/auroracramer/hear-baseline
This script prepares audio features for audio encoders.

"""

from typing import Optional

import math
import numpy as np
import torch
import ffmpeg
from itertools import groupby
from operator import itemgetter
from torchaudio.functional import amplitude_to_DB, melscale_fbanks
from ffmpeg import Error as FFmpegError


class SpectrogramGcc(torch.nn.Module):
    r"""Create a spectrogram+gcc feature from a single or a batch of multi-channel audio in shape (..., time).

    Args:
        win_length (int or None, optional): Window size. (Default: ``n_fft``)
        hop_length (int or None, optional): Length of hop between STFT windows.
        n_mels (int): Number of mel filterbanks
        n_fft (int, optional): Size of FFT, creates ``n_fft // 2 + 1`` bins.
        include_gcc_phat (bool) : Whether to concatenate gcc phat after spectrogram
        window: (Callable[..., Tensor], optional): A function to create a window tensor
            that is applied/multiplied to each frame/window. (Default: ``torch.hann_window``)
        mel_scale : Triangular filter banks (fb matrix) of size (``n_freqs``, ``n_mels``)

    """
    _hop_size_ms = 20
    _win_size_ms = 40
    _n_mels = 64
    _include_gcc_phat = True

    def __init__(self, sample_rate, duration, device="cpu") -> None:
        super(SpectrogramGcc, self).__init__()
        self._sample_rate = sample_rate
        self._hop_length = int(self._sample_rate * (self._hop_size_ms / 1000))
        self._win_length = int(self._sample_rate * (self._win_size_ms / 1000))
        self._n_fft = int(self.next_greater_power_of_2(self._win_length))
        self._num_samples = int(duration * self._sample_rate)
        if device:
            self._device = device
        else:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.register_buffer(
            "_window",
            torch.hann_window(self._win_length, device=self._device).to(dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_mel_scale",
            melscale_fbanks(
                n_freqs=(self._n_fft // 2) + 1,
                f_min=0,
                f_max=self._sample_rate / 2,  # nyquist
                n_mels=self._n_mels,
                sample_rate=self._sample_rate,
                mel_scale="htk",
                norm=None,
            ).to(device=self._device, dtype=torch.float32),
            persistent=False,
        )
        self.feature_shape = tuple(self.forward(torch.ones(2, self._num_samples)).shape)

    def forward(self, waveform, center: bool = True, time_first: bool = False):
        return self.compute_spectrogram(
            waveform,
            win_length=self._win_length,
            hop_length=self._hop_length,
            n_fft=self._n_fft,
            n_mels=self._n_mels,
            window=self._window,
            mel_scale=self._mel_scale,
            include_gcc_phat=self._include_gcc_phat,
            center=center,
            time_first=time_first,
        )

    @staticmethod
    def next_greater_power_of_2(x):
        return 2 ** (x - 1).bit_length()

    def compute_spectrogram(
            self, audio_data, win_length: int, hop_length: int, n_fft: int, n_mels: int,
            window: Optional[torch.Tensor], mel_scale: Optional[torch.Tensor],
            include_gcc_phat: bool,
            center: bool = True,
            time_first: bool = False,
    ):
        # multichannel stft returns (..., F, T)
        stft = torch.stft(
                    input=audio_data.to(device=self._device, dtype=torch.float32),
                    win_length=win_length,
                    hop_length=hop_length,
                    n_fft=n_fft,
                    center=center,
                    window=(
                        window if window is not None
                        else torch.hann_window(win_length, device=self._device)
                    ),
                    pad_mode="constant",  # constant for zero padding
                    return_complex=True,
                ).transpose(-1, -2)
        # Compute power spectrogram
        spectrogram = (torch.abs(stft) ** 2.0).to(dtype=torch.float32)
        # Apply the mel-scale filter to the power spectrogram
        if mel_scale is not None:
            spectrogram = torch.matmul(spectrogram, mel_scale)
        # Convert to decibels
        spectrogram = amplitude_to_DB(
            spectrogram,
            multiplier=20.0,
            amin=1e-10,
            db_multiplier=0.0,
            top_db=80,
        )

        if include_gcc_phat:
            num_channels = spectrogram.shape[0]
            n_freqs = n_mels if (mel_scale is not None) else ((n_fft // 2) + 1)
            # compute gcc_phat : (comb, T, F)
            out_list = []
            for ch1 in range(num_channels - 1):
                for ch2 in range(ch1 + 1, num_channels):
                    x1 = stft[ch1]
                    x2 = stft[ch2]
                    xcc = torch.angle(x1 * torch.conj(x2))
                    xcc = torch.exp(1j * xcc.type(torch.complex64))
                    gcc_phat = torch.fft.irfft(xcc)
                    # Just get a subset of GCC values to match dimensionality
                    gcc_phat = torch.cat(
                        [
                            gcc_phat[..., -n_freqs // 2:],
                            gcc_phat[..., :n_freqs // 2],
                        ],
                        dim=-1,
                    )
                    out_list.append(gcc_phat)
            gcc_phat = torch.stack(out_list, dim=0)

            # spectrogram.shape = (C=3, T, F)
            spectrogram = torch.cat([spectrogram, gcc_phat], dim=0)

        # Reshape to how soundspaces expects
        # spectrogram.shape = (F, T, C)
        # spectrogram = spectrogram.permute(2, 1, 0)

        if not time_first:
            # output input in shape (C, F, T) for both CNN and ResNet
            spectrogram = spectrogram.permute(0, 2, 1)
        else:
            # For preprocessing, makes more sense to store time-first (T, F, C)
            spectrogram = spectrogram.permute(1, 2, 0)

        return spectrogram


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


def get_silence_ratio_spectrogram(spec, db_range=80.0):
    # Get overall silence ratio (this is faster, but not exactly what we want)
    # return (spec == 0).to(dtype=torch.float32).mean()
    # Get the ratio of longest contiguous silence to the whole spectrogram
    # For simplicity, we're only considering digital silence
    # https://stackoverflow.com/a/58920786

    # audio.shape (F, Ta)
    nfreq, ntime = spec.shape
    spec = spec.permute(1, 0)
    return max(
        (
            len([i for i, _ in group])
            for is_zero, group in groupby(
                enumerate(
                    # Since we're using spectrogram, check that the decibels
                    # are at the low end of the dynamic range
                    ((spec <= -db_range).sum(axis=-1) == nfreq).tolist()
                ),
                key=itemgetter(1),
            )
            if not is_zero
        ),
        default=0,
    ) / float(ntime)


def create_spectrogram_coroutine(
    spec_tf: SpectrogramGcc,
):
    """Coroutine for computing a valid spectrogram in chunks"""
    start = True
    # Double check this, but it works for the 50% hop case
    num_invalid_pad = math.ceil(spec_tf._win_size_ms / spec_tf._hop_size_ms) - 1
    while True:
        audio, end = (yield)
        # Get spectrogram, using centering if we're at a boundary
        spec = spec_tf.forward(audio, center=(start or end), time_first=True)
        audio = None # help out gc
        # Ignore boundary frames
        if start and not end:
            spec = spec[:-num_invalid_pad]
            start = False
        elif end and not start:
            spec = spec[num_invalid_pad:]
        yield spec
        spec = None # help out gc

        # Prevent infinite loops
        if end:
            break


########################
#     FFMPEG stuff     #
########################


def get_stream(probe, codec_type):
    return next(
        (
            stream for stream in probe['streams']
            if stream['codec_type'] == codec_type
        ),
        None,
    )


def read_ffmpeg_raw(filepath, dtype, fps=None, **output_kwargs):
    try:
        tf = ffmpeg.input(filepath)
        if fps:
            tf = tf.filter("fps", fps=fps, round="up")
        tf = tf.output("pipe:", **output_kwargs, **{
            "threads": 1
        })
        out, err = (
            ffmpeg
            .input(filepath)
            .output("pipe:", **output_kwargs)
            .run(capture_stdout=True, capture_stderr=True)
        )
    except FFmpegError as e:
        print(err.stderr)
        raise
    res = torch.frombuffer(out, dtype=dtype).clone() # clone to load into memory
    return res


def read_mp4_video_ffmpeg(filepath, probe, frame_rate):
    video_stream = get_stream(probe, "video")
    width = int(video_stream["width"])
    height = int(video_stream["height"])
    fps = video_stream["r_frame_rate"]
    if isinstance(fps, str):
        assert fps.count("/") == 1
        num, den = fps.split("/")
        fps = int(num) // int(den)
    # https://github.com/kkroening/ffmpeg-python/blob/master/examples/README.md#convert-video-to-numpy-array
    return read_ffmpeg_raw(
        filepath,
        dtype=torch.uint8,
        fps=(frame_rate if frame_rate != fps else None),
        format="rawvideo",
        pix_fmt="rgb24",
    ).reshape(-1, height, width, 3)


def read_mp4_audio_ffmpeg(filepath, probe, sample_rate):
    audio_stream = get_stream(probe, "audio")
    num_channels = int(audio_stream["channels"])
    assert num_channels == 2, f"expected stereo audio for {filepath}"
    # https://www.kaggle.com/code/josecarmona/ffmpeg-python-example-to-extract-audio-from-mp4
    return read_ffmpeg_raw(
        filepath,
        dtype=torch.float32,
        format='f32le',
        acodec='pcm_f32le',
        ac=2,
        ar=int(sample_rate),
    ).reshape(-1, num_channels)

