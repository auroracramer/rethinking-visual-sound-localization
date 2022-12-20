"""

Reused and modified from https://github.com/auroracramer/hear-baseline
This script prepares audio features for audio encoders.

"""

from typing import Optional, Type

import numpy as np
import torch
from torch.nn.functional import avg_pool2d
from torchaudio.functional import amplitude_to_DB, melscale_fbanks


class SpectrogramGcc(torch.nn.Module):
    r"""Create a spectrogram+gcc feature from a single or a batch of multi-channel audio in shape (..., time).

    Args:
        win_length (int or None, optional): Window size. (Default: ``n_fft``)
        hop_length (int or None, optional): Length of hop between STFT windows.
        n_mels (int): Number of mel filterbanks
        n_fft (int, optional): Size of FFT, creates ``n_fft // 2 + 1`` bins.
        downsample (int): Average pool kernel size
        include_gcc_phat (bool) : Whether to concatenate gcc phat after spectrogram
        window: (Callable[..., Tensor], optional): A function to create a window tensor
            that is applied/multiplied to each frame/window. (Default: ``torch.hann_window``)
        mel_scale : Triangular filter banks (fb matrix) of size (``n_freqs``, ``n_mels``)

    """

    def __init__(self, config) -> None:
        super(SpectrogramGcc, self).__init__()
        self._sample_rate = config["SAMPLE_RATE"]
        spec_info = self.get_spectrogram_info(config, self._sample_rate)
        self._hop_length = spec_info["hop_length"]
        self._win_length = spec_info["win_length"]
        self._n_mels = spec_info["n_mels"]
        self._n_fft = spec_info["n_fft"]
        self._downsample = spec_info["downsample"]
        self._include_gcc_phat = spec_info["include_gcc_phat"]
        self._window = torch.hann_window(self._win_length, device="cpu")
        self._mel_scale = melscale_fbanks(
            n_freqs=(self._n_fft // 2) + 1,
            f_min=0,
            f_max=self._sample_rate / 2,  # nyquist
            n_mels=self._n_mels,
            sample_rate=self._sample_rate,
            mel_scale="htk",
            norm=None,
        ).to(device="cpu", dtype=torch.float32) if self._n_mels else None
        self.feature_shape = self.forward(np.ones((2, self._sample_rate))).shape


    def forward(self, waveform):
        return self.compute_spectrogram(
            waveform,
            win_length=self._win_length,
            hop_length=self._hop_length,
            n_fft=self._n_fft,
            n_mels=self._n_mels,
            window=self._window,
            mel_scale=self._mel_scale,
            downsample=self._downsample,
            include_gcc_phat=self._include_gcc_phat,
            backend="numpy",
        )

    def next_greater_power_of_2(self, x):
        return 2 ** (x - 1).bit_length()

    def get_spectrogram_info(self, spec_config, sampling_rate):
        win_length = int(sampling_rate * (spec_config["WIN_SIZE_MS"] / 1000.0))
        n_mels = int(spec_config["NUM_MELS"])
        n_fft = int(self.next_greater_power_of_2(win_length))
        return dict(
            sampling_rate=sampling_rate,
            win_length=win_length,
            hop_length=int(sampling_rate * (spec_config["HOP_SIZE_MS"] / 1000.0)),
            n_mels=n_mels,
            n_fft=n_fft,
            n_freqs=(n_mels if n_mels else ((n_fft // 2) + 1)),
            downsample=spec_config["DOWNSAMPLE"],
            include_gcc_phat=bool(spec_config["GCC_PHAT"]),
            n_channels=(2 + (1 if spec_config["GCC_PHAT"] else 0)),
        )

    def compute_spectrogram(
            self, audio_data, win_length: int, hop_length: int, n_fft: int, n_mels: int,
            window: Optional[torch.Tensor], mel_scale: Optional[torch.Tensor],
            downsample: Optional[int], include_gcc_phat: bool, backend: str = "torch",
    ):
        assert backend in ("torch", "numpy")
        # multichannel stft returns (..., F, T)
        # TODO: modify this according to torchaudio Spectrogram
        stft = torch.stft(
                    input=torch.tensor(audio_data, device='cpu', dtype=torch.float32),
                    win_length=win_length,
                    hop_length=hop_length,
                    n_fft=n_fft,
                    center=True,
                    window=(
                        window if window is not None
                        else torch.hann_window(win_length, device="cpu")
                    ),
                    pad_mode="constant",  # constant for zero padding
                    return_complex=True,
                )
        # Compute power spectrogram
        spectrogram = (torch.abs(stft) ** 2.0).to(dtype=torch.float32)
        # Apply the mel-scale filter to the power spectrogram
        if mel_scale is not None:
            spectrogram = torch.matmul(spectrogram, mel_scale)
        # Optionally downsample
        if downsample:
            spectrogram = avg_pool2d(
                spectrogram.unsqueeze(dim=0),
                kernel_size=(downsample, downsample),
            ).squeeze(dim=0)
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

            # Downsample
            if downsample:
                gcc_phat = torch.nn.functional.avg_pool2d(
                    gcc_phat,
                    kernel_size=(downsample, downsample),
                )

            # spectrogram.shape = (C=3, T, F)
            spectrogram = torch.cat([spectrogram, gcc_phat], dim=0)

        # Reshape to how soundspaces expects
        # spectrogram.shape = (F, T, C)
        # spectrogram = spectrogram.permute(2, 1, 0)

        # output input in shape (C, F, T) for both CNN and ResNet
        spectrogram = spectrogram.permute(0, 2, 1)
        if backend == "torch":
            return spectrogram
        elif backend == "numpy":
            return spectrogram.numpy().astype(np.float32)