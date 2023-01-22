import h5py
import math
import torch
import ffmpeg
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from warnings import warn
from torchaudio.io import StreamReader
from ..audio_utils import SpectrogramGcc
from ..training.data import _transform

### DEBUG
import logging


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


def get_spectrogram(
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


def preprocess_video(
    video_path: str,
    hdf5_dir: str,
    buffer_duration: int,
    chunk_duration: int,
    sample_rate: int,
    fps: int,
    silence_threshold: float = 0.1,
    num_threads: int = 1,
):

    video_path = Path(video_path)
    hdf5_path = Path(hdf5_dir).joinpath(f"{video_path.stem}.h5")
    hdf5_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info("    - running ffprobe to get metadata")
    probe = ffmpeg.probe(video_path)
    num_buffer_audio_samples = buffer_duration * sample_rate
    num_buffer_video_frames = buffer_duration * fps
    video_dim = 224
    video_nchan = 3
    # Take minimum duration of streams
    full_duration = min(float(stream['duration']) for stream in probe['streams'])
    logging.info("    - setting up audio transform")
    # Set up transforms
    spec_tf = SpectrogramGcc(sample_rate, buffer_duration)
    spec_coro = get_spectrogram(spec_tf)
    next(spec_coro) # initialize coroutine
    audio_transform = lambda audio, last: spec_coro.send((audio, last))
    logging.info("    - setting up image transform")
    video_transform = _transform(video_dim)

    audio_hop_size_s = spec_tf._hop_size_ms / 1000
    total_audio_frames = int(full_duration / audio_hop_size_s) + 1
    total_video_frames = int(full_duration * fps)
    num_chunk_audio_frames = int(chunk_duration / audio_hop_size_s) + 1
    num_chunk_video_frames = int(chunk_duration * fps)
    audio_nfreq = spec_tf._n_mels
    audio_nchan = (3 if spec_tf._include_gcc_phat else 2)

    metadata = dict(
        path=str(video_path),
        buffer_duration=int(buffer_duration),
        sample_rate=int(sample_rate),
        fps=int(fps),
        silence_threshold=float(silence_threshold),
        audio_frame_hop=float(audio_hop_size_s),
        audio_nfreq=int(audio_nfreq),
        audio_nchan=int(audio_nchan), # stereo + GCC
        video_frame_hop=1.0/fps,
        video_dim=int(video_dim),
        video_nchan=int(video_nchan), # RGB
    )

    logging.info(f"    - opening h5 file: {str(hdf5_path)}")
    # Set up HDF5 file
    with h5py.File(str(hdf5_path), 'w') as h5:
        logging.info(f"    - creating audio dataset")
        audio_dataset = h5.create_dataset(
            "audio",
            shape=(total_audio_frames, audio_nfreq, audio_nchan),
            chunks=(num_chunk_audio_frames, audio_nfreq, audio_nchan),
            dtype='f8',
            compression='lzf',
        )
        logging.info(f"    - creating video dataset")
        video_dataset = h5.create_dataset(
            "video",
            shape=(total_video_frames, video_dim, video_dim, video_nchan),
            chunks=(num_chunk_video_frames, video_dim, video_dim, video_nchan),
            dtype='f8',
            compression='lzf',
        )
        logging.info(f"    - updating metadata")
        for k, v in metadata.items():
            h5.attrs[k] = v

        logging.info(f"    - opening video for streaming")
        # Open video for streaming via ffmpeg
        with open(video_path, 'rb') as vfile:
            logging.info(f"    - creating stream reader")
            streamer = StreamReader(vfile)
            logging.info(f"    - checking stream info")
            num_channels = streamer.get_src_stream_info(
                streamer.default_audio_stream
            ).num_channels
            assert num_channels == 2, (
                f"{video_path} must have two audio channels"
            )

            logging.info(f"    - adding output audio stream")
            # add output streams
            streamer.add_basic_audio_stream(
                frames_per_chunk=num_buffer_audio_samples,
                sample_rate=sample_rate,
                format='fltp',
                decoder_option={
                    "threads": str(num_threads),
                }
            )
            logging.info(f"    - adding output video stream")
            streamer.add_basic_video_stream(
                frames_per_chunk=num_buffer_video_frames,
                frame_rate=fps,
                format="rgb24",
                decoder_option={
                    "threads": str(num_threads),
                }
            )
            logging.info(f"    - seeking to start of file")
            # https://github.com/dmlc/decord/issues/208#issuecomment-1157632702
            streamer.seek(0)

            num_chunks = math.ceil(full_duration / buffer_duration)

            audio_frame_idx = 0
            video_frame_idx = 0
            end = False
            # process video in chunks
            log_interval = 60 // buffer_duration
            logging.info(f"    - initializing stream")
            for chunk_idx, (audio, video) in enumerate(streamer.stream()):
                if chunk_idx % log_interval == 0:
                    logging.info(f"    - processing output for chunk {chunk_idx+1}/{num_chunks}")

                # Stream file to only load the relevant chunk at at time
                start_ts = buffer_duration * chunk_idx
                end_ts = (
                    start_ts + buffer_duration
                    if (chunk_idx == (num_chunks - 1))
                    else full_duration
                )

                # audio.shape = (frames, channels)
                # video.shape = (frames, channels, height, width)
                assert audio.shape == (num_buffer_audio_samples, 2)
                assert video.shape[:2] == (num_buffer_video_frames, 3)

                if chunk_idx % log_interval == 0:
                    logging.info(f"        * checking for duplicate channels")
                # If both channels are the same, warn user
                if not torch.allclose(audio[0], audio[1]):
                    warn(
                        f"video '{Path(video_path).name}': "
                        f"[{float(start_ts)} - {end_ts}] has duplicate channels"
                    )

                if chunk_idx % log_interval == 0:
                    logging.info(f"        * checking for silence")
                # If either channel has too much digital silence, warn user
                silence_ratio_0 = get_silence_ratio(audio[0])
                silence_ratio_1 = get_silence_ratio(audio[1])
                if max(silence_ratio_0, silence_ratio_1) >= silence_threshold:
                    warn(
                        f"video '{Path(path).name}': "
                        f"[{float(start_ts)} - {end_ts}] contains more than "
                        f"{int(silence_threshold * 100)}% digital silence"
                    )

                # Process audio and video
                if chunk_idx % log_interval == 0:
                    logging.info(f"        * transforming audio")
                audio = audio_transform(audio, end).detach().cpu().numpy()
                if chunk_idx % log_interval == 0:
                    logging.info(f"        * transforming video")
                video = video_transform(video).detach().cpu().numpy()

                if chunk_idx % log_interval == 0:
                    logging.info(f"        * restructuring video")
                # Not completely necessary but it feels weird to have this saved
                # so the channels are in different places, so for video
                # we'll move channel dim before image dims
                video = video.permute(0, 3, 1, 2)

                # Update HDF5
                # audio.shape (Ta, F, C)
                # video.shape (Tv, D, D, C)
                if chunk_idx % log_interval == 0:
                    logging.info(f"        * storing audio in hdf5")
                audio_dataset[audio_frame_idx:audio_frame_idx + audio.shape[0]] = audio
                if chunk_idx % log_interval == 0:
                    logging.info(f"        * storing video in hdf5")
                video_dataset[video_frame_idx:video_frame_idx + video.shape[0]] = video

                audio_frame_idx += audio.shape[0]
                video_frame_idx += video.shape[0]


def get_video_files(data_root: str, stereo_only: bool = False):
    data_root = Path(data_root)
    file_list = []
    for fpath in data_root.glob("*.mp4"):
        metadata = ffmpeg.probe(str(fpath))
        num_channels = None
        for stream in metadata["streams"]:
            if stream["codec_type"] == "audio":
                if num_channels is not None:
                    raise ValueError(
                        f"Found more than one audio stream for {str(fpath)}"
        )
                num_channels = int(stream["channels"])
        if not stereo_only or (num_channels == 2):
            file_list.append(str(fpath))
    # Return in sorted order for consistency
    return sorted(file_list, key=lambda x: Path(x).relative_to(data_root))