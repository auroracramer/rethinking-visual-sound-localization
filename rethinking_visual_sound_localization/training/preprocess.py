import logging
import math
import h5py
import torch
import ffmpeg
from PIL import Image
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from typing import Optional
from warnings import warn
from torchaudio.io import StreamReader
from ..audio_utils import (
    create_spectrogram_coroutine,
    get_silence_ratio,
    get_stream,
    SpectrogramGcc,
)
from ..training.data import _transform


def preprocess_video(
    video_path: str,
    hdf5_dir: str,
    buffer_duration: int,
    chunk_duration: int,
    sample_rate: int,
    fps: int,
    silence_threshold: float = 0.1,
    num_threads: int = 1,
    device: Optional[str] = None,
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
    audio_duration = float(get_stream(probe, "audio")["duration"])
    video_duration = float(get_stream(probe, "video")["duration"])
    full_duration = max(audio_duration, video_duration)
    # Set up transforms
    logging.info("    - setting up audio transform")
    spec_tf = SpectrogramGcc(sample_rate, buffer_duration, device=device)
    spec_coro = create_spectrogram_coroutine(spec_tf)
    def audio_transform(audio, last):
        next(spec_coro) # advance coroutine to first yield
        return spec_coro.send((audio, last))
    logging.info("    - setting up image transform")
    video_transform = _transform(video_dim)

    audio_hop_size_s = spec_tf._hop_size_ms / 1000
    total_audio_frames = int(audio_duration / audio_hop_size_s) + 1
    total_video_frames = int(video_duration * fps)
    num_chunk_audio_frames = int(chunk_duration / audio_hop_size_s) + 1
    num_chunk_video_frames = int(chunk_duration * fps)
    audio_nfreq = spec_tf._n_mels
    audio_nchan = (3 if spec_tf._include_gcc_phat else 2)

    # Set up HDF5 file
    logging.info(f"    - opening hdf5 file: {str(hdf5_path)}")
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

        # Open video for streaming via ffmpeg
        logging.info(f"    - opening video for streaming")
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

            # add output streams
            logging.info(f"    - adding output audio stream")
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
            # Seek to start to avoid ffmpeg decoding in the background
            # https://github.com/dmlc/decord/issues/208#issuecomment-1157632702
            logging.info(f"    - seeking to start of file")
            streamer.seek(0)

            # Note: because we're using the min of the stream durations, we might
            #       have an additional buffer, but this is a good enough
            #       estimate for now
            num_chunks = math.ceil(full_duration / buffer_duration)

            num_valid_chunks = 0
            silent = True
            identical_audio_channels = True
            silent_chunks = set()
            identical_audio_channels_chunks = set()
            too_short_chunks = set()
            missing_chunks = set()

            audio_frame_idx = 0
            video_frame_idx = 0
            end = False
            # process video in chunks
            log_interval = 60 // buffer_duration
            logging.info(f"    - initializing stream")
            for chunk_idx, (audio, video) in enumerate(streamer.stream()):
                if chunk_idx % log_interval == 0:
                    logging.info(f"    - processing output for chunk {chunk_idx+1}/{num_chunks}")
                
                start_ts = buffer_duration * chunk_idx
                end_ts = min(start_ts + buffer_duration, full_duration)

                # Process audio
                if audio is not None: # account for if there's video and no audio
                    # This check is most important for audio, since we need
                    # the end flag to handle the last spectrogram frame
                    if not (chunk_idx < (num_chunks - 1)):
                        assert not end, f"Expected {num_chunks}, but found at least {chunk_idx+1}"
                        end = True

                    audio = audio.to(device=device).transpose(0, 1) # put channels first
                    # Shape sanity checks
                    if chunk_idx % log_interval == 0:
                        logging.info(f"        * audio buffer shape: {audio.shape}")
                    assert audio.shape[0] == num_channels
                    # Due to decoding quirks, it seems like we don't necessarily
                    # get a predictable buffer size, so just warn the user if
                    # the buffer is bigger than expected
                    if not (audio.shape[1] <= num_buffer_audio_samples):
                        too_short_chunks.add(chunk_idx)
                        logging.info(
                            f"expected audio shape "
                            f"({num_channels}, x <= {num_buffer_audio_samples}), "
                            f"but got {audio.shape}"
                        )

                    # If both channels are the same, warn user
                    if chunk_idx % log_interval == 0:
                        logging.info(f"        * checking for duplicate audio channels")
                    if torch.allclose(audio[0], audio[1]):
                        identical_audio_channels_chunks.add(chunk_idx)
                        identical_audio_channels = identical_audio_channels and True
                        warn(
                            f"video '{Path(video_path).name}': "
                            f"[{float(start_ts)} - {end_ts}] has duplicate audio channels"
                        )
                    else:
                        identical_audio_channels = False

                    # If either channel has too much digital silence, warn user
                    if chunk_idx % log_interval == 0:
                        logging.info(f"        * checking for silence")
                    silence_ratio_0 = get_silence_ratio(audio[0])
                    silence_ratio_1 = get_silence_ratio(audio[1])
                    if max(silence_ratio_0, silence_ratio_1) >= silence_threshold:
                        silent_chunks.add(chunk_idx)
                        silent = silent and True
                        warn(
                            f"video '{Path(video_path).name}': "
                            f"[{float(start_ts)} - {end_ts}] contains more than "
                            f"{int(silence_threshold * 100)}% digital silence"
                        )
                    else:
                        silent = False

                    # Process audio
                    if chunk_idx % log_interval == 0:
                        logging.info(f"        * transforming audio {str(tuple(audio.shape))}")
                    audio = audio_transform(audio, end).detach().cpu().numpy()

                    # Update HDF5
                    # audio.shape (Ta, F, C)
                    assert audio.shape[1:] == (audio_nfreq, audio_nchan)
                    if chunk_idx % log_interval == 0:
                        logging.info(f"        * storing audio in hdf5 {str(tuple(audio.shape))}")
                    audio_dataset[audio_frame_idx:audio_frame_idx + audio.shape[0]] = audio
                    audio_frame_idx += audio.shape[0]

                # Process video
                if video is not None:  # account for if there's audio and no video
                    # Shape sanity checks
                    if chunk_idx % log_interval == 0:
                        logging.info(f"        * video buffer shape: {video.shape}")
                    assert video.shape[1] == video_nchan
                    if not (video.shape[0] <= num_buffer_video_frames):
                        too_short_chunks.add(chunk_idx)
                        logging.info(
                            f"expected video shape "
                            f"(x <= {num_buffer_video_frames}, {video_nchan}, :, :), "
                            f"but got {video.shape}"
                        )

                    if chunk_idx % log_interval == 0:
                        logging.info(f"        * transforming video {str(tuple(video.shape))}")
                    video = torch.stack(
                        [
                            video_transform(
                                Image.fromarray(
                                    frame.permute(1, 2, 0).detach().cpu().numpy()
                                )
                            )
                            for frame in video
                        ],
                        dim=0,
                    ).permute(0, 2, 3, 1).detach().cpu().numpy()

                    # Not completely necessary but it feels weird to have this saved
                    # so the channels are in different places, so for video
                    # we'll move channel dim before image dims 
                    if chunk_idx % log_interval == 0:
                        logging.info(f"        * restructuring video")

                    # Update HDF5
                    # video.shape (Tv, D, D, C)
                    assert video.shape[1:] == (video_dim, video_dim, video_nchan)
                    if chunk_idx % log_interval == 0:
                        logging.info(f"        * storing video in hdf5 {str(tuple(video.shape))}")
                    video_dataset[video_frame_idx:video_frame_idx + video.shape[0]] = video
                    video_frame_idx += video.shape[0]

                if (not audio) or (not video):
                    missing_chunks.add(chunk_idx)
                else:
                    num_valid_chunks += 1
                

        logging.info(f"    - updating metadata")
        metadata = dict(
            path=str(video_path),
            buffer_duration=int(buffer_duration),
            duration=float(max(audio_duration, video_duration)),
            num_audio_frames=audio_frame_idx, # since we're preallocating
            num_video_frames=video_frame_idx, # we need to keep track of this
            sample_rate=int(sample_rate),
            fps=int(fps),
            silence_threshold=float(silence_threshold),
            audio_frame_hop=float(audio_hop_size_s),
            audio_nfreq=int(audio_nfreq),
            audio_nchan=int(audio_nchan), # stereo + GCC
            video_frame_hop=1.0/fps,
            video_dim=int(video_dim),
            video_nchan=int(video_nchan), # RGB
            num_chunks=num_chunks,
            num_valid_chunks=int(num_valid_chunks),
            silent=bool(silent),
            identical_audio_channels=bool(identical_audio_channels),
            silent_chunks=list(silent_chunks),
            identical_audio_channels_chunks=list(identical_audio_channels_chunks),
            too_short_chunks=list(too_short_chunks),
            missing_chunks=list(missing_chunks),
            num_silent_chunks=len(silent_chunks),
            num_identical_audio_channels_chunks=len(identical_audio_channels_chunks),
            num_too_short_chunks=len(too_short_chunks),
            num_missing_chunks=len(missing_chunks),
        )
        for k, v in metadata.items():
            h5.attrs[k] = v

def is_stereo(fpath):
    metadata = ffmpeg.probe(str(fpath))
    num_channels = None
    for stream in metadata["streams"]:
        if stream["codec_type"] == "audio":
            if num_channels is not None:
                raise ValueError(
                    f"Found more than one audio stream for {str(fpath)}"
                )
            num_channels = int(stream["channels"])
    return num_channels == 2


def get_video_files(data_root: str, stereo_only: bool = False):
    data_root = Path(data_root)
    file_list = []
    for fpath in data_root.glob("*.mp4"):
        if not stereo_only or is_stereo(fpath):
            file_list.append(str(fpath))
    # Return in sorted order for consistency
    return sorted(file_list, key=lambda x: Path(x).relative_to(data_root))