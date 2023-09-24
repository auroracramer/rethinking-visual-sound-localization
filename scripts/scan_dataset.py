import os
import json
from pathlib import Path

import ffmpeg
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader


from rethinking_visual_sound_localization.training.data import AudioVisualDataset, Ego4DDataset, Normalize
from rethinking_visual_sound_localization.training.data import worker_init_fn
from rethinking_visual_sound_localization.audio_utils import SpectrogramGcc


def get_video_files(data_root: Path, expected_num_channels: int = 2):
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
        if num_channels == expected_num_channels:
            file_list.append(fpath.stem)
    # Return in sorted order for reproducibility
    return sorted(file_list)


if __name__ == "__main__":
    # data source location
    ego = "/home/acramer/datasets/ego4d/v2_50gb/video_540ss"

    args = {
        "batch_size": 1,  # original 256
        "num_workers": 0,  # original 8
        "random_state": 2021,
        "path_to_project_root": "/home/acramer/outputs/rethink_ego",
        "path_to_data_root": ego,
        "spec_config": {
            "STEREO": True,
            "SAMPLE_RATE": 16000,
            "WIN_SIZE_MS": 40,
            "NUM_MELS": 64,
            "HOP_SIZE_MS": 20,
            "DOWNSAMPLE": 0,
            "GCC_PHAT": True,
        }
    }
    seed_everything(args["random_state"])

    sr = int(args["spec_config"]["SAMPLE_RATE"])
    dataset = args["path_to_data_root"]
    project_root = args["path_to_project_root"]
    os.makedirs(project_root, exist_ok=True)

    fps = 30
    image_dim = 128

    duration: int = 5
    sample_rate: int = 16000
    chunk_duration: int = 10
    num_channels: int = 2
    random_seed: int = 1337
    valid_ratio: float = 0.1
    silence_threshold: float = 0.1

    video_transform = Normalize(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711),
    )
    image_feature_shape = (3, image_dim, image_dim)
    spec_tf = SpectrogramGcc(sample_rate, duration)
    audio_transform = partial(spec_tf.forward, center=True, time_first=False)
    num_channels = num_channels
    data_root = Path(data_root)
    silence_threshold = silence_threshold
    device = 'cuda'
    num_retry_silence = 3
    assert chunk_duration >= duration


    def get_video_filepath(video_name):
        return os.path.join(data_root, f"{video_name}.mp4")


    def create_stream_reader(
        vfile, num_chunk_audio_samples,
        num_chunk_video_frames, video_width, video_height, video_codec,
    ):
        streamer = StreamReader(vfile)
        num_channels = streamer.get_src_stream_info(
            streamer.default_audio_stream
        ).num_channels
        assert num_channels == 2, (
            f"{vfile.name} must have two audio channels"
        )

        # add output streams
        audio_filter_desc = ",".join(
            [
                f"aresample={sample_rate}",
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
        
        video_decoder = video_codec
        video_hw_accel = None

        if video_width > video_height:
            new_video_width = -1
            new_video_height = int(image_dim)
        else:
            new_video_width = int(image_dim)
            new_video_height = -1

        video_filter_desc = ",".join(
            [
                f"scale=width={new_video_width}:height={new_video_height}",
                f"crop={image_dim}:{image_dim}:exact=1",
                f"fps={fps}",
                f"format=pix_fmts=rgb24",
            ]
        )
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

    def sample_offset_ts(start_ts, end_ts, duration):
        valid_duration = (end_ts - start_ts) - duration
        return rng.uniform(0.0, valid_duration)

    files = get_video_files(data_root)

    for f in files:
        fpath = get_video_filepath(f)
        try:
            probe = ffmpeg.probe(fpath)
        except ffmpeg.Error as e:
            print(e.stderr)
            raise e
        audio_probe = get_stream(probe, "audio")
        video_probe = get_stream(probe, "video")
        audio_duration = float(audio_probe["duration"])
        video_duration = float(video_probe["duration"])
        video_width = int(video_probe["width"])
        video_height = int(video_probe["height"])
        video_codec = video_probe["codec_name"]
        full_duration = max(audio_duration, video_duration)
        if duration > full_duration:
            # Don't need to add if we're scanning ahead of time
            print(
                f"WARNING: "
                f"video '{Path(fpath).name}' (duration: {full_duration} "
                f"seconds) is too short (less than {self.duration} seconds). "
                f"Video will be skipped for sampling."
            )
            continue

        num_audio_samples = duration * sample_rate
        num_video_samples = duration * fps
        num_chunk_audio_samples = chunk_duration * sample_rate
        num_chunk_video_frames = chunk_duration * fps
        num_expected_chunks = int(full_duration / chunk_duration)

        num_chunks = 0
        num_missing_chunks = 0
        num_silent_chunks = 0
        num_dupe_chunks = 0
        num_ignore_chunks = 0
        num_short_chunks = 0
        num_failed_chunks = 0
        num_valid_chunks = 0

        with open(fpath, 'rb') as vfile:
            streamer = create_stream_reader(
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
            for chunk_idx in range(num_expected_chunks):
                # Get chunk boundaries
                start_ts = float(chunk_duration * chunk_idx)
                end_ts = float(min(start_ts + chunk_duration, full_duration))

                num_chunks += 1

                # Seek to the start of each chunk and get audio/video.
                # This is done instead of streamer.stream() to avoid
                # decoding chunks we'll end up skipping
                streamer.seek(start_ts)
                streamer.fill_buffer(timeout=None, backoff=10.0)

                # Skip chunk if no chunk is available
                if not streamer.is_buffer_ready():
                    num_missing_chunks += 1
                    continue

                audio, video = streamer.pop_chunks()

                # Skip chunk if missing audio or video
                if audio is None or video is None:
                    num_missing_chunks += 1
                    continue

                # audio.shape = (C, T)
                audio = audio.to(device=device).transpose(0, 1) # put channels first
                end_ts_audio = start_ts + audio.shape[-1] / sample_rate

                # Shape sanity checks
                assert audio.shape[0] == num_channels

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
                if chunk_silence_ratio >= silence_threshold:
                    print(
                        f"WARNING: "
                        f"video '{Path(fpath).name}': "
                        f"[{float(start_ts)} - {end_ts}] contains more than "
                        f"{int(silence_threshold * 100)}% digital silence"
                    )

                if duration > (min(end_ts, end_ts_audio) - start_ts):
                    too_short_chunks.add(chunk_idx)
                    # We don't have enough for a full window, so skip
                    num_short_chunks += 1
                    continue

                # sample a random window within the chunk
                for _ in range(num_retry_silence):
                    # Sample a start time relative to start of the chunk (w.r.t audio)
                    offset_ts = sample_offset_ts(start_ts, end_ts_audio)

                    # Get corresponding indices for audio and video data
                    audio_index = int(offset_ts * sample_rate)
                    video_index = int(offset_ts * fps) + (num_video_samples // 2)

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
                        f"in {num_retry_silence} attempts. Skipping chunk..."
                    )
                    continue

                # audio.shape (C, F, Ta)
                audio = audio_transform(audio)
                # video.shape (C, D, D)
                video = video_transform(
                    _video_to_float_tensor(
                        video[video_index]
                    )
                )#.permute(1, 2, 0)
                num_valid_chunks += 1
                yield audio, video

            # Clean up streamer
            streamer.remove_stream(1)
            streamer.remove_stream(0)
            del streamer

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
