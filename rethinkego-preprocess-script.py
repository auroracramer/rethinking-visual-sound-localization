import sys
import logging
logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
)
from rethinking_visual_sound_localization.training.preprocess import (
    get_video_files, preprocess_video
)

if __name__ == "__main__":
    # data source location
    data_root: str = "/vast/work/public/ml-datasets/ego4d/v1/full_scale"
    output_dir: str = "/scratch/jtc440/ego4d-preprocessed-data"
    sample_rate: int = 16000
    num_channels = 2
    fps: int = 30
    silence_threshold: float = 0.1
    buffer_duration: int = 10
    chunk_duration: int = 5
    num_threads: int = 8

    logging.info(f"Looking for videos in '{data_root}' ...")
    video_paths = get_video_files(data_root, stereo_only=True)
    logging.info("Processing videos ...")
    for video_path in video_paths:
        logging.info(f" * {video_path}")
        preprocess_video(
            video_path,
            output_dir,
            buffer_duration,
            chunk_duration,
            sample_rate,
            fps,
            silence_threshold,
            num_threads=num_threads,
        )
    logging.info("Done! Yay!!!!")
