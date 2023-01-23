import argparse
import sys
import logging
logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
)
import torch
from rethinking_visual_sound_localization.training.preprocess import (
    is_stereo, preprocess_video
)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=str)
    parser.add_argument("output_dir", type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    sample_rate: int = 16000
    num_channels = 2
    fps: int = 30
    silence_threshold: float = 0.1
    buffer_duration: int = 10
    chunk_duration: int = 5
    num_threads: int = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Running ffmpeg with {num_threads} threads")
    logging.info(f"Running transforms with device '{device}'")

    logging.info(f"Checking if '{args.video_path}' is stereo")
    if is_stereo(args.video_path):
        logging.info(f"Processing '{args.video_path}'")
        preprocess_video(
            args.video_path,
            args.output_dir,
            buffer_duration,
            chunk_duration,
            sample_rate,
            fps,
            silence_threshold,
            num_threads=num_threads,
            device=device,
        )
    else:
        logging.info(f"'{args.video_path}' is not stereo. Skipping")

    logging.info("Done! Yay!!!!")