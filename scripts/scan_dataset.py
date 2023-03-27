import os
import pickle as pk

from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader

from rethinking_visual_sound_localization.training.data import AudioVisualDataset, Ego4DDataset
from rethinking_visual_sound_localization.training.data import worker_init_fn

if __name__ == "__main__":
    # data source location
    vgg = "/vast/sd5397/data/vggsound/data"
    #ego = "/vast/work/public/ml-datasets/ego4d/v1"
    ego = "/vast/work/public/ml-datasets/ego4d/v1/full_scale"

    args = {
        "batch_size": 256,  # original 256
        "num_workers": 20,  # original 8
        "random_state": 2021,
        "path_to_project_root": "/scratch/jtc440/rethink_ego",
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
    train_files_path = os.path.join(project_root, "train_files.pkl")
    train_ignore_files_path = os.path.join(project_root, "train_ignore_files.pkl")
    train_ignore_segments_path = os.path.join(project_root, "train_ignore_segments.pkl")
    valid_files_path = os.path.join(project_root, "valid_files.pkl")
    valid_ignore_files_path = os.path.join(project_root, "valid_ignore_files.pkl")
    valid_ignore_segments_path = os.path.join(project_root, "valid_ignore_segments.pkl")

    # assign datasets
    if dataset == ego:
        # train_dataset =
        train_dataset = Ego4DDataset(
            data_root=args['path_to_data_root'],
            split="train",
            duration=5,
            sample_rate=sr,
            num_jobs=os.environ.get("SLURM_ARRAY_TASK_COUNT"),
            job_idx=os.environ.get("SLURM_ARRAY_TASK_ID"),
        )
        val_dataset = Ego4DDataset(
            data_root=args['path_to_data_root'],
            split="valid",
            duration=5,
            sample_rate=sr,
            num_jobs=os.environ.get("SLURM_ARRAY_TASK_COUNT"),
            job_idx=os.environ.get("SLURM_ARRAY_TASK_ID"),
        )
    elif dataset == vgg:
        train_dataset = AudioVisualDataset(
            data_root=args['path_to_data_root'],
            split="train",
            duration=5,
            sample_rate=sr,
        )
        val_dataset = AudioVisualDataset(
            data_root=args['path_to_data_root'],
            split="valid",
            duration=5,
            sample_rate=sr,
        )
    else:
        raise Exception("Not Implemented")

    train_loader = DataLoader(
        train_dataset,
        num_workers=args["num_workers"],
        batch_size=args["batch_size"],
        pin_memory=True,
        drop_last=False,
        worker_init_fn=worker_init_fn,
    )
    print("- scanning train dataset...")
    for batch_idx, batch in enumerate(train_loader):
        pass
    print(f"   * found {batch_idx + 1} batches")
    with open(train_files_path, "wb") as f:
        pk.dump(train_dataset.files, f)
    print(f"   * saved {train_files_path}")
    with open(train_ignore_files_path, "wb") as f:
        pk.dump(train_dataset.ignore_files, f)
    print(f"   * saved {train_ignore_files_path}")
    with open(train_ignore_segments_path, "wb") as f:
        pk.dump(train_dataset.ignore_segments, f)
    print(f"   * saved {train_ignore_segments_path}")

    valid_loader = DataLoader(
        val_dataset,
        num_workers=args["num_workers"],
        batch_size=args["batch_size"],
        pin_memory=True,
        drop_last=False,
        worker_init_fn=worker_init_fn,
    )
    print("- scanning valid dataset...")
    for batch_idx, batch in enumerate(valid_loader):
        pass
    print(f"   * found {batch_idx + 1} batches")
    with open(valid_files_path, "wb") as f:
        pk.dump(val_dataset.files, f)
    print(f"   * saved {valid_files_path}")
    with open(valid_ignore_files_path, "wb") as f:
        pk.dump(val_dataset.ignore_files, f)
    print(f"   * saved {valid_ignore_files_path}")
    with open(valid_ignore_segments_path, "wb") as f:
        pk.dump(val_dataset.ignore_segments, f)
    print(f"   * saved {valid_ignore_segments_path}")
