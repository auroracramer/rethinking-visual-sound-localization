import os
import json

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
        "batch_size": 1,  # original 256
        "num_workers": 1,  # original 8
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
    num_jobs = os.environ.get("SLURM_ARRAY_TASK_COUNT")
    job_idx = os.environ.get("SLURM_ARRAY_TASK_ID")
    num_jobs = int(num_jobs) if num_jobs else None
    job_idx = int(job_idx) if job_idx else None

    # assign datasets
    if dataset == ego:
        # train_dataset =
        dataset = Ego4DDataset(
            data_root=args['path_to_data_root'],
            split="full",
            duration=5,
            sample_rate=sr,
            num_jobs=num_jobs,
            job_idx=job_idx,
            project_root=project_root,
        )
    elif dataset == vgg:
        dataset = AudioVisualDataset(
            data_root=args['path_to_data_root'],
            split="full",
            duration=5,
            sample_rate=sr,
        )
    else:
        raise Exception("Not Implemented")

    dataloader = DataLoader(
        dataset,
        num_workers=args["num_workers"],
        batch_size=args["batch_size"],
        pin_memory=False,
        drop_last=False,
        worker_init_fn=worker_init_fn,
    )
    print("- scanning dataset...")
    batch_idx = -1
    for batch_idx, batch in enumerate(dataloader):
        pass
    print(f"   * found {batch_idx + 1} batches")
