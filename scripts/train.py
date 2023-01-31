import os

import torch
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchsummary import summary

from rethinking_visual_sound_localization.training.data import AudioVisualDataset, Ego4DDataset
from rethinking_visual_sound_localization.training.data import worker_init_fn
from rethinking_visual_sound_localization.training.model import RCGrad, RCGradSavi

if __name__ == "__main__":
    # data source location
    vgg = "/vast/sd5397/data/vggsound/data"
    #ego = "/vast/work/public/ml-datasets/ego4d/v1"
    ego = "/vast/work/public/ml-datasets/ego4d/v1/full_scale"

    args = {
        "num_devices": 1,
        "batch_size": 256,  # original 256
        "learning_rate": 0.001,
        "lr_scheduler_patience": 5,
        "early_stopping_patience": 10,
        "optimizer": "Adam",
        "num_workers": 20,  # original 8
        "random_state": 2021,
        "args.debug": False,
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
    tensorboard_logger = TensorBoardLogger(save_dir="{}/logs/".format(project_root))
    dirpath = "{}/models/".format(project_root)
    filename = "{epoch}-{val_loss:.4f}"

    # assign datasets
    if dataset == ego:
        # train_dataset =
        train_dataset = Ego4DDataset(
            data_root=args['path_to_data_root'],
            split="train",
            duration=5,
            sample_rate=sr,
        )
        val_dataset = Ego4DDataset(
            data_root=args['path_to_data_root'],
            split="valid",
            duration=5,
            sample_rate=sr,
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

    trainer = Trainer(
        logger=tensorboard_logger,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=args["early_stopping_patience"]),
            ModelCheckpoint(
                dirpath=dirpath, filename=filename, monitor="val_loss", save_top_k=-1
            ),
        ],
        devices=args["num_devices"],
        accelerator=("gpu" if torch.cuda.is_available() else "cpu"),
        max_epochs=100,
    )
    train_loader = DataLoader(
        train_dataset,
        num_workers=args["num_workers"],
        batch_size=args["batch_size"],
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )
    valid_loader = DataLoader(
        val_dataset,
        num_workers=args["num_workers"],
        batch_size=args["batch_size"],
        pin_memory=True,
        drop_last=False,
        worker_init_fn=worker_init_fn,
    )

    if dataset == ego:
        rc_grad = RCGradSavi(args, train_dataset.spec_tf.feature_shape)
    elif dataset == vgg:
        rc_grad = RCGrad(args)
    else:
        raise Exception("Not Implemented")
    # print("rcgrad", rc_grad)
    trainer.fit(rc_grad, train_loader, valid_loader)
