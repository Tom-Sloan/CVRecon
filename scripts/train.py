import argparse
import glob
import json
import os
import random
import subprocess

import numpy as np
import pytorch_lightning as pl
import torch
import torchsparse
import yaml
from pytorch_lightning.strategies import DDPStrategy

from cvrecon import collate, data, lightningmodel, utils


def print_gpu_memory():
    if torch.cuda.is_available():
        print("\nGPU Memory Usage:")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_properties(i).name}")
            print(f"- Allocated: {torch.cuda.memory_allocated(i)/1024**2:.1f}MB")
            print(f"- Cached: {torch.cuda.memory_reserved(i)/1024**2:.1f}MB")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--devices", type=int, default=1)
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    pl.seed_everything(config["seed"])
    
    if config['wandb_runid'] is not None:
        logger = pl.loggers.WandbLogger(project=config["wandb_project_name"], config=config, id=config['wandb_runid'], resume="must")
    else:
        logger = pl.loggers.WandbLogger(project=config["wandb_project_name"], config=config)
    subprocess.call(
        [
            "zip",
            "-q",
            os.path.join(str(logger.experiment.dir), "code.zip"),
            "config.yml",
            *glob.glob("cvrecon/*.py"),
            *glob.glob("scripts/*.py"),
        ]
    )
    
    ckpt_dir = os.path.join(str(logger.experiment.dir), "ckpts")
    checkpointer = pl.callbacks.ModelCheckpoint(
        save_last=True,
        dirpath=ckpt_dir,
        filename='{epoch}-{val/voxel_loss_medium:.4f}',
        verbose=True,
        save_top_k=20,
        monitor="val/voxel_loss_medium",
    )
    callbacks = [checkpointer, lightningmodel.FineTuning(config["initial_epochs"], config["cost_volume"])]
    
    if config["use_amp"]:
        amp_kwargs = {"precision": "16-mixed"}
    else:
        amp_kwargs = {"precision": "32-true"}
    
    # Create model first
    model = lightningmodel.LightningModel(config)

    # Add configuration debug info
    print("Model configuration:")
    print(f"- Using {args.devices} devices")
    print(f"- Batch size: {config['initial_batch_size']}")
    print(f"- Crop size: {config['crop_size_train']}")

    # Validate data dimensions before trainer setup
    print("\nValidating data dimensions:")
    try:
        sample_batch = next(iter(model.train_dataloader()))
        print(f"Sample batch keys: {sample_batch.keys()}")
        for key, value in sample_batch.items():
            if isinstance(value, torch.Tensor):
                print(f"- {key}: shape={value.shape}, dtype={value.dtype}")
            elif isinstance(value, torchsparse.SparseTensor):
                print(f"- {key}: F={value.F.shape}, C={value.C.shape}, stride={value.stride}")
    except Exception as e:
        print(f"Error during data validation: {str(e)}")
        raise e

    # Add memory tracking
    print_gpu_memory()

    # Then create trainer with proper settings
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.devices,
        logger=logger,
        max_epochs=config["initial_epochs"] + config["finetune_epochs"],
        check_val_every_n_epoch=5,
        detect_anomaly=True,  # Add anomaly detection
        callbacks=callbacks,
        reload_dataloaders_every_n_epochs=1,
        strategy=DDPStrategy(find_unused_parameters=True),
        accumulate_grad_batches=config["accu_grad"],
        num_sanity_val_steps=1,
        gradient_clip_val=1.0,  # Add gradient clipping
        **amp_kwargs,
    )

    trainer.fit(model, ckpt_path=config["ckpt"])
