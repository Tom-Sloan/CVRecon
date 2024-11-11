import os

import numpy as np
import open3d as o3d
import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt
from cvrecon import collate, data, utils, cvrecon


class FineTuning(pl.callbacks.Callback):
    def __init__(self, initial_epochs, use_cost_volume=False):
        super().__init__()
        self.initial_epochs = initial_epochs
        self.use_cost_volume = use_cost_volume
        self._frozen_modules = []

    def on_fit_start(self, trainer, pl_module):
        modules = [
            pl_module.cvrecon.cnn2d.conv0,
            pl_module.cvrecon.cnn2d.conv1,
            pl_module.cvrecon.cnn2d.conv2,
            pl_module.cvrecon.upsampler,
        ] + ([
            pl_module.cvrecon.matching_encoder,
            pl_module.cvrecon.cost_volume.mlp.net[:4],
        ] if self.use_cost_volume else [])
        
        for mod in modules:
            self._freeze_module(mod)
            self._frozen_modules.append(mod)

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch >= self.initial_epochs:
            # Unfreeze modules
            for mod in self._frozen_modules:
                self._unfreeze_module(mod)
            
            # Update learning rate
            optimizer = trainer.optimizers[0]
            for group in optimizer.param_groups:
                group["lr"] = pl_module.config["finetune_lr"]
            
            pl_module.cvrecon.use_proj_occ = True

    def _freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False
        module.eval()

    def _unfreeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = True
        module.train()


class LightningModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        print(f"Initializing LightningModel with config: {config}")
        self.save_hyperparameters()
        self.cvrecon = cvrecon.cvrecon(
            config["attn_heads"], config["attn_layers"], config["use_proj_occ"], config["SRfeat"],
            config["SR_vi_ebd"], config["SRCV"], config["cost_volume"], config["cv_dim"], 
            config["cv_overall"], config["depth_head"],
        )
        self.config = config
        self.automatic_optimization = True

    def configure_optimizers(self):
        print(f"Configuring optimizers")
        return torch.optim.Adam(
            [param for param in self.parameters() if param.requires_grad],
            lr=self.config["initial_lr"],
        )

    def step(self, batch, batch_idx):
        try:
            voxel_coords_16 = batch["input_voxels_16"].C
            voxel_outputs, proj_occ_logits, bp_data, depth_out = self.cvrecon(batch, voxel_coords_16)
            loss, logs = self.cvrecon.losses(
                voxel_outputs,
                batch,
                proj_occ_logits,
                bp_data,
                batch["depth_imgs"],
                depth_out,
            )
            
            # Ensure all tensors in logs are on GPU
            logs = {k: v.to(voxel_coords_16.device) if isinstance(v, torch.Tensor) else v 
                   for k, v in logs.items()}
            
            return loss, logs, voxel_outputs

        except Exception as e:
            print(f"Error in step:")
            print(f"Error type: {type(e)}")
            print(f"Error message: {str(e)}")
            raise e

    def training_step(self, batch, batch_idx):
        try:
            loss, logs, _ = self.step(batch, batch_idx)
            
            # Ensure all metrics are on GPU
            logs = {k: v.to(loss.device) if isinstance(v, torch.Tensor) else v 
                   for k, v in logs.items()}
            logs['loss'] = loss.to(loss.device)
            
            self.log_dict(
                logs,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
                batch_size=1
            )
            
            return loss

        except Exception as e:
            print(f"Error in training_step:")
            print(f"Error type: {type(e)}")
            print(f"Error message: {str(e)}")
            print(f"Device info:")
            print(f"- CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"- Current device: {torch.cuda.current_device()}")
                print(f"- Device name: {torch.cuda.get_device_name()}")
            raise e

    def validation_step(self, batch, batch_idx):
        try:
            loss, logs, voxel_outputs = self.step(batch, batch_idx)
            
            # Ensure all metrics are on GPU
            logs = {k: v.to(loss.device) if isinstance(v, torch.Tensor) else v 
                   for k, v in logs.items()}
            logs['loss'] = loss.to(loss.device)
            
            self.log_dict(
                logs,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=1
            )
            
            return logs

        except Exception as e:
            print(f"Error in validation_step:")
            print(f"Error type: {type(e)}")
            print(f"Error message: {str(e)}")
            print(f"Device info:")
            print(f"- CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"- Current device: {torch.cuda.current_device()}")
                print(f"- Device name: {torch.cuda.get_device_name()}")
            raise e

    def train_dataloader(self):
        print("Train dataloader called")
        return self.dataloader("train", augment=True)

    def val_dataloader(self):
        print("Val dataloader called")
        return self.dataloader("test")

    def dataloader(self, split, augment=False):
        print(f"Dataloader called with split {split}")
        nworkers = self.config["nworkers"]
        if split in ["val", "test"]:
            batch_size = 1
            nworkers //= 2
        elif self.current_epoch < self.config["initial_epochs"]:
            batch_size = self.config["initial_batch_size"]
        else:
            batch_size = self.config["finetune_batch_size"]

        print("nworkers: ", nworkers, "batch_size: ", batch_size)
        info_files = utils.load_info_files(self.config["scannet_dir"], split)
        dset = data.Dataset(
            info_files,
            self.config["tsdf_dir"],
            self.config[f"n_imgs_{split}"],
            self.config[f"crop_size_{split}"],
            augment=augment,
            split=split,
            SRfeat=self.config["SRfeat"],
            SRCV=self.config["SRCV"],
            cost_volume=self.config["cost_volume"],
        )
        
        # Get first sample
        sample = dset[0]
        
        # Assuming 'sample' is your dictionary
        for key, value in sample.items():
            print(f"Key: {key}, Type: {type(value)}")
        # Plot first 10 samples
        for i in range(min(10, len(dset))):
            sample = dset[i]
            depth_imgs = sample["depth_imgs"]
            rgb_imgs = sample["rgb_imgs"] 
            
            # Create figure with subplots for each image pair
            fig, axes = plt.subplots(2, len(depth_imgs), figsize=(4*len(depth_imgs), 8))
            
            for j in range(len(depth_imgs)):
                # Convert depth image to tensor and permute
                axes[0].imshow(depth_imgs[j], cmap='gray')
                axes[0].set_title(f'Depth {j}')
                axes[0].axis('off')
                
                # Convert RGB image to tensor and permute
                rgb_img = torch.from_numpy(rgb_imgs[j]).permute(1, 2, 0)  # Change shape from (C, H, W) to (H, W, C)

                # Normalize the RGB image to the range [0, 1]
                rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())

                # Ensure the values are clipped to the range [0, 1]
                rgb_img = torch.clamp(rgb_img, 0, 1)

                axes[1].imshow(rgb_img)
                axes[1].set_title(f'RGB {j}')
                axes[1].axis('off')
                
            plt.tight_layout()
            # Create the data_sampled directory if it doesn't exist
            output_dir = "data_sampled"
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f'sample_{i}.png'))
            plt.close()

        return torch.utils.data.DataLoader(
            dset,
            batch_size=batch_size,
            num_workers=nworkers,
            collate_fn=collate.sparse_collate_fn,
            drop_last=True,
            #persistent_workers=True,
        )


def write_mesh(outfile, logits_04):
    print(f"Writing mesh to")
    batch_mask = logits_04.C[:, 3] == 0
    inds = logits_04.C[batch_mask, :3].cpu().numpy()
    tsdf_logits = logits_04.F[batch_mask, 0].cpu().numpy()
    tsdf = 1.05 * np.tanh(tsdf_logits)
    tsdf_vol = utils.to_vol(inds, tsdf)

    mesh = utils.to_mesh(tsdf_vol, voxel_size=0.04, level=0, mask=~np.isnan(tsdf_vol))
    o3d.io.write_triangle_mesh(outfile, mesh)
