import json
import os
import pickle

import imageio
import numpy as np
import PIL.Image
import skimage.morphology
import torch
import torchsparse
import torchvision
from collections import defaultdict

from cvrecon import utils
from cvrecon.utils import debug_print


img_mean_rgb = np.array([127.71, 114.66, 99.32], dtype=np.float32)
img_std_rgb = np.array([75.31, 73.53, 71.66], dtype=np.float32)


def load_tsdf(tsdf_dir, scene_name):
    tsdf_fname = os.path.join(tsdf_dir, scene_name, "full_tsdf_layer0.npz")
    with np.load(tsdf_fname) as tsdf_04_npz:
        tsdf = tsdf_04_npz["arr_0"]

    pkl_fname = os.path.join(tsdf_dir, scene_name, "tsdf_info.pkl")
    with open(pkl_fname, "rb") as tsdf_pkl:
        tsdf_info = pickle.load(tsdf_pkl)
        origin = tsdf_info['vol_origin']
        voxel_size = tsdf_info['voxel_size']

    return tsdf, origin, voxel_size


def reflect_pose(pose, plane_pt=None, plane_normal=None):
    pts = pose @ np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [1, 1, 1, 1],
        ],
        dtype=np.float32,
    )
    plane_pt = np.array([*plane_pt, 1], dtype=np.float32)

    pts = pts - plane_pt[None, :, None]

    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    m = np.zeros((4, 4), dtype=np.float32)
    m[:3, :3] = np.eye(3) - 2 * plane_normal[None].T @ plane_normal[None]

    pts = m @ pts + plane_pt[None, :, None]

    result = np.eye(4, dtype=np.float32)[None].repeat(len(pose), axis=0)
    result[:, :, :3] = pts[:, :, :3] - pts[:, :, 3:]
    result[:, :, 3] = pts[:, :, 3]
    return result


def get_proj_mats(intr, pose, factors):
    k = np.eye(4, dtype=np.float32)
    k[:3, :3] = intr
    k[0] = k[0] * factors[0]
    k[1] = k[1] * factors[0]
    proj_lowres = k @ pose

    k = np.eye(4, dtype=np.float32)
    k[:3, :3] = intr
    k[0] = k[0] * factors[1]
    k[1] = k[1] * factors[1]
    proj_midres = k @ pose

    k = np.eye(4, dtype=np.float32)
    k[:3, :3] = intr
    k[0] = k[0] * factors[2]
    k[1] = k[1] * factors[2]
    proj_highres = k @ pose

    k = np.eye(4, dtype=np.float32)
    k[:3, :3] = intr
    proj_depth = k @ pose

    return {
        "coarse": proj_lowres,
        "medium": proj_midres,
        "fine": proj_highres,
        "fullres": proj_depth,
    }


def load_rgb_imgs(imgfiles, imheight, imwidth, augment=False):
    """Load and preprocess RGB images"""
    if augment:
        transforms = [
            (
                torchvision.transforms.functional.adjust_brightness,
                np.random.uniform(0.5, 1.5),
            ),
            (
                torchvision.transforms.functional.adjust_contrast,
                np.random.uniform(0.5, 1.5),
            ),
            (
                torchvision.transforms.functional.adjust_hue,
                np.random.uniform(-0.05, 0.05),
            ),
            (
                torchvision.transforms.functional.adjust_saturation,
                np.random.uniform(0.5, 1.5),
            ),
            (
                torchvision.transforms.functional.gaussian_blur,
                7,
                np.random.randint(1, 4),
            ),
        ]
        transforms = [
            transforms[i]
            for i in np.random.choice(len(transforms), size=2, replace=False)
        ]

    rgb_imgs = np.empty((len(imgfiles), imheight, imwidth, 3), dtype=np.float32)
    skipped = []
    
    for i, f in enumerate(imgfiles):
        try:
            img = PIL.Image.open(f)
            img = img.resize((imwidth, imheight), PIL.Image.BILINEAR)
            if augment:
                for t, *params in transforms:
                    img = t(img, *params)
            rgb_imgs[i] = np.array(img, dtype=np.float32) / 255.0
        except (PIL.UnidentifiedImageError, FileNotFoundError, OSError) as e:
            debug_print(f"\n[WARNING] Skipping corrupted/missing image: {f}", force=True)
            debug_print(f"Error: {str(e)}", force=True)
            skipped.append(i)
            # Use a black image as placeholder
            rgb_imgs[i] = np.zeros((imheight, imwidth, 3), dtype=np.float32)

    if len(skipped) > len(imgfiles) // 2:
        debug_print(f"\n[ERROR] Too many corrupted images ({len(skipped)}/{len(imgfiles)}). Skipping this sample.", force=True)
        return None

    # Apply normalization with their specific mean and std values
    rgb_imgs -= img_mean_rgb
    rgb_imgs /= img_std_rgb
    
    # [n_imgs, height, width, channels] -> [n_imgs, channels, height, width]
    rgb_imgs = np.transpose(rgb_imgs, (0, 3, 1, 2))
    return rgb_imgs


def load_SRfeats(scene, frame_inds):
    '''
    [64, 96, 128], [128, 48, 64], [256, 24, 32]
    '''
    # if scene == 'scene0230_00':
    #     import pdb; pdb.set_trace()
    scale0 = np.empty((len(frame_inds), 64, 96, 128), dtype=np.float32)
    scale1 = np.empty((len(frame_inds), 128, 48, 64), dtype=np.float32)
    scale2 = np.empty((len(frame_inds), 256, 24, 32), dtype=np.float32)

    for i, frame_ind in enumerate(frame_inds):
        fname = '/ScanNet2/SRfeats/' + scene + '/' + frame_ind
        if os.path.exists(fname + '_s0.npy'):
            scale0[i] = np.load(fname + '_s0.npy')
            scale1[i] = np.load(fname + '_s1.npy')
            scale2[i] = np.load(fname + '_s2.npy')
        else:
            scale0[i] = np.zeros((64, 96, 128), dtype=np.float32)
            scale1[i] = np.zeros((128, 48, 64), dtype=np.float32)
            scale2[i] = np.zeros((256, 24, 32), dtype=np.float32)
    return [scale0, scale1, scale2]


def load_SRCV(scene, frame_inds):
    '''
    [64, 96, 128]
    '''
    CV = np.empty((len(frame_inds), 64, 96, 128), dtype=np.float32)

    for i, frame_ind in enumerate(frame_inds):
        fname = '/ScanNet2/SRCV/' + scene + '/' + frame_ind
        if os.path.exists(fname + '_cv.npy'):
            CV[i] = np.load(fname + '_cv.npy')
        else:
            CV[i] = np.zeros((64, 96, 128), dtype=np.float32)
    return CV


def pose_distance(pose_b44):
    """
    DVMVS frame pose distance.
    """

    R = pose_b44[:, :3, :3]
    t = pose_b44[:, :3, 3]
    R_trace = R.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
    R_measure = torch.sqrt(2 * 
                (1 - torch.minimum(torch.ones_like(R_trace)*3.0, R_trace) / 3))
    t_measure = torch.norm(t, dim=1)
    combined_measure = torch.sqrt(t_measure ** 2 + R_measure ** 2)

    return combined_measure, R_measure, t_measure



class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        info_files,
        tsdf_dir,
        n_imgs,
        cropsize,
        augment=True,
        load_extra=False,
        split=None,
        SRfeat=False,
        SRCV=False,
        cost_volume=False,
        debug=False
    ):
        self.info_files = info_files
        self.n_imgs = n_imgs
        self.cropsize = np.array(cropsize)
        self.augment = augment
        self.load_extra = load_extra
        self.tsdf_dir = tsdf_dir

        self.tmin = 0.1
        self.rmin_deg = 15

        self.SRfeat = SRfeat
        self.SRCV = SRCV
        self.cost_volume = cost_volume
        if self.SRfeat or self.SRCV or self.cost_volume:
            self.CVDicts = defaultdict(dict)
            fname = 'data_splits/ScanNetv2/standard_split/{}.txt'.format(split)
            if split == 'test':
                fname = 'data_splits/ScanNetv2/standard_split/test_eight_view_deepvmvs_dense.txt'
            with open(fname, 'r') as f:
                lines = f.read().splitlines()
            for line in lines:
                scan_id, *frame_id = line.split(" ")
                self.CVDicts[scan_id][frame_id[0]] = frame_id[1:]
        
        self.debug = debug


    def __len__(self):
        return len(self.info_files)

    def getitem(self, ind, **kwargs):
        return self.__getitem__(ind, **kwargs)

    def __getitem__(self, ind):  # ind is the index of the scene
        with open(self.info_files[ind], "r") as f:
            info = json.load(f)

        scene_name = info["scene"]
        tsdf_04, origin, _ = load_tsdf(self.tsdf_dir, scene_name)

        rgb_imgfiles = [frame["filename_color"] for frame in info["frames"]]
        depth_imgfiles = [frame["filename_depth"] for frame in info["frames"]]
        pose = np.empty((len(info["frames"]), 4, 4), dtype=np.float32)
        for i, frame in enumerate(info["frames"]):
            pose[i] = frame["pose"]
        intr = np.array(info["intrinsics"], dtype=np.float32)

        test_img = imageio.imread(rgb_imgfiles[0])
        imheight, imwidth, _ = test_img.shape

        assert not np.any(np.isinf(pose) | np.isnan(pose))

        seen_coords = np.argwhere(np.abs(tsdf_04) < 0.999) * 0.04 + origin
        i = np.random.randint(len(seen_coords))
        anchor_pt = seen_coords[i]  # anchor of the current fragment
        offset = np.array(
            [
                np.random.uniform(0.04, self.cropsize[0] * 0.04 - 0.04),
                np.random.uniform(0.04, self.cropsize[1] * 0.04 - 0.04),
                np.random.uniform(0.04, self.cropsize[2] * 0.04 - 0.04),
            ]
        )
        minbound = anchor_pt - offset
        maxbound = minbound + self.cropsize.astype(np.float32) * 0.04

        # the GT TSDF will be sampled at these points
        x = np.arange(minbound[0], maxbound[0], .04, dtype=np.float32)
        y = np.arange(minbound[1], maxbound[1], .04, dtype=np.float32)
        z = np.arange(minbound[2], maxbound[2], .04, dtype=np.float32)
        x = x[: self.cropsize[0]]
        y = y[: self.cropsize[0]]
        z = z[: self.cropsize[0]]
        yy, xx, zz = np.meshgrid(y, x, z)
        sample_pts = np.stack([xx, yy, zz], axis=-1)  # meter as unit, global coordinate.

        flip = False
        if self.augment:
            center = np.zeros((4, 4), dtype=np.float32)
            center[:3, 3] = anchor_pt

            # rotate
            t = np.random.uniform(0, 2 * np.pi)
            R = np.array(
                [
                    [np.cos(t), -np.sin(t), 0, 0],
                    [np.sin(t), np.cos(t), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ],
                dtype=np.float32,
            )

            shape = sample_pts.shape
            sample_pts = (
                R[:3, :3] @ (sample_pts.reshape(-1, 3) - center[:3, 3]).T
            ).T + center[:3, 3]
            sample_pts = sample_pts.reshape(shape)

            # flip
            if np.random.uniform() > 0.5:
                flip = True
                sample_pts[..., 0] = -(sample_pts[..., 0] - center[0, 3]) + center[0, 3]

        selected_frame_inds = np.array(
            utils.remove_redundant(pose, self.rmin_deg, self.tmin)  # remove redundant (too small pose changes) using the NeuralRecon's strategy
        )

        ############ remove frames that are not in SRlist
        if self.SRfeat or self.SRCV or self.cost_volume:
            SRlist_inds = []
            for frame_ind in selected_frame_inds:
                if '0' + info['frames'][frame_ind]['filename_color'][-9:-4] in self.CVDicts[scene_name]:
                    SRlist_inds.append(frame_ind)
            selected_frame_inds = SRlist_inds
        ###################################################

        if self.n_imgs is not None:
            if len(selected_frame_inds) < self.n_imgs:
                # after redundant frame removal we can end up with too few frames--
                # add some back in
                # print('!!!!!!!!!!!!!!!!!', len(selected_frame_inds), scene_name)
                avail_inds = list(set(np.arange(len(pose))) - set(selected_frame_inds))
                n_needed = self.n_imgs - len(selected_frame_inds)
                extra_inds = np.random.choice(avail_inds, size=n_needed, replace=False)
                selected_frame_inds = np.concatenate((selected_frame_inds, extra_inds))
            elif len(selected_frame_inds) == self.n_imgs:
                ...
            else:
                # after redundant frame removal we still have more than the target # images--
                # remove even more. 
                pose = pose[selected_frame_inds]
                rgb_imgfiles = [rgb_imgfiles[i] for i in selected_frame_inds]
                depth_imgfiles = [depth_imgfiles[i] for i in selected_frame_inds]

                selected_frame_inds, score = utils.frame_selection(  # First remove frames that has no intersection with the current fragment, then random select n_imgs (20 for train and val)
                    pose,
                    intr,
                    imwidth,
                    imheight,
                    sample_pts.reshape(-1, 3)[::100],  # every 100th pt for efficiency
                    self.tmin,
                    self.rmin_deg,
                    self.n_imgs,
                )
        pose = pose[selected_frame_inds]
        rgb_imgfiles = [rgb_imgfiles[i] for i in selected_frame_inds]
        depth_imgfiles = [depth_imgfiles[i] for i in selected_frame_inds]

        if self.cost_volume:
            cv_invalid_mask = np.zeros(len(rgb_imgfiles), dtype=int)
            frame2id = {'0'+frame["filename_color"][-9:-4]:i for i, frame in enumerate(info["frames"])}
            for i, fname in enumerate(rgb_imgfiles.copy()):
                if '0' + fname[-9: -4] in self.CVDicts[scene_name]:
                    for frameid in self.CVDicts[scene_name]['0' + fname[-9: -4]]:
                        pose = np.concatenate((pose, np.array(info['frames'][frame2id[frameid]]['pose'], dtype=np.float32)[None,...]))
                        rgb_imgfiles.append(info['frames'][frame2id[frameid]]['filename_color'])
                else:
                    cv_invalid_mask[i] = 1
                    pose = np.concatenate((pose, np.array(info['frames'][frame2id['0'+fname[-9: -4]]]['pose'], dtype=np.float32)[None,...].repeat(7,0)))
                    rgb_imgfiles += [fname] * 7

        if self.augment:
            pose = np.linalg.inv(R) @ (pose - center) + center
            if flip:
                pose = reflect_pose(
                    pose,
                    plane_pt=center[:3, 3],
                    plane_normal=-np.array(
                        [np.cos(-t), np.sin(-t), 0], dtype=np.float32
                    ),
                )
        pose[:, :3, 3] -= minbound
        
        # scale the coordinate within current fragment to [-1, 1]
        grid = (sample_pts - origin) / (
            (np.array(tsdf_04.shape, dtype=np.float32) - 1) * 0.04
        ) * 2 - 1
        grid = grid[..., [2, 1, 0]]

        # GT TSDF of the current fragment.
        tsdf_04_n = torch.nn.functional.grid_sample(
            torch.from_numpy(tsdf_04)[None, None],
            torch.from_numpy(grid[None]),
            align_corners=False,
            mode="nearest",
        )[0, 0].numpy()

        tsdf_04_b = torch.nn.functional.grid_sample(
            torch.from_numpy(tsdf_04)[None, None],
            torch.from_numpy(grid[None]),
            align_corners=False,
            mode="bilinear",
        )[0, 0].numpy()

        # occupied area use bilinear sample, empty area set to 1
        tsdf_04 = tsdf_04_b
        inds = np.abs(tsdf_04_n) > 0.999
        tsdf_04[inds] = tsdf_04_n[inds]
        oob_inds = np.any(np.abs(grid) >= 1, axis=-1)
        tsdf_04[oob_inds] = 1

        occ_04 = np.abs(tsdf_04) < 0.999
        seen_04 = tsdf_04 < 0.999

        # seems like a bug -- dilation should happen before cropping
        occ_08 = skimage.morphology.dilation(occ_04, footprint=np.ones((3, 3, 3)))  # voxel of size 0.08 meter
        not_occ_08 = seen_04 & ~occ_08
        occ_08 = occ_08[::2, ::2, ::2]
        not_occ_08 = not_occ_08[::2, ::2, ::2]
        seen_08 = occ_08 | not_occ_08

        occ_16 = skimage.morphology.dilation(occ_08, footprint=np.ones((3, 3, 3)))
        not_occ_16 = seen_08 & ~occ_16
        occ_16 = occ_16[::2, ::2, ::2]
        not_occ_16 = not_occ_16[::2, ::2, ::2]
        seen_16 = occ_16 | not_occ_16

        rgb_imgs = load_rgb_imgs(rgb_imgfiles, imheight, imwidth, augment=self.augment)
        if rgb_imgs is None:
            debug_print(f"\n[WARNING] Skipping sample {ind} due to corrupted images", force=True)
            # Try next sample
            return self.__getitem__((ind + 1) % len(self))

        depth_imgs = np.empty((len(depth_imgfiles), imheight, imwidth), dtype=np.uint16)
        for i, f in enumerate(depth_imgfiles):
            depth_imgs[i] = imageio.imread(f)
        depth_imgs = depth_imgs / np.float32(1000)
        
        if self.SRfeat:
            SRfeat0, SRfeat1, SRfeat2 = load_SRfeats(scene_name, ['0'+x[-9:-4] for x in rgb_imgfiles])
            if self.augment and flip:
                SRfeat0 = np.ascontiguousarray(np.flip(SRfeat0, axis=-1))
                SRfeat1 = np.ascontiguousarray(np.flip(SRfeat1, axis=-1))
                SRfeat2 = np.ascontiguousarray(np.flip(SRfeat2, axis=-1))
        if self.SRCV:
            SRCV = load_SRCV(scene_name, ['0'+x[-9:-4] for x in rgb_imgfiles])
            if self.augment and flip:
                SRCV = np.ascontiguousarray(np.flip(SRCV, axis=-1))

        if self.augment and flip:
            # flip images
            depth_imgs = np.ascontiguousarray(np.flip(depth_imgs, axis=-1))
            rgb_imgs = np.ascontiguousarray(np.flip(rgb_imgs, axis=-1))
            intr[0, 0] *= -1

        inds_04 = np.argwhere(
            (tsdf_04 < 0.999) | np.all(tsdf_04 > 0.999, axis=-1, keepdims=True)  # from Atlas, penalize the areas outside the room (entire column is unseen(1))
        )
        inds_08 = np.argwhere(seen_08 | np.all(~seen_08, axis=-1, keepdims=True))
        inds_16 = np.argwhere(seen_16 | np.all(~seen_16, axis=-1, keepdims=True))

        tsdf_04 = tsdf_04[inds_04[:, 0], inds_04[:, 1], inds_04[:, 2]]
        occ_08 = occ_08[inds_08[:, 0], inds_08[:, 1], inds_08[:, 2]].astype(np.float32)
        occ_16 = occ_16[inds_16[:, 0], inds_16[:, 1], inds_16[:, 2]].astype(np.float32)

        tsdf_04 = torchsparse.SparseTensor(
            torch.from_numpy(tsdf_04), torch.from_numpy(inds_04)
        )
        occ_08 = torchsparse.SparseTensor(
            torch.from_numpy(occ_08), torch.from_numpy(inds_08)
        )
        occ_16 = torchsparse.SparseTensor(
            torch.from_numpy(occ_16), torch.from_numpy(inds_16)
        )

        cam_positions = pose[:self.n_imgs, :3, 3]

        # world to camera
        pose_w2c = np.linalg.inv(pose)

        # refers to the downsampling ratios at various levels of the CNN feature maps
        factors = np.array([1 / 16, 1 / 8, 1 / 4])
        proj_mats = get_proj_mats(intr, pose_w2c[:self.n_imgs], factors)

        # generate dense initial grid
        x = torch.arange(seen_16.shape[0], dtype=torch.int32)
        y = torch.arange(seen_16.shape[1], dtype=torch.int32)
        z = torch.arange(seen_16.shape[2], dtype=torch.int32)
        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        input_voxels_16 = torch.stack(
            (xx.flatten(), yy.flatten(), zz.flatten()), dim=-1
        )
        input_voxels_16 = torchsparse.SparseTensor(torch.zeros(0), input_voxels_16)

        # the scene has been adjusted to origin 0
        origin = np.zeros(3, dtype=np.float32)

        scene = {
            "input_voxels_16": input_voxels_16,  # dense, all zero voxel grid
            "rgb_imgs": rgb_imgs,  # random selected n_imgs (20) that have enough pose difference and at least 1 intersect with current fragment
            "cam_positions": cam_positions,
            "proj_mats": proj_mats,
            "voxel_gt_fine": tsdf_04,  # near surface area, inside objects area, and outside room area (all 1 column)
            "voxel_gt_medium": occ_08,
            "voxel_gt_coarse": occ_16,
            "scene_name": scene_name,
            "index": ind,  # index of the scene
            "depth_imgs": depth_imgs,
            "origin": origin,
        }

        if self.load_extra:
            scene.update(
                {
                    "intr_fullres": intr,
                    "pose": pose_w2c,
                }
            )
        
        if self.SRfeat:
            scene.update(
                {
                    "SRfeat0": SRfeat0,
                    "SRfeat1": SRfeat1,
                    "SRfeat2": SRfeat2,
                }
            )
        if self.SRCV:
            scene.update(
                {
                    "SRCV": SRCV,
                }
            )
        if self.cost_volume:
            k = np.eye(4, dtype=np.float32)
            k[:3, :3] = intr
            k[0] = k[0] * 0.125
            k[1] = k[1] * 0.125
            invK = np.linalg.inv(k)
            scene.update(
                {
                    "cv_k": k,
                    "cv_invK": invK,
                    "pose": pose,
                    "inv_pose": pose_w2c,
                    "cv_invalid_mask": cv_invalid_mask,
                }
            )

        # Verify depth values
        if np.isnan(depth_imgs).any():
            debug_print(f"[WARNING] NaN values found in depth images for scene {scene_name}", force=True)
            depth_imgs = np.nan_to_num(depth_imgs, nan=0.0)
        
        debug_print(f"[DEBUG] Depth stats for scene {scene_name}:", self.debug)
        debug_print(f"  min: {depth_imgs.min():.3f}, max: {depth_imgs.max():.3f}", self.debug)

        return scene


if __name__ == "__main__":

    import glob
    import yaml

    import matplotlib.pyplot as plt
    import open3d as o3d
    import skimage.measure

    import collate

    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    with open(os.path.join(config["scannet_dir"], "scannetv2_train.txt"), "r") as f:
        train_split = f.read().split()

    with open(os.path.join(config["scannet_dir"], "scannetv2_val.txt"), "r") as f:
        val_split = f.read().split()

    info_files = sorted(
        glob.glob(os.path.join(config["scannet_dir"], "scans/*/info.json"))
    )
    train_info_files = [
        f for f in info_files if os.path.basename(os.path.dirname(f)) in train_split
    ]
    val_info_files = [
        f for f in info_files if os.path.basename(os.path.dirname(f)) in val_split
    ]

    dset = Dataset(
        train_info_files,
        config["tsdf_dir"],
        35,
        (48, 48, 32),
        augment=True,
        load_extra=True,
    )

    loader = torch.utils.data.DataLoader(
        dset, batch_size=2, collate_fn=collate.sparse_collate_fn
    )
    batch = next(iter(loader))
    resolutions = {
        "coarse": 0.16,
        "medium": 0.08,
        "fine": 0.04,
    }
