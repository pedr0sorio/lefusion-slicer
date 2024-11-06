from torch.utils.data.dataset import Dataset
import torchio as tio
import os
import glob
import numpy as np

import pathlib

from paths import SERVER_ASSET_DATA_DIR


PREPROCESSING_TRANSORMS = tio.Compose([
    tio.Clamp(out_min=-1000, out_max=400),
    tio.RescaleIntensity(in_min_max=(-1000, 400),
                         out_min_max=(-1.0, 1.0)),
    tio.CropOrPad(target_shape=(32, 64, 64))
])

POSTROCESSING_TRANSORMS = tio.Compose([
    tio.RescaleIntensity(in_min_max=(-1.0, 1.0), 
                         out_min_max=(-1000, 400)),
])

PREPROCESSING_MASK_TRANSORMS = tio.Compose([
    tio.CropOrPad(target_shape=(32, 64, 64))
])

class LIDC3D_HIST_InSlicerDataset(Dataset):
    """Dataset that iterates over ROIs in 3D volumes from .npz files."""
    def __init__(self, root_dir='', augmentation=False, use_diffmask=False):
        self.root_dir = root_dir # Path to folder with .npz files
        # NOTE one .npz file is one ROI from a given volume

        # Iterate over ROIs through all volumes -> Define list of ROIs and og volume
        self.npz_file_names = sorted(glob.glob(os.path.join(self.root_dir, './*.npz'), recursive=True))
        if len(self.npz_file_names) == 0:
            raise ValueError(f"No .npz files found in {self.root_dir}")

        self.augmentation = augmentation
        self.preprocessing_img = PREPROCESSING_TRANSORMS
        self.postprocessing_img = POSTROCESSING_TRANSORMS
        self.preprocessing_mask = PREPROCESSING_MASK_TRANSORMS

        # Whether to infer mask with DiffMask
        self.use_diffmask = use_diffmask # TODO Add when weight become available
        self.template_mask_path = SERVER_ASSET_DATA_DIR / "LIDC-IDRI-0032_CMask_21.nii.gz"

    def load_npz_data(self, path):
        """Load .npz sample scan file with bbox"""
        npz_data = np.load(path, 'r', allow_pickle=True)
        return npz_data

    def __len__(self):
        """Number of ROIs to be inpainted based on Slicer's annotated ROIs."""
        return len(self.npz_file_names)

    def __getitem__(self, index):
        
        # Get crop path
        npz_data_path = self.npz_file_names[index]
        # ... Load volume from path
        npz_data = self.load_npz_data(path = npz_data_path)
        crop_scan = npz_data['data'][None, ...] # add channel dim for tio tfs
        affine = npz_data['affine']
        histogram = npz_data['histogram']
        bbox_kji = npz_data['boxes_numpy']

        if self.use_diffmask:
            # Infer mask from bbox with DiffMask model
            # TODO Implement DiffMask (await LeFusion guys otherwise train our own)
            NotImplementedError("DiffMask not implemented yet")
        else:
            # Load a fixed mask for testing
            mask = tio.LabelMap(self.template_mask_path) 

        crop_scan = self.preprocessing_img(crop_scan)
        mask = self.preprocessing_mask(mask)

        return {
            'GT': crop_scan,
            'GT_name': npz_data_path,
            'gt_keep_mask': mask.data,
            'affine': affine,
            'histogram': histogram,
            'bbox_kji': bbox_kji,
        }


