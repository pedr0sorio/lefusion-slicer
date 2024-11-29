"""
python lefusion/lidc-idri-processing/prepare_dataset.py \
lidc_dicom_path=/home/pedroosorio/data/LIDC-IDRI/images/TCIA_LIDC-IDRI_20200921/LIDC-IDRI \
debug=true
"""
import os
from pathlib import Path

import pandas as pd
import numpy as np # because np. version is >=1.24.0 and pylidc uses np.int which is deprected then
np.int = np.int32
np.float = np.float64
np.bool = np.bool_
from omegaconf import DictConfig, OmegaConf
import hydra
import warnings
import pylidc as pl
from tqdm import tqdm
from statistics import median_high, median, mean
from pylidc.utils import consensus
import torchio as tio

from lefusion.paths import TRAIN_DATA_DIR  


warnings.filterwarnings(action='ignore')

def is_dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


class MakeInpaintingDataSet:
    def __init__(self, LIDC_Patients_list, image_path, mask_path, meta_path, padding, training_resolution, crop_dim, confidence_level=0.5):
        self.IDRI_list = LIDC_Patients_list
        self.img_path = image_path
        self.mask_path = mask_path
        self.meta_path = meta_path
        self.c_level = confidence_level
        self.padding = padding
        self.processing_tfs = tio.Compose(
            [
                tio.Clamp(out_min=-1000, out_max=400),
                tio.Resample(
                    target=training_resolution, image_interpolation="bspline", label_interpolation="nearest"
                ),
                tio.CropOrPad(target_shape=crop_dim), # center crop
            ]
        )


    def calculate_malignancy(self,nodule):
        # Calculate the malignancy of a nodule with the annotations made by 4 doctors. 
        # Return median high of the annotated cancer, True or False label for cancer
        # if median high is above 3, we return a label True for cancer
        # if it is below 3, we return a label False for non-cancer
        # if it is 3, we return ambiguous
        list_of_malignancy =[annotation.malignancy for annotation in nodule]

        malignancy = median_high(list_of_malignancy)
        is_cancer = "True" if malignancy > 3 else "False" if malignancy < 3 else "Ambiguous"
        return {
            "malignancy_median_high": malignancy,
            "is_cancer": is_cancer
        }

    def aggregate_annotation_labels(self, nodule, mode = "median_high"):
        # Calculate the malignancy of a nodule with the annotations made by 4 doctors. 
        # Return median high of the annotated cancer, True or False label for cancer
        # if median high is above 3, we return a label True for cancer
        # if it is below 3, we return a label False for non-cancer
        # if it is 3, we return ambiguous
        # TODO consider whether making this variables continuous helps convergence

        # define the mode of aggregation
        if mode == "median_high":
            agg_fn = median_high
        elif mode == "median":
            agg_fn = median
        elif mode == "mean":
            agg_fn = mean
        else:
            raise ValueError(f"Mode {mode} not supported.")

        agg_labels = {}
        for lbl in pl.annotation_feature_names:
            lbl_val = [getattr(annotation, lbl, None) for annotation in nodule]
            agg_labels[lbl] = agg_fn(lbl_val)

        return agg_labels

    def aggregate_contour_labels(self, nodule, mode = "mean"):

        # define the mode of aggregation
        if mode == "median_high":
            agg_fn = median_high
        elif mode == "median":
            agg_fn = median
        elif mode == "mean":
            agg_fn = mean
        else:
            raise ValueError(f"Mode {mode} not supported.")

        agg_shape_metrics = {}
        for lbl in ["diameter", "surface_area", "volume"]:
            lbl_val = [getattr(annotation, lbl, None) for annotation in nodule]
            agg_shape_metrics[lbl] = agg_fn(lbl_val)

        return agg_shape_metrics

    def prepare_dataset(self):
        # Make directory
        if not os.path.exists(self.img_path):
            os.makedirs(self.img_path)
        if not os.path.exists(self.mask_path):
            os.makedirs(self.mask_path)
        if not os.path.exists(self.meta_path):
            os.makedirs(self.meta_path)

        IMAGE_DIR = Path(self.img_path)
        MASK_DIR = Path(self.mask_path)

        meta_dict_list = []
        for patient in tqdm(self.IDRI_list):
            pid = patient #LIDC-IDRI-0001~
            scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
            
            # Define matrix to convert voxel coordinates to world coordinates
            scan_res = (scan.pixel_spacing, scan.pixel_spacing, scan.slice_spacing)
            affine = np.eye(4)
            affine[:3, :3] = np.diag(scan_res)

            if not (scan.slice_spacing == scan.slice_thickness):
                warnings.warn(
                    f"Warning........... {pid = }: Slice spacing ({scan.slice_spacing})"
                    f" and slice thickness ({scan.slice_thickness}) are not equal."
                )

            nodules_annotation = scan.cluster_annotations()
            vol = scan.to_volume()
            print("Patient ID: {} Dicom Shape: {} Number of Annotated Nodules: {}".format(pid,vol.shape,len(nodules_annotation)))

            patient_image_dir = IMAGE_DIR / pid
            patient_mask_dir = MASK_DIR / pid
            Path(patient_image_dir).mkdir(parents=True, exist_ok=True)
            Path(patient_mask_dir).mkdir(parents=True, exist_ok=True)


            if len(nodules_annotation) > 0:
                # Patients with nodules
                for nodule_idx, nodule in enumerate(nodules_annotation):
                    # Call nodule images. Each Patient will have at maximum 4 annotations as there are only 4 doctors
                    # This current for loop iterates over total number of nodules in a single patient

                    # ... Get nodule crop & masks ...
                    # TODO For each nodule get crop fixed dim crop around the nodule with extra padding
                    # to avoid resampling artifacts in edges e.g. only need 
                    # crop of 80.0 mm around the nodule, so that we can resample freely
                    # without voxel index conversion
                    mask, cbbox, masks = consensus(nodule, self.c_level, self.padding)
                    lung_np_array = vol[cbbox] # Pre-cropped, nodule centred volume 80mm isotropic

                    # Build the tio nodule subject to facilitate the processing
                    nodule_subject = tio.Subject(
                        nodule_crop=tio.ScalarImage(tensor=lung_np_array[None, ...], affine = affine),
                        nodule_mask=tio.LabelMap(tensor=mask[None, ...], affine = affine),
                    )

                    # Resample & Centre Crop nodule crop & mask to 1mm x 1mm x 1mm 
                    # and 64x64x32 dimensions. Use tio to process mask and vol together
                    nodule_subject_tfed = self.processing_tfs(nodule_subject)

                    # Convert vols to kji to match expected order in training code
                    nodule_subject_tfed.nodule_crop.set_data(
                        np.moveaxis(
                            nodule_subject_tfed.nodule_crop.data.numpy(), [1,2], [2,3]
                        )
                    )
                    nodule_subject_tfed.nodule_mask.set_data(
                        np.moveaxis(
                            nodule_subject_tfed.nodule_mask.data.numpy(), [1,2], [2,3]
                        )
                    )

                    # Save masks and lesion to nii.gz
                    crop_path = patient_image_dir / f"{pid}-{str(nodule_idx).zfill(2)}.nii.gz"
                    nodule_subject_tfed.nodule_crop.save(
                        crop_path
                    )
                    mask_path = patient_mask_dir / f"{pid}-{str(nodule_idx).zfill(2)}.nii.gz"
                    nodule_subject_tfed.nodule_mask.save(
                        mask_path
                    )

                    # ... Get nodule metadata ...
                    meta_dict = {
                        "patient_id": pid,
                        "is_clean": False,
                        "nodule_no": nodule_idx,
                        "crop_path": crop_path,
                        "mask_path": mask_path,
                    }
                    # We aggreagate all annotation labels based on median
                    agg_lbl_dict = self.aggregate_annotation_labels(nodule, mode="median_high")

                    # We calculate the malignancy information based on median high of labels
                    is_cancer_dict = self.calculate_malignancy(nodule)

                    # Get the nodule shape metrics
                    agg_shape_lbl_dict = self.aggregate_contour_labels(nodule, mode="mean")

                    meta_dict = {**meta_dict, **agg_lbl_dict, **is_cancer_dict, **agg_shape_lbl_dict}

                    meta_dict_list.append(meta_dict)

            else:
                print(f"Clean Volume: Skipping Patient ID {pid}")
                # ... Get nodule metadata ...
                meta_dict = {
                    "patient_id": pid,
                    "is_clean": True,
                }
                meta_dict_list.append(meta_dict)


        print("Saved Meta data: ")
        df_meta = pd.DataFrame.from_records(meta_dict_list)
        df_meta.to_csv(Path(self.meta_path) / 'meta_info.csv', index=False)


@hydra.main(config_path='confs', config_name='lidc-1iso-64x64x32.yaml')
def main(conf: DictConfig):
    # Manage absence of save directories
    if conf.prepare_dataset.image_path is None:
        conf.prepare_dataset.image_path = TRAIN_DATA_DIR / "Image"

    if conf.prepare_dataset.mask_path is None:
        conf.prepare_dataset.mask_path = TRAIN_DATA_DIR / "Mask"

    if conf.prepare_dataset.meta_path is None:
        conf.prepare_dataset.meta_path = TRAIN_DATA_DIR / "Meta"

    print("[Configuration Settings]")
    print(OmegaConf.to_yaml(conf))

    LIDC_IDRI_list = [f for f in os.listdir(conf.lidc_dicom_path) if not f.startswith('.')]
    if conf.debug:
        LIDC_IDRI_list = LIDC_IDRI_list[:4]
    LIDC_IDRI_list.sort()

    # Instantiate the dataset builder & procces the data
    dataset_builder = MakeInpaintingDataSet(LIDC_Patients_list = LIDC_IDRI_list, **conf.prepare_dataset)
    dataset_builder.prepare_dataset()

    # Save the configuration file in the parent directory of conf.image_path
    config_save_path = Path(conf.prepare_dataset.image_path).parent / 'config.yaml'
    with open(config_save_path, 'w') as config_file:
        config_file.write(OmegaConf.to_yaml(conf))

if __name__ == "__main__":
    main()