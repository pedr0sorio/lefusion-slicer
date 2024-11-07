import os
from pathlib import Path
import io
import blobfile as bf
import torch as th
import json
import sys


import torchio as tio
import yaml
from omegaconf import DictConfig
import hydra
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
print(f"parent_dir: {parent_dir}")
from ddpm import Unet3D, GaussianDiffusion_Nolatent
from get_dataset.get_dataset import get_inference_dataloader
from paths import IN_SERVER_DATA_DIR, OUT_SERVER_DATA_DIR


def dev(device):
    if device is None:
        if th.cuda.is_available():
            return th.device(f"cuda")
        return th.device("cpu")
    return th.device(device)


def load_state_dict(path, backend=None, **kwargs):
    with bf.BlobFile(path, "rb") as f:
        data = f.read()
    return th.load(io.BytesIO(data), **kwargs)

try:
    import ctypes
    libgcc_s = ctypes.CDLL('libgcc_s.so.1')
except:
    pass


def perturb_tensor(tensor, mean=0.0, std=1.0, bili=0.1):
    perturbation = th.normal(mean, std, size=tensor.size())
    perturbation -= perturbation.mean()
    max_perturbation = tensor.abs() * bili
    perturbation = perturbation / perturbation.abs().max() * max_perturbation
    perturbed_tensor = tensor + perturbation
    return perturbed_tensor


@hydra.main(config_path='confs', config_name='infer', version_base=None)
def main(conf: DictConfig):
    # Manage absence of save and load directories
    conf.target_img_path = Path(conf.target_img_path)
    if conf.target_img_path is None:
        conf.target_img_path = OUT_SERVER_DATA_DIR

    conf.dataset_root_dir = Path(conf.dataset_root_dir)
    if conf.dataset_root_dir is None:
        conf.dataset_root_dir = IN_SERVER_DATA_DIR

    device = dev(conf.get('device'))

    # Define model
    model = Unet3D(
        dim=conf.diffusion_img_size,
        dim_mults=conf.dim_mults,
        channels=conf.diffusion_num_channels,
        cond_dim=16,
    )
    diffusion = GaussianDiffusion_Nolatent(
        model,
        image_size=conf.diffusion_img_size,
        num_frames=conf.diffusion_depth_size,
        channels=conf.diffusion_num_channels,
        timesteps=conf.timesteps,
        loss_type=conf.loss_type,
    )
    diffusion.to(device)

    # Load pretrained weights
    path_to_weights = Path(parent_dir) / conf.model_path
    weights_dict = {}
    for k, v in (load_state_dict(path_to_weights, map_location="cpu")["model"].items()):
        new_k = k.replace('module.', '') if 'module' in k else k
        weights_dict[new_k] = v
    diffusion.load_state_dict(weights_dict)
    # Enforce eval mode and fp16 if needed
    if conf.use_fp16:
        model.convert_to_fp16()
    model.eval()
    # Set progress bar
    show_progress = conf.show_progress
    # Load TEXTURE clusters
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'hist_clusters', 'clusters.json')
    with open(file_path, 'r') as f:
        clusters = json.load(f)
    cluster_centers = clusters[0]['centers']

    # Define dataloader
    dl = get_inference_dataloader(
        dataset_root_dir=conf.dataset_root_dir,
        test_txt_dir=conf.test_txt_dir,
        batch_size=conf.batch_size,
        slicer=conf.slicer # NOTE defines whether the loaded data is from slicer or not
    )
    n_batches = len(dl)
    print(f"Length of dataset: {n_batches}")
    print(f"Batch Size: {conf.batch_size}")

    # Sampling Loop
    print("sampling...")
    idx = 0
    for batch in iter(dl):
        print("Batch %i / %i" % (idx + 1, n_batches))

        histogram_types = batch['histogram']
        print(f"selected texture clusters (1, 2 or 3): {histogram_types}")
        # TODO allow different conds for different elements in the batch. For now, all 
        # elements in the batch will have the same cond
        type_ = histogram_types[0] # use the first cond in the batch for all elements
        hist = th.tensor(cluster_centers[type_ - 1])
        print(f"{perturb_tensor(tensor=hist).shape = }")
        hist = perturb_tensor(tensor=hist)
        print(f"{hist.shape = }")
        hist = hist.unsqueeze(0)
        print(f"{hist.shape = }")

        # Send everything to the GPU
        for k in batch.keys():
            if isinstance(batch[k], th.Tensor):
                batch[k] = batch[k].to(device)

        # Defining model kwargs
        model_kwargs = {}
        model_kwargs["gt"] = batch['GT'] # Batch of crops
        gt_keep_mask = batch.get('gt_keep_mask') # Batch of masks
        if gt_keep_mask is not None:
            model_kwargs['gt_keep_mask'] = gt_keep_mask

        # Get this batch's size because of drop_last=True
        batch_size = model_kwargs["gt"].shape[0]

        # Get sample function from model class
        sample_fn = diffusion.p_sample_loop_repaint
        # Run Sampling
        result = sample_fn(
            shape = (batch_size, 1, 32, 64, 64),
            model_kwargs=model_kwargs,
            device=device,
            progress=show_progress,
            conf=conf,
            cond=hist
        )

        # Batch of INPAINTED crop scans to CPU
        result = result.cpu()

        # Batch of masks to CPU
        label = batch.get('gt_keep_mask').cpu()

        # Batch of bbox idxs to CPU
        bbox_kji = batch.get('bbox_kji').cpu()

        # Save results
        for b_idx in range(batch_size):
            # Get and post process inpainted crop scan
            # inpianted image sto original slicer format. 
            # i.e. undo rescaling tfs and remove channel dimension
            pp_result = dl.dataset.postprocessing_img(result[b_idx])[0]
            
            # Get bbox idxs so that their available for slicer without having to be 
            # in memory
            bbox_kji_ = bbox_kji[b_idx]

            # Get base .npz file name
            gt_name = os.path.basename(batch['GT_name'][b_idx])
            print(f"Saving sample in batch {b_idx} with name {gt_name}")

            # Save data for slicer
            np.savez_compressed(
                conf.target_img_path / gt_name,
                data_inpainted=pp_result,
                mask=label[b_idx],
                histogram=hist,
                boxes_numpy=bbox_kji_
            )

        idx += 1
        if conf.debug:
            raise SystemExit("Debugging mode. Exiting after first batch.")
            break


    print("sampling complete")


if __name__ == "__main__":
    main()