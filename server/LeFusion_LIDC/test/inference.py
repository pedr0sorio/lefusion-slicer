import os
import io
import blobfile as bf
import torch as th
import json
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from ddpm import Unet3D, GaussianDiffusion_Nolatent
from get_dataset.get_dataset import get_inference_dataloader
import torchio as tio
import yaml
from omegaconf import DictConfig
import hydra

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
    device = dev(conf.get('device'))

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

    weights_dict = {}
    for k, v in (load_state_dict(os.path.expanduser(conf.model_path), map_location="cpu")["model"].items()):
        new_k = k.replace('module.', '') if 'module' in k else k
        weights_dict[new_k] = v

    diffusion.load_state_dict(weights_dict)

    if conf.use_fp16:
        model.convert_to_fp16()
    model.eval()

    show_progress = conf.show_progress

    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'hist_clusters', 'clusters.json')
    with open(file_path, 'r') as f:
        clusters = json.load(f)

    cluster_centers = clusters[0]['centers']

    print("sampling...")

    dl = get_inference_dataloader(dataset_root_dir=conf.dataset_root_dir, test_txt_dir=conf.test_txt_dir)  
    print("length of dataset:", len(dl))

    
    idx = 0
    types = 3
    for batch in iter(dl):
        for type in range(types):
            print("idx:",idx+1)
            print("type_of_cond:", type+1)
            hist = th.tensor(cluster_centers[type])
            hist = perturb_tensor(tensor=hist)
            hist = hist.unsqueeze(0)
            for k in batch.keys():
                if isinstance(batch[k], th.Tensor):
                    batch[k] = batch[k].to(device)
            model_kwargs = {}
            model_kwargs["gt"] = batch['GT']
            gt_name = batch['GT_name'][0]
            name_part, extension = gt_name.rsplit('.nii.gz', 1)
            gt_name = f"{name_part}.nii.gz"
            gt_keep_mask = batch.get('gt_keep_mask')
            if gt_keep_mask is not None:
                model_kwargs['gt_keep_mask'] = gt_keep_mask

            batch_size = model_kwargs["gt"].shape[0]


            sample_fn = diffusion.p_sample_loop_repaint

            result = sample_fn(
                shape = (batch_size, 1, 32, 64, 64),
                model_kwargs=model_kwargs,
                device=device,
                progress=show_progress,
                conf=conf,
                cond=hist
            )

            result = result.squeeze(0).cpu()


            restore_affine = batch['affine'].squeeze(0).cpu()


            origin_image = tio.ScalarImage(tensor=result, channels_last=False, affine=restore_affine)
            
            image_fold = f"Image_{type+1}"
            os.makedirs(os.path.join(conf.target_img_path, image_fold), exist_ok=True)
            origin_image.save(os.path.join(conf.target_img_path, image_fold, gt_name))

            label = batch.get('gt_keep_mask').squeeze(0).cpu()


            name_part, extension = gt_name.rsplit('.nii.gz', 1)[0], '.nii.gz'

            main_name, vol_part = name_part.rsplit('_CVol_', 1)
            mask_name = f"{main_name}_Mask_{vol_part}{extension}"

            label = tio.LabelMap(tensor=label, channels_last=False, affine=restore_affine)
            label_fold = f"Mask_{type+1}"
            os.makedirs(os.path.join(conf.target_label_path, label_fold), exist_ok=True)
            label.save(os.path.join(conf.target_label_path, label_fold, mask_name))

        idx += 1

    print("sampling complete")


if __name__ == "__main__":
    main()