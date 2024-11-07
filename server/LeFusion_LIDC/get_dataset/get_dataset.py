from torch.utils.data import DataLoader
from dataset import LIDC3D_HIST_Dataset, LIDC3D_HIST_InDataset, LIDC3D_HIST_InSlicerDataset

def get_slicer_inference_dataloader(dataset_root_dir, batch_size=2, drop_last=False, return_dataset=False):
    train_dataset = LIDC3D_HIST_InSlicerDataset(root_dir=dataset_root_dir)
    loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=drop_last
    )
    if return_dataset:
        return loader, train_dataset
    return loader

def get_inference_dataloader(dataset_root_dir, test_txt_dir, batch_size=1, drop_last=False, slicer=False):
    if slicer:
        # test_txt_dir unused
        return get_slicer_inference_dataloader(dataset_root_dir, batch_size=batch_size, drop_last=drop_last)
    else:
        train_dataset = LIDC3D_HIST_InDataset(root_dir=dataset_root_dir, test_txt_dir=test_txt_dir)
        loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=drop_last
        )

    return loader

def get_train_dataset(cfg):
    

    if cfg.dataset.name == 'LIDC_HIST_3D':
        train_dataset = LIDC3D_HIST_Dataset(
            root_dir=cfg.dataset.root_dir, test_txt_dir=cfg.dataset.test_txt_dir)
        val_dataset = LIDC3D_HIST_Dataset(
            root_dir=cfg.dataset.root_dir, test_txt_dir=cfg.dataset.test_txt_dir)
        sampler = None

        return train_dataset, val_dataset, sampler


    raise ValueError(f'{cfg.dataset.name} Dataset is not available')
