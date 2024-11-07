from torch.utils.data.dataset import Dataset
import torchio as tio
import os
import glob


PREPROCESSING_TRANSORMS = tio.Compose([
    tio.Clamp(out_min=-1000, out_max=400),
    tio.RescaleIntensity(in_min_max=(-1000, 400),
                         out_min_max=(-1.0, 1.0)),
    tio.CropOrPad(target_shape=(32, 64, 64))
])

PREPROCESSING_MASK_TRANSORMS = tio.Compose([
    tio.CropOrPad(target_shape=(32, 64, 64))
])

class LIDC3D_HIST_InDataset(Dataset):
    def __init__(self, root_dir='', test_txt_dir='' , augmentation=False):
        self.root_dir = root_dir
        self.remove_test_path = test_txt_dir
        self.file_names = self.get_file_names()
        self.augmentation = augmentation
        self.preprocessing_img = PREPROCESSING_TRANSORMS
        self.preprocessing_mask = PREPROCESSING_MASK_TRANSORMS


    def get_file_names(self):
        all_file_names = glob.glob(os.path.join(self.root_dir, './**/*.nii.gz'), recursive=True)

        test_file_names = set()

        with open(self.remove_test_path, 'r') as file:
            for line in file:
                test_file_name = line.strip()  
                test_file_names.add(test_file_name)

        filtered_file_names = [
            f for f in all_file_names 
            if os.path.basename(f)[:-7] not in test_file_names 
        ]
        return sorted(filtered_file_names)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):

        path = self.file_names[index] 


        img = tio.ScalarImage(path)
 
        mask_path = path.replace("/Image", "/Mask")

        filename = mask_path.split('/')[-1]
        new_filename = filename.replace("Vol_", "Mask_")


        mask_path = mask_path.replace(filename, new_filename)

        mask = tio.LabelMap(mask_path) 

        img = self.preprocessing_img(img)
        mask = self.preprocessing_mask(mask)



        return {
            'GT': img.data,
            'GT_name': filename,
            'gt_keep_mask': mask.data,
            'affine': img.affine
        }


