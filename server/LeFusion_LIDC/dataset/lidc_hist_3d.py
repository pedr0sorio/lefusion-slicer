import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os
import glob
import cv2
import torchio as tio
import matplotlib.pyplot as plt



PREPROCESSING_TRANSORMS = tio.Compose([
    tio.Clamp(out_min=-1000, out_max=400),
    tio.RescaleIntensity(in_min_max=(-1000, 400),
                         out_min_max=(-1.0, 1.0)),
    tio.CropOrPad(target_shape=(32, 64, 64))
])

PREPROCESSING_MASK_TRANSORMS = tio.Compose([
    tio.CropOrPad(target_shape=(32, 64, 64))
])

TRAIN_TRANSFORMS = tio.Compose([
    tio.RandomFlip(axes=(1), flip_probability=0.5),
])

class LIDC3D_HIST_Dataset(Dataset):
    def __init__(self, root_dir='', test_txt_dir = '',augmentation=False):
        self.root_dir = root_dir
        self.remove_test_path = test_txt_dir
        self.file_names = self.get_file_names()
        self.augmentation = augmentation
        self.preprocessing_img = PREPROCESSING_TRANSORMS
        self.preprocessing_mask = PREPROCESSING_MASK_TRANSORMS
    def train_transform(self, image, label, p):
        TRAIN_TRANSFORMS = tio.Compose([
            tio.RandomFlip(axes=(1), flip_probability=p),
        ])
        image = TRAIN_TRANSFORMS(image)
        label = TRAIN_TRANSFORMS(label)
        return image, label

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
        return filtered_file_names

    def __len__(self):
        return len(self.file_names)

    @staticmethod
    def create_mask(shape):
        return torch.zeros(shape, dtype=torch.uint8)

    @staticmethod 
    def project_to_2d(mask):
        projection = torch.max(mask, dim=0)[0]
        return projection.numpy()

    @staticmethod
    def min_enclosing_circle(projection):

        points = np.column_stack(np.where(projection > 0))
        points = points.astype(np.float32)
        print(points.shape)       
        (x, y), radius = cv2.minEnclosingCircle(points.astype(np.float32))
        center = (int(x), int(y))
        radius = int(radius)
        return center, radius

    @staticmethod
    def create_circle_mask_2d(shape, center, radius):
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.circle(mask, center, radius, 1, thickness=-1)
        return mask

    @staticmethod
    def apply_circle_mask_to_3d(mask, circle_mask_2d):
        for i in range(mask.shape[0]):
            mask[i] = torch.from_numpy(circle_mask_2d)
        return mask


    def __getitem__(self, index):
        path = self.file_names[index]

        img = tio.ScalarImage(path)
        mask_path = path.replace("/Image/", "/Mask/")
        filename = mask_path.split('/')[-1]
        new_filename = filename.replace("Vol_", "Mask_")

        mask_path = mask_path.replace(filename, new_filename)
        mask = tio.LabelMap(mask_path) 

        img = self.preprocessing_img(img)
        mask = self.preprocessing_mask(mask)

        p = np.random.choice([0, 1])

        img, mask = self.train_transform(img, mask, p)


        mask = mask.data
        img = img.data
        hist = torch.histc(img[mask > 0], bins=16, min=-1, max=1) / mask.sum()
        if torch.sum(hist) == 0 or torch.isnan(hist).any():
            print(index, mask.sum(), "----", hist)
            print(img[mask > 0])


        return {
            'data': img,
            'label': mask,
            'hist': hist,
        }

def test_dataset():
    dataset = LIDC3D_HIST_Dataset(root_dir='')
    sample = dataset[0]  
    mask = sample['label']

    fig = plt.figure(figsize=(10, 5))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.voxels(mask[0].numpy(), edgecolor='k')
    ax1.set_title('Mask')

    plt.show()

if __name__ == '__main__':
    test_dataset()