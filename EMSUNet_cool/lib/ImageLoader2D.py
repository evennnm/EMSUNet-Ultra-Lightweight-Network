import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2


class PolypDataset(Dataset):

    def __init__(self, folder_path, img_size=(512, 512), images_to_load=-1):
        self.img_size = img_size
        self.target_size = (img_size[1], img_size[0])
        self.images_to_load = images_to_load

        self.image_files, self.mask_files = self._detect_dataset_structure(folder_path)


        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ])


        self._validate_sample_files()

    def _detect_dataset_structure(self, folder_path):

        img_exts = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']
        mask_exts = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif']


        possible_structures = [

            {
                'images': os.path.join(folder_path, 'image'),
                'masks': os.path.join(folder_path, 'masks')
            },

            {
                'images': os.path.join(folder_path, 'images'),
                'masks': os.path.join(folder_path, 'masks')
            },

            {
                'images': os.path.join(folder_path, 'Images'),
                'masks': os.path.join(folder_path, 'Masks')
            },

            {
                'images': os.path.join(folder_path, 'train', 'images'),
                'masks': os.path.join(folder_path, 'train', 'masks')
            },

            {
                'images': folder_path,
                'masks': folder_path
            }
        ]


        for structure in possible_structures:
            images_dir = structure['images']
            masks_dir = structure['masks']

            if os.path.exists(images_dir) and os.path.exists(masks_dir):
                image_files = []
                for ext in img_exts:
                    image_files.extend(glob.glob(os.path.join(images_dir, f'*{ext}')))
                    image_files.extend(glob.glob(os.path.join(images_dir, f'*{ext.upper()}')))

                if not image_files:
                    continue


                image_mask_pairs = []
                for img_path in image_files:
                    base_name = os.path.basename(img_path)
                    base_name_no_ext = os.path.splitext(base_name)[0]


                    mask_found = False
                    for ext in mask_exts:
                        mask_path = os.path.join(masks_dir, f"{base_name_no_ext}{ext}")
                        if os.path.exists(mask_path):
                            image_mask_pairs.append((img_path, mask_path))
                            mask_found = True
                            break


                    if not mask_found:
                        for ext in mask_exts:
                            mask_path = os.path.join(masks_dir, f"{base_name_no_ext}{ext.upper()}")
                            if os.path.exists(mask_path):
                                image_mask_pairs.append((img_path, mask_path))
                                mask_found = True
                                break

                if image_mask_pairs:

                    if self.images_to_load > 0 and self.images_to_load < len(image_mask_pairs):
                        image_mask_pairs = image_mask_pairs[:self.images_to_load]


                    image_files, mask_files = zip(*image_mask_pairs)
                    return list(image_files), list(mask_files)


        available_dirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
        raise FileNotFoundError(
            f"Unable to detect a valid dataset structure! Directory contents: {available_dirs}\n"
            "Please ensure that the dataset contains images and corresponding mask files."
        )

    def _validate_sample_files(self):
        import random
        sample_size = min(5, len(self.image_files))
        sample_indices = random.sample(range(len(self.image_files)), sample_size)

        for idx in sample_indices:
            img_path = self.image_files[idx]
            mask_path = self.mask_files[idx]


            try:
                self._read_image(img_path)
            except Exception as e:
                raise IOError(f"Unable to read image file: {img_path} - {str(e)}")


            try:
                self._read_mask(mask_path)
            except Exception as e:
                raise IOError(f"Unable to read mask file: {mask_path} - {str(e)}")

    def _read_image(self, path):
        """Read image file"""
        try:
            return Image.open(path).convert('RGB')
        except:
            img_arr = cv2.imread(path)
            if img_arr is None:
                raise ValueError(f"Unable to read image: {path}")
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
            return Image.fromarray(img_arr)

    def _read_mask(self, path):

        try:
            mask = Image.open(path).convert('L')
            return np.array(mask)
        except:
            mask_arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if mask_arr is None:
                raise ValueError(f"Unable to read mask file: {path}")
            return mask_arr

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        mask_path = self.mask_files[idx]


        image = self._read_image(img_path)
        mask_arr = self._read_mask(mask_path)


        image = self.transform(image)


        mask = Image.fromarray(mask_arr).convert('L')
        mask = mask.resize(self.target_size, Image.NEAREST)
        mask_arr = np.array(mask, dtype=np.uint8)


        mask_arr = (mask_arr >= 127).astype(np.uint8)
        mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0)

        return image, mask_tensor