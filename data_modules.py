from pytorch_lightning.core.datamodule import LightningDataModule
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import os
from torch.utils.data import DataLoader
import tifffile as tiff
import random


IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff', '.TIFF', '.tif', '.TIF']


class CycleGANDataset(Dataset):
    def __init__(self, images_folder_A, images_folder_B, transforms_A=None, transforms_B=None):
        self.images_folder_A = images_folder_A
        self.images_folder_B = images_folder_B
        self.transforms_A = transforms_A
        self.transforms_B = transforms_B

        # get paths of all images
        self.paths_A = self.get_image_paths(self.images_folder_A)
        self.paths_B = self.get_image_paths(self.images_folder_B)

        # save size of A and B for __getitem__ and __len__ method
        self.len_A = len(self.paths_A)
        self.len_B = len(self.paths_B)

        print(f"Initialized dataset with {self.len_A} images in A and {self.len_B} images in B")

    def __len__(self):
        return self.len_A

    def __getitem__(self, idx):
        A = Image.open(self.paths_A[idx])
        A = np.asarray(A).astype(np.float32)
        idx_B = np.random.randint(0, self.len_B-1)  # get random image to avoid fixed pairs
        B = Image.open(self.paths_B[idx_B])
        B = np.asarray(B).astype(np.float32)

        if self.transforms_A:
            A = self.transforms_A(image=A)["image"]
        if self.transforms_B:
            B = self.transforms_B(image=B)["image"]
        return {"A": A, "B": B, "A_stem": self.paths_A[idx].stem, "B_stem": self.paths_B[idx_B].stem}  # stem is only needed for inference

    def get_image_paths(self, folder):
        image_paths = []
        for (root, _, filenames) in os.walk(folder):
            for filename in filenames:
                if any(filename.endswith(extension) for extension in IMG_EXTENSIONS):
                    image_paths.append(Path(os.path.join(root, filename)))
        return image_paths


class ToTensor3D(ImageOnlyTransform):
    def __init__(self, always_apply=True, p=1.0):
        super(ToTensor3D, self).__init__(always_apply, p)

    def apply(self, image, **params):
        img = torch.from_numpy(image)
        return img


class MinMaxRangeNormalize(ImageOnlyTransform):
    """Normalization is applied by the formula: `img = (max-min)/max_pixel_value*img+min`
    Standard parameters convert an image in range [0,255] to range [0,1]
    Args:
        min (float, list of float): min values of output image
        max  (float, list of float): max values of output image
        max_pixel_value (float): maximum possible pixel value of input image
    Targets:
        image
    Image types:
        uint8, float32
    """

    def __init__(self, min=0, max=1, max_pixel_value=255.0, always_apply=False, p=1.0):
        super(MinMaxRangeNormalize, self).__init__(always_apply, p)
        self.min = np.array(min, dtype=np.float32)
        self.max = np.array(max, dtype=np.float32)
        self.max_pixel_value = max_pixel_value
        # min_pixel_value is allways 0 for normal images not needed here
        # for fast calculations calculate a for a*img+min=(max-min)/max_pixel_value*img+min once
        self.a = (self.max-self.min)/self.max_pixel_value

    def apply(self, image, **params):
        img = self.a*image+self.min
        return img

    def get_transform_init_args_names(self):
        return ("min", "max", "max_pixel_value")


class GrayToRGB(ImageOnlyTransform):
    """ Expands the grayscale dimensions to RGB
    """

    def __init__(self, always_apply=False, p=1.0):
        super(GrayToRGB, self).__init__(always_apply, p)

    def apply(self, image, **params):
        img = np.concatenate([image[..., np.newaxis]]*3, axis=-1)
        return img

    def get_transform_init_args_names(self):
        return ("")


class CycleGANDataModule(LightningDataModule):
    def __init__(self, batch_size, num_workers, image_folder_train_A, image_folder_train_B,
                 image_folder_test_A, image_folder_test_B, transforms_train_A, transforms_train_B, transforms_test_A,
                 transforms_test_B, min_A, max_A, min_B, max_B):
        """ Generate LightningDataModule that prepares the dataloaders

        Args:
            batch_size (int): Batch size used for training and evaluation
            num_workers (int): Number of workers used for training and evaluation

            image_folder_train_A (Path): Pathlib Path to image folder train A
            image_folder_train_B (Path): Pathlib Path to image folder train B
            image_folder_test_A (Path): Pathlib Path to image folder test A
            image_folder_test_B (Path): Pathlib Path to image folder test B
            transforms_train_A (List): List of albumentations transformations used for training data A
            transforms_train_B (List): List of albumentations transformations used for training data B
            transforms_test_A (List): List of albumentations transformations used for test data A
            transforms_test_B (List): List of albumentations transformations used for test data B
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_folder_train_A = image_folder_train_A
        self.image_folder_train_B = image_folder_train_B
        self.image_folder_test_A = image_folder_test_A
        self.image_folder_test_B = image_folder_test_B
        self.transforms_train_A = transforms_train_A
        self.transforms_train_B = transforms_train_B
        self.transforms_test_A = transforms_test_A
        self.transforms_test_B = transforms_test_B
        self.min_A = min_A
        self.max_A = max_A
        self.min_B = min_B
        self.max_B = max_B

    def train_dataloader(self):
        train_dataset = CycleGANDataset(images_folder_A=self.image_folder_train_A,
                                        images_folder_B=self.image_folder_train_B,
                                        transforms_A=A.Compose(self.transforms_train_A),
                                        transforms_B=A.Compose(self.transforms_train_B))
        return DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        test_dataset = CycleGANDataset(images_folder_A=self.image_folder_test_A,
                                       images_folder_B=self.image_folder_test_B,
                                       transforms_A=A.Compose(self.transforms_test_A),
                                       transforms_B=A.Compose(self.transforms_test_B))
        return DataLoader(test_dataset, batch_size=1, num_workers=self.num_workers)

  
class CycleGANDataModule3D(LightningDataModule):
    def __init__(self, batch_size, num_workers, image_path_train_A, image_path_train_B,
                 image_path_test_A, image_path_test_B, crop_size, transforms_train_A, transforms_train_B,
                 transforms_test_A, transforms_test_B, min_A, max_A, min_B, max_B):
        """ Generate LightningDataModule that prepares the dataloaders

        Args:
            batch_size (int): Batch size used for training and evaluation
            num_workers (int): Number of workers used for training and evaluation

            image_path_train_A (Path): Pathlib Path to image folder train A
            image_path_train_B (Path): Pathlib Path to image folder train B
            image_path_test_A (Path): Pathlib Path to image path test A
            image_path_test_B (Path): Pathlib Path to image path test B
            crop_size (int or list[int]): Size of the crop x,y,z loaded from the big image
            transforms_train_A (List): List of albumentations transformations used for training data A
            transforms_train_B (List): List of albumentations transformations used for training data B
            transforms_test_A (List): List of albumentations transformations used for test data A
            transforms_test_B (List): List of albumentations transformations used for test data B
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_path_train_A = self.get_images(image_path_train_A)
        self.image_path_train_B = self.get_images(image_path_train_B)
        self.image_path_test_A = self.get_images(image_path_test_A)
        self.image_path_test_B = self.get_images(image_path_test_B)
        self.crop_size = crop_size
        self.transforms_train_A = transforms_train_A
        self.transforms_train_B = transforms_train_B
        self.transforms_test_A = transforms_test_A
        self.transforms_test_B = transforms_test_B
        self.min_A = min_A
        self.max_A = max_A
        self.min_B = min_B
        self.max_B = max_B

    def train_dataloader(self):
        train_dataset = CycleGANDataset3D(image_path_A=self.image_path_train_A, image_path_B=self.image_path_train_B,
                                          crop_size=self.crop_size,
                                          transforms_A=A.Compose(self.transforms_train_A), transforms_B=A.Compose(self.transforms_train_B))
        return DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        test_dataset = CycleGANDataset3D(image_path_A=self.image_path_train_A,
                                         image_path_B=self.image_path_train_B,
                                         crop_size=self.crop_size,
                                         transforms_A=A.Compose(self.transforms_test_A),
                                         transforms_B=A.Compose(self.transforms_test_B))
        return DataLoader(test_dataset, batch_size=1, num_workers=self.num_workers)
    
    def get_images(self, folder):
        """Extract all tiff images in folder

        Args:
            folder (Pathlib Path): Folder with images

        Returns:
            list[Pathlib Path]: List of images in folder
        """
        images = list(folder.glob("*.tif"))
        
        return images

class CycleGANDataset3D(Dataset):
    def __init__(self, image_path_A, image_path_B, virtual_num_images=256, crop_size=[64, 64, 64], transforms_A=None, transforms_B=None):
        """ Dataset for 3D data. Crops are randomly loaded from one volume for each image domain.

        Args:
            image_path_A (list[Path]): Path to the images from domain A
            image_path_B (list[Path]): Path to the images from domain B
            virtual_num_images (int, optional): Number of random crops for each epoch. Defaults to 256.
            crop_size (list[int], optional): Size of the crop as list [Z, X, Y]. Defaults to 64.
            transforms_A (transforms,  optional): Transformations performed on A. Defaults to None.
            transforms_B (transforms, optional): Transformations performed on B. Defaults to None.
        """
        self.image_path_A = image_path_A
        self.image_path_B = image_path_B
        self.transforms_A = transforms_A  # Z,X,Y
        self.transforms_B = transforms_B  # Z,X,Y
        self.virtual_num_images = virtual_num_images
        self.crop_size = crop_size

        # Need to find out if memmap for multiple workers needs to be performed in getitem
        self.img_A = [tiff.memmap(path_A) for path_A in self.image_path_A]
        self.img_B = [tiff.memmap(path_B) for path_B in self.image_path_B]

        self.img_A_shape = [img_A.shape for img_A in self.img_A]
        self.img_B_shape = [img_B.shape for img_B in self.img_B]
        print(f"Initialized dataset with {self.virtual_num_images} virtual images per epoch.")

    def __len__(self):
        return self.virtual_num_images

    def __getitem__(self, idx):
        id_A = random.randint(0, len(self.img_A)-1)
        id_B = random.randint(0, len(self.img_B)-1)
        crop_a = self.get_random_crop(self.img_A_shape[id_A])
        crop_b = self.get_random_crop(self.img_B_shape[id_B])

        a = np.array(self.img_A[id_A][crop_a]).astype(np.float32)
        b = np.array(self.img_B[id_B][crop_b]).astype(np.float32)

        # Add channel dimension if no 3D RGB image
        # If 3D RGB, RGB dim needs to be at 0!
        if a.ndim == 3:
            a = np.expand_dims(a, 0)
        if b.ndim == 3:
            b = np.expand_dims(b, 0)

        if self.transforms_A:
            a = self.transforms_A(image=a)["image"]
        if self.transforms_B:
            b = self.transforms_B(image=b)["image"]
        return {"A": a, "B": b, "A_stem": self.image_path_A[id_A].stem, "B_stem": self.image_path_B[id_B].stem}  # stem is only needed for inference

    def get_random_crop(self, img_shape):
        # create crop according to crop size and image shape
        if (self.crop_size is None) or (self.crop_size == [None, None, None]):
            return np.s_[0:img_shape[0], 0:img_shape[1], 0:img_shape[2]]
        
        start_z = np.random.randint(0, img_shape[0]-self.crop_size[0]+1)
        start_x = np.random.randint(0, img_shape[1]-self.crop_size[1]+1) # +1 because (start, end]
        start_y = np.random.randint(0, img_shape[2]-self.crop_size[2]+1)
        return np.s_[start_z:start_z+self.crop_size[0], start_x:start_x+self.crop_size[1], start_y:start_y+self.crop_size[2]]
            

def get_data_module(module_name, batch_size, num_workers, crop_size, minmax_scale_A, minmax_scale_B):
    
    # expand crop_size if only one value is given. 3rd value not needed for 2D data
    if crop_size == None:
        crop_size = [None, None, None]
    if len(crop_size) == 1:
        crop_size = [crop_size[0], crop_size[0], crop_size[0]]
    
    min_A = minmax_scale_A[0]
    max_A = minmax_scale_A[1]
    min_B = minmax_scale_B[0]
    max_B = minmax_scale_B[1]
    if "BenchmarkColorRGBDataModule" == module_name:
        transforms_train_A = [A.RandomCrop(crop_size[0], crop_size[1], p=1),
                              A.ToFloat(max_value=np.iinfo(np.uint16).max),  # output is in range [0,1]
                              MinMaxRangeNormalize(min=min_A, max=max_A, max_pixel_value=1),
                              GrayToRGB(p=1),
                              ToTensorV2()]
        transforms_train_B = [A.RandomCrop(crop_size[0], crop_size[1], p=1),
                              A.ToFloat(max_value=np.iinfo(np.uint8).max),  # output is in range [0,1]
                              MinMaxRangeNormalize(min=min_B, max=max_B, max_pixel_value=1),
                              ToTensorV2()]
        transforms_test_A = [A.ToFloat(max_value=np.iinfo(np.uint16).max),  # output is in range [0,1]
                             MinMaxRangeNormalize(min=min_A, max=max_A, max_pixel_value=1),
                             GrayToRGB(p=1),
                             ToTensorV2()]
        transforms_test_B = [A.RandomCrop(crop_size[0], crop_size[1], p=1),
                             A.ToFloat(max_value=np.iinfo(np.uint8).max),  # output is in range [0,1]
                             MinMaxRangeNormalize(min=min_B, max=max_B, max_pixel_value=1),
                             ToTensorV2()]
        image_folder_train_A = Path(__file__).parent.joinpath("datasets/tiling_strategy_benchmark_1/labels_train")
        image_folder_train_B = Path(__file__).parent.joinpath("datasets/tiling_strategy_benchmark_2/targets")
        image_folder_test_A = image_folder_train_A
        image_folder_test_B = image_folder_train_B
        return CycleGANDataModule(batch_size, num_workers, image_folder_train_A, image_folder_train_B,
                                  image_folder_test_A, image_folder_test_B, transforms_train_A, transforms_train_B,
                                  transforms_test_A, transforms_test_B, min_A, max_A, min_B, max_B)
    elif "realworldDataModule" == module_name:
        transforms_train_A = [A.ToFloat(max_value=np.iinfo(np.uint8).max),  # output is in range [0,1]
                              MinMaxRangeNormalize(min=min_A, max=max_A, max_pixel_value=1),
                              ToTensor3D()]
        transforms_train_B = [A.ToFloat(max_value=np.iinfo(np.uint8).max),  # output is in range [0,1]
                              MinMaxRangeNormalize(min=min_B, max=max_B, max_pixel_value=1),
                              ToTensor3D()]
        transforms_test_A = [A.ToFloat(max_value=np.iinfo(np.uint8).max),  # output is in range [0,1]
                             MinMaxRangeNormalize(min=min_A, max=max_A, max_pixel_value=1),
                             ToTensor3D()]
        transforms_test_B = [A.ToFloat(max_value=np.iinfo(np.uint8).max),  # output is in range [0,1]
                             MinMaxRangeNormalize(min=min_B, max=max_B, max_pixel_value=1),
                             ToTensor3D()]
        image_path_train_A = Path(__file__).parent.joinpath("datasets/real_world/A")
        image_path_train_B = Path(__file__).parent.joinpath("datasets/real_world/B")
        image_path_test_A = image_path_train_A
        image_path_test_B = image_path_train_B
        return CycleGANDataModule3D(batch_size, num_workers, image_path_train_A, image_path_train_B,
                                    image_path_test_A, image_path_test_B, crop_size, transforms_train_A, transforms_train_B,
                                    transforms_test_A, transforms_test_B, min_A, max_A, min_B, max_B)
    elif "realworldDataModule2D" == module_name:
        transforms_train_A = [A.RandomCrop(crop_size[0], crop_size[1], p=1),
                              A.ToFloat(max_value=np.iinfo(np.uint8).max),  # output is in range [0,1]
                              MinMaxRangeNormalize(min=min_A, max=max_A, max_pixel_value=1),
                              ToTensorV2()]
        transforms_train_B = [A.RandomCrop(crop_size[0], crop_size[1], p=1),
                              A.ToFloat(max_value=np.iinfo(np.uint8).max),  # output is in range [0,1]
                              MinMaxRangeNormalize(min=min_B, max=max_B, max_pixel_value=1),
                              ToTensorV2()]
        transforms_test_A = [A.ToFloat(max_value=np.iinfo(np.uint8).max),  # output is in range [0,1]
                             MinMaxRangeNormalize(min=min_A, max=max_A, max_pixel_value=1),
                             ToTensorV2()]
        transforms_test_B = [A.RandomCrop(crop_size[0], crop_size[1], p=1),
                             A.ToFloat(max_value=np.iinfo(np.uint8).max),  # output is in range [0,1]
                             MinMaxRangeNormalize(min=min_B, max=max_B, max_pixel_value=1),
                             ToTensorV2()]
        image_folder_train_A = Path(__file__).parent.joinpath("datasets/real_world_2D/A")
        image_folder_train_B = Path(__file__).parent.joinpath("datasets/real_world_2D/B")
        image_folder_test_A = image_folder_train_A
        image_folder_test_B = image_folder_train_B
        return CycleGANDataModule(batch_size, num_workers, image_folder_train_A, image_folder_train_B,
                                  image_folder_test_A, image_folder_test_B, transforms_train_A, transforms_train_B,
                                  transforms_test_A, transforms_test_B, min_A, max_A, min_B, max_B)
    
    else:
        raise ValueError(f'{module_name} not implemented')
