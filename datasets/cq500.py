import os
from pathlib import Path
from utils.utils import window
from utils.masking import calculate_masks
from torch.utils.data import Dataset
from monai.transforms import IdentityD, Compose, AddChannelD, ResizeD, ScaleIntensityD, LoadImageD


class CQ500(Dataset):

    def __init__(self, root, mode="train", stripped=False, augmentations=None, spatial_size=128):
        # get the correct root paths
        self.mode = mode
        if mode == "train":
            folder = "TRAIN"
        elif mode == "val":
            folder = "VAL"
        elif mode == "test-healthy":
            folder = os.path.join("TEST", "HEALTHY")
        elif mode == "test-ATY":
            folder = os.path.join("TEST", "ATY")
        elif mode == "test-MASS":
            folder = os.path.join("TEST", "MASS")
        elif mode == "test-OTHER":
            folder = os.path.join("TEST", "OTHER")
        elif mode == "test-ICH":
            folder = os.path.join("TEST", "ICH")
        elif mode == "test-ISCH":
            folder = os.path.join("TEST", "ISCH")
        elif mode == "test-FRAC":
            folder = os.path.join("TEST", "FRAC")
        elif mode == "test-TOTAL":
            folder = os.path.join("TEST", "TOTAL")
        else:
            raise NameError("The specified dataset mode is not expected. Specify either train, val or test")

        images_root = os.path.join(root, folder, "head_volumes")
        stripped_images_root = os.path.join(root, folder, "stripped_volumes")

        # save specifics
        self.stripped = stripped
        self.spatial_size = spatial_size
        
        # save the augmentation functions, with identity in position 0
        # SO FAR NO AUGMENTATIONS
        self.augmentations = [IdentityD(keys=["image"])]
        if augmentations is not None:
            self.augmentations.extend(augmentations)

        # data multiplies with the number of augmentation functions
        self.data_multiplier = len(self.augmentations)  

        # get the names of all images
        image_names = sorted(os.listdir(images_root))
        stripped_image_names = sorted(os.listdir(stripped_images_root))

        # save the complete paths to the individual images and labels
        self.image_paths = [os.path.join(images_root, x) for x in image_names]
        self.stripped_image_paths = [os.path.join(stripped_images_root, x) for x in stripped_image_names]

    def __len__(self):
        return len(self.image_paths) * self.data_multiplier

    def __getitem__(self, index):
        # find out if augmentation needed or not
        augmentation_type = index // len(self.image_paths)
        path_index = index % len(self.image_paths)

        # load the image and label
        path = self.stripped_image_paths[path_index] if self.stripped else self.image_paths[path_index]
        data = {"image": path}
        loading = Compose(
            [
                LoadImageD(keys=["image"], reader="NibabelReader"),
                AddChannelD(keys=["image"]),
            ]
        )
        data = loading(data)

        # window
        center = 50
        width = 100
        windowed = window(data["image"], center, width)
        data["image"] = windowed

        # resizing because 3D
        resizing = Compose(
            [
                ResizeD(keys=["image"],
                        spatial_size=(self.spatial_size, self.spatial_size, self.spatial_size)),
            ]
        )
        output = resizing(data)
        
        if "HEALTHY" in self.image_paths[index]:
            output["label"] = 0
        else:
            output["label"] = 1
        
        # add scan number for debugging
        output["number"] = int(Path(self.image_paths[path_index]).name.split(".")[0].split('_')[1])
        return output
