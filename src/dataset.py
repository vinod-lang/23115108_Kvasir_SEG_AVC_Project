import random

import numpy as np

import torch

from PIL import Image

from torch.utils.data import Dataset


from torchvision.transforms import functional as TF

from torchvision.transforms import InterpolationMode


from config import (
    IMAGE_DIR,
    MASK_DIR,
    IMAGE_SIZE
)


class KvasirSegDataset(Dataset):

    def __init__(
        self,
        samples,
        augment=False
    ):

        self.samples = samples

        self.augment = augment


    def __len__(self):

        return len(self.samples)


    def __getitem__(self, index):

        image_path, mask_path = self.samples[index]


        # -----------------------------
        # Load image
        # -----------------------------

        image = Image.open(
            image_path
        ).convert("RGB")


        # -----------------------------
        # Load mask
        # -----------------------------

        mask = Image.open(
            mask_path
        ).convert("L")


        # -----------------------------
        # Resize
        # -----------------------------

        image = TF.resize(
            image,
            [IMAGE_SIZE, IMAGE_SIZE],
            interpolation=InterpolationMode.BILINEAR
        )


        mask = TF.resize(
            mask,
            [IMAGE_SIZE, IMAGE_SIZE],
            interpolation=InterpolationMode.NEAREST
        )


        # -----------------------------
        # Data augmentation
        # -----------------------------

        if self.augment:

            if random.random() < 0.5:

                image = TF.hflip(image)

                mask = TF.hflip(mask)


        # -----------------------------
        # Image → Tensor
        # -----------------------------

        image = TF.to_tensor(image)


        # -----------------------------
        # Normalize image
        # -----------------------------

        image = TF.normalize(
            image,
            mean=[
                0.485,
                0.456,
                0.406
            ],
            std=[
                0.229,
                0.224,
                0.225
            ]
        )


        # -----------------------------
        # Mask → Tensor
        # -----------------------------

        mask = np.array(
            mask,
            dtype=np.uint8
        )


        # Convert:
        # 0 → background
        # >0 → polyp

        mask = (
            mask > 127
        ).astype(np.int64)


        mask = torch.from_numpy(
            mask
        )


        return image, mask