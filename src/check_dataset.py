import sys

from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent)
)


import matplotlib.pyplot as plt

import torch


from dataset import KvasirSegDataset

from config import (
    IMAGE_DIR,
    MASK_DIR
)


def main():

    image_files = sorted(
        IMAGE_DIR.glob("*.jpg")
    )


    samples = []


    for image_path in image_files:

        mask_path = (
            MASK_DIR /
            image_path.name
        )


        if mask_path.exists():

            samples.append(
                (
                    image_path,
                    mask_path
                )
            )


    dataset = KvasirSegDataset(
        samples=samples,
        augment=False
    )


    image, mask = dataset[0]


    # Undo normalization

    image = (
        image *
        torch.tensor(
            [
                0.229,
                0.224,
                0.225
            ]
        ).view(3, 1, 1)
        +
        torch.tensor(
            [
                0.485,
                0.456,
                0.406
            ]
        ).view(3, 1, 1)
    )


    image = image.clamp(
        0,
        1
    )


    image = image.permute(
        1,
        2,
        0
    )


    plt.figure(
        figsize=(10, 5)
    )


    plt.subplot(
        1,
        2,
        1
    )

    plt.imshow(image)

    plt.title(
        "Kvasir-SEG Image"
    )

    plt.axis("off")


    plt.subplot(
        1,
        2,
        2
    )

    plt.imshow(
        mask,
        cmap="gray"
    )

    plt.title(
        "Ground Truth Polyp Mask"
    )

    plt.axis("off")


    plt.tight_layout()


    output = (
        Path(__file__).resolve().parent.parent
        / "outputs"
        / "dataset_check.png"
    )


    plt.savefig(
        output,
        dpi=200
    )


    print(
        "Saved:",
        output
    )


if __name__ == "__main__":

    main()