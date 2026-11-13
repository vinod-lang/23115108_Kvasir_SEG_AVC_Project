from pathlib import Path

import random

import csv


from config import (
    IMAGE_DIR,
    MASK_DIR,
    SEED,
    OUTPUT_DIR
)


def main():

    # ----------------------------------
    # Find all images
    # ----------------------------------

    image_files = sorted(
        IMAGE_DIR.glob("*.jpg")
    )


    # ----------------------------------
    # Match each image with its mask
    # ----------------------------------

    samples = []


    for image_path in image_files:

        mask_path = (
            MASK_DIR /
            image_path.name
        )


        if not mask_path.exists():

            print(
                "WARNING: Missing mask:",
                image_path.name
            )

            continue


        samples.append(
            (
                image_path,
                mask_path
            )
        )


    print(
        "Valid image-mask pairs:",
        len(samples)
    )


    # ----------------------------------
    # Reproducible shuffle
    # ----------------------------------

    rng = random.Random(
        SEED
    )

    rng.shuffle(samples)


    # ----------------------------------
    # 70 / 15 / 15 split
    # ----------------------------------

    total = len(samples)

    train_end = int(
        total * 0.70
    )

    val_end = int(
        total * 0.85
    )


    train = samples[
        :train_end
    ]

    validation = samples[
        train_end:val_end
    ]

    test = samples[
        val_end:
    ]


    print(
        "Training:",
        len(train)
    )

    print(
        "Validation:",
        len(validation)
    )

    print(
        "Testing:",
        len(test)
    )


    # ----------------------------------
    # Save split file
    # ----------------------------------

    split_file = (
        OUTPUT_DIR /
        "data_split.csv"
    )


    with open(
        split_file,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "split",
                "image",
                "mask"
            ]
        )


        for image_path, mask_path in train:

            writer.writerow(
                [
                    "train",
                    image_path.name,
                    mask_path.name
                ]
            )


        for image_path, mask_path in validation:

            writer.writerow(
                [
                    "validation",
                    image_path.name,
                    mask_path.name
                ]
            )


        for image_path, mask_path in test:

            writer.writerow(
                [
                    "test",
                    image_path.name,
                    mask_path.name
                ]
            )


    print(
        "Saved split file:",
        split_file
    )


if __name__ == "__main__":

    main()