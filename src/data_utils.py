import os
from config import IMAGE_DIR, MASK_DIR


def make_samples(df):
    cols = list(df.columns)

    # Case 1: the CSV already has full paths
    if "image_path" in cols and "mask_path" in cols:
        return list(zip(df["image_path"], df["mask_path"]))

    # Case 2: the CSV has only file names (first column that is not "split")
    name_col = next(c for c in cols if c != "split")
    samples = []
    for name in df[name_col]:
        name = os.path.basename(str(name))
        samples.append(
            (os.path.join(IMAGE_DIR, name), os.path.join(MASK_DIR, name))
        )
    return samples