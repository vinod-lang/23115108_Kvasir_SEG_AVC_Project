import os
import pandas as pd
import torch
import matplotlib.pyplot as plt

from config import OUTPUT_DIR, BASELINE_DIR, SKIP_DIR, PREDICTION_DIR, DEVICE
from dataset import KvasirSegDataset
from model import BaselineModel, SkipFusionModel
from data_utils import make_samples

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load(model, path):
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model.to(DEVICE).eval()


def save_figure(idx, dataset, base, skip, per_image, out_path):
    image, mask = dataset[idx]
    x = image.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pb = base(x).argmax(1)[0].cpu().numpy()
        ps = skip(x).argmax(1)[0].cpu().numpy()

    shown = (image * STD + MEAN).permute(1, 2, 0).clamp(0, 1).numpy()
    row = per_image.iloc[idx]

    fig, ax = plt.subplots(1, 4, figsize=(12, 3.4))
    ax[0].imshow(shown); ax[0].set_title("Original")
    ax[1].imshow(mask.numpy(), cmap="gray"); ax[1].set_title("Ground truth")
    ax[2].imshow(pb, cmap="gray")
    ax[2].set_title(f"Baseline\nIoU {row.baseline_polyp_iou:.2f}")
    ax[3].imshow(ps, cmap="gray")
    ax[3].set_title(f"Skip Fusion\nIoU {row.skip_polyp_iou:.2f}")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    os.makedirs(PREDICTION_DIR, exist_ok=True)

    df = pd.read_csv(os.path.join(OUTPUT_DIR, "data_split.csv"))
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    dataset = KvasirSegDataset(make_samples(test_df), augment=False)

    per_image = pd.read_csv(os.path.join(OUTPUT_DIR, "per_image_polyp_iou.csv"))

    base = load(BaselineModel(num_classes=2), os.path.join(BASELINE_DIR, "best_model.pth"))
    skip = load(SkipFusionModel(num_classes=2), os.path.join(SKIP_DIR, "best_model.pth"))

    # 10 fixed examples: the first 10 test images, with no cherry-picking
    for i in range(10):
        save_figure(i, dataset, base, skip, per_image,
                    os.path.join(PREDICTION_DIR, f"example_{i+1:02d}.png"))

    # 5 worst failures for the skip-fusion model
    worst = per_image.sort_values("skip_polyp_iou").head(5).index
    for rank, i in enumerate(worst, 1):
        save_figure(i, dataset, base, skip, per_image,
                    os.path.join(PREDICTION_DIR, f"failure_{rank:02d}.png"))

    print("Saved figures in", PREDICTION_DIR)


if __name__ == "__main__":
    main()