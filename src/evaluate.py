import os
import time
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import OUTPUT_DIR, BASELINE_DIR, SKIP_DIR, DEVICE
from dataset import KvasirSegDataset
from model import BaselineModel, SkipFusionModel, count_parameters
from data_utils import make_samples


def sync():
    if DEVICE.type == "mps":
        torch.mps.synchronize()
    elif DEVICE.type == "cuda":
        torch.cuda.synchronize()


def load(model, path):
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model.to(DEVICE).eval()


def measure_latency(model, dataset):
    """Batch size 1, 10 warm-up passes (not timed), mean over all test images."""
    with torch.no_grad():
        warm = torch.randn(1, 3, 256, 256).to(DEVICE)
        for _ in range(10):
            model(warm)
        sync()

        times = []
        for i in range(len(dataset)):
            x = dataset[i][0].unsqueeze(0).to(DEVICE)
            sync()
            t0 = time.perf_counter()
            model(x)
            sync()
            times.append(time.perf_counter() - t0)
    return 1000 * sum(times) / len(times)


def evaluate(name, model, dataset):
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    criterion = nn.CrossEntropyLoss()

    inter = [0, 0]
    union = [0, 0]
    total_loss, n = 0.0, 0
    per_image = []

    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            out = model(images)
            total_loss += criterion(out, masks).item() * images.size(0)
            n += images.size(0)
            pred = out.argmax(1)

            for c in range(2):
                inter[c] += ((pred == c) & (masks == c)).sum().item()
                union[c] += ((pred == c) | (masks == c)).sum().item()

            for p, m in zip(pred, masks):
                i = ((p == 1) & (m == 1)).sum().item()
                u = ((p == 1) | (m == 1)).sum().item()
                per_image.append(1.0 if u == 0 else i / u)

    bg = inter[0] / union[0]
    polyp = inter[1] / union[1]

    return {
        "model": name,
        "parameters": count_parameters(model),
        "test_loss": total_loss / n,
        "background_iou": bg,
        "polyp_iou": polyp,
        "mIoU": (bg + polyp) / 2,
        "latency_ms_per_image": measure_latency(model, dataset),
    }, per_image


def main():
    df = pd.read_csv(os.path.join(OUTPUT_DIR, "data_split.csv"))
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    print("Test images:", len(test_df))

    dataset = KvasirSegDataset(make_samples(test_df), augment=False)

    base = load(BaselineModel(num_classes=2),
                os.path.join(BASELINE_DIR, "best_model.pth"))
    skip = load(SkipFusionModel(num_classes=2),
                os.path.join(SKIP_DIR, "best_model.pth"))

    r_base, iou_base = evaluate("Baseline", base, dataset)
    r_skip, iou_skip = evaluate("Skip Fusion", skip, dataset)

    results = pd.DataFrame([r_base, r_skip])
    results.to_csv(os.path.join(OUTPUT_DIR, "final_results.csv"), index=False)

    per_image = pd.DataFrame({
        "test_index": range(len(test_df)),
        "image": test_df["image"],
        "baseline_polyp_iou": iou_base,
        "skip_polyp_iou": iou_skip,
    })
    per_image.to_csv(os.path.join(OUTPUT_DIR, "per_image_polyp_iou.csv"), index=False)

    print("\n" + results.to_string(index=False))
    print("\nSkip better on",
          (per_image.skip_polyp_iou > per_image.baseline_polyp_iou).sum(),
          "of", len(per_image), "test images")
    print("Latency conditions: device =", DEVICE,
          "| batch size 1 | 256x256 | 10 warm-up passes | mean of", len(dataset), "images")


if __name__ == "__main__":
    main()