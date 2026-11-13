import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_utils import make_samples

from config import (
    IMAGE_DIR,
    MASK_DIR,
    OUTPUT_DIR,
    BASELINE_DIR,
    SKIP_DIR,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    NUM_WORKERS,
    DEVICE,
    SEED,
)

from dataset import KvasirSegDataset
from model import BaselineModel, SkipFusionModel


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_iou(predictions, targets, num_classes=2):
    predictions = predictions.view(-1)
    targets = targets.view(-1)

    ious = []

    for class_id in range(num_classes):
        pred_class = predictions == class_id
        target_class = targets == class_id

        intersection = (pred_class & target_class).sum().item()
        union = (pred_class | target_class).sum().item()

        if union == 0:
            iou = 1.0
        else:
            iou = intersection / union

        ious.append(iou)

    return ious


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0

    total_intersection = torch.zeros(2, dtype=torch.float64)
    total_union = torch.zeros(2, dtype=torch.float64)

    for images, masks in loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            outputs = model(images)

            loss = criterion(outputs, masks)

            if training:
                loss.backward()
                optimizer.step()

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        predictions = torch.argmax(outputs, dim=1)

        for class_id in range(2):
            pred_class = predictions == class_id
            target_class = masks == class_id

            intersection = (pred_class & target_class).sum().item()
            union = (pred_class | target_class).sum().item()

            total_intersection[class_id] += intersection
            total_union[class_id] += union

    average_loss = total_loss / total_samples

    class_ious = []

    for class_id in range(2):
        if total_union[class_id] == 0:
            iou = 1.0
        else:
            iou = (
                total_intersection[class_id]
                / total_union[class_id]
            )

        class_ious.append(float(iou))

    mean_iou = sum(class_ious) / 2

    return average_loss, class_ious, mean_iou


def train_model(model_name, model, output_dir):
    print("\n" + "=" * 60)
    print(f"TRAINING: {model_name}")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    split_file = os.path.join(
        OUTPUT_DIR,
        "data_split.csv"
    )

    dataframe = pd.read_csv(split_file)

    train_df = dataframe[
        dataframe["split"] == "train"
    ].reset_index(drop=True)

    val_df = dataframe[
        dataframe["split"] == "validation"
    ].reset_index(drop=True)

    train_dataset = KvasirSegDataset(
        make_samples(train_df),
        augment=True,
    )

    val_dataset = KvasirSegDataset(
        make_samples(val_df),
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_val_iou = -1.0
    history = []

    for epoch in range(1, EPOCHS + 1):

        train_loss, train_ious, train_miou = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )

        val_loss, val_ious, val_miou = run_epoch(
            model,
            val_loader,
            criterion,
            None,
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train mIoU: {train_miou:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val mIoU: {val_miou:.4f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_background_iou": train_ious[0],
                "train_polyp_iou": train_ious[1],
                "train_miou": train_miou,
                "val_loss": val_loss,
                "val_background_iou": val_ious[0],
                "val_polyp_iou": val_ious[1],
                "val_miou": val_miou,
            }
        )

        if val_miou > best_val_iou:
            best_val_iou = val_miou

            torch.save(
                model.state_dict(),
                os.path.join(
                    output_dir,
                    "best_model.pth",
                ),
            )

            print(
                f"  Saved best model "
                f"(Val mIoU: {best_val_iou:.4f})"
            )

    history_df = pd.DataFrame(history)

    history_df.to_csv(
        os.path.join(
            output_dir,
            "training_history.csv",
        ),
        index=False,
    )

    print("\nTraining complete.")
    print(
        f"Best validation mIoU: "
        f"{best_val_iou:.4f}"
    )


def main():
    set_seed(SEED)

    print(f"Using device: {DEVICE}")

    baseline = BaselineModel(num_classes=2)

    train_model(
        "BASELINE MODEL",
        baseline,
        BASELINE_DIR,
    )

    set_seed(SEED)

    skip_fusion = SkipFusionModel(num_classes=2)

    train_model(
        "SKIP FUSION MODEL",
        skip_fusion,
        SKIP_DIR,
    )


if __name__ == "__main__":
    main()