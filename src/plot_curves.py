import os
import pandas as pd
import matplotlib.pyplot as plt
from config import OUTPUT_DIR, BASELINE_DIR, SKIP_DIR

b = pd.read_csv(os.path.join(BASELINE_DIR, "training_history.csv"))
s = pd.read_csv(os.path.join(SKIP_DIR, "training_history.csv"))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(b.epoch, b.val_loss, label="Baseline")
ax[0].plot(s.epoch, s.val_loss, label="Skip Fusion")
ax[0].set_title("Validation loss"); ax[0].set_xlabel("Epoch"); ax[0].legend()

ax[1].plot(b.epoch, b.val_miou, label="Baseline")
ax[1].plot(s.epoch, s.val_miou, label="Skip Fusion")
ax[1].set_title("Validation mIoU"); ax[1].set_xlabel("Epoch"); ax[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=150)
print("Saved training_curves.png")