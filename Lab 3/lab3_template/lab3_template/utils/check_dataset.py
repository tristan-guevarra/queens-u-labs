"""
ELEC 475: Lab 3 - Dataset Reality Check

Validates every sample of both splits (shape and label bounds) and saves a
figure of eight training images with their nose labels to plots/dataset_check.png.

Run from the lab folder:
    python -m utils.check_dataset
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from utils.dataset import IMG_SIZE, PetNoseDataset, denormalize


def validate(split: str) -> int:
    """Check every image/label pair in a split. Returns the sample count."""
    ds = PetNoseDataset(split=split)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    for images, labels, orig_sizes in loader:
        assert images.shape[1:] == (3, IMG_SIZE, IMG_SIZE), \
            f"{split}: bad image shape {tuple(images.shape)}"
        assert labels.shape[1] == 2, f"{split}: bad label shape {tuple(labels.shape)}"
        assert torch.isfinite(images).all(), f"{split}: non-finite image values"
        assert torch.isfinite(labels).all(), f"{split}: non-finite labels"
        assert (labels >= 0).all() and (labels <= IMG_SIZE).all(), \
            f"{split}: label outside the {IMG_SIZE}x{IMG_SIZE} frame"
        assert (orig_sizes > 0).all(), f"{split}: non-positive original size"
    print(f"  {split:5s}: {len(ds):,} samples OK "
          f"(images (3, {IMG_SIZE}, {IMG_SIZE}), labels in frame)")
    return len(ds)


def plot_samples(dataset: PetNoseDataset, out_path: Path, n: int = 8) -> None:
    """Plot n training images with their label drawn as a dot."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 9))
    for i, ax in enumerate(axes.flat):
        image, label, _ = dataset[i]
        ax.imshow(denormalize(image).permute(1, 2, 0).numpy())
        u, v = label.tolist()
        ax.plot(u, v, "o", markersize=8, markerfacecolor="none",
                markeredgecolor="red", markeredgewidth=2)
        ax.set_title(dataset.samples[i][0], fontsize=8)
        ax.axis("off")
    fig.suptitle("PetNoseDataset reality check: labels must sit on the nose")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  saved {out_path}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    print("Validating PetNoseDataset:")
    total = sum(validate(split) for split in ["train", "test"])

    print("Plotting eight training samples:")
    plot_samples(PetNoseDataset(split="train"), root / "plots" / "dataset_check.png")

    print(f"All {total:,} samples validated.")
