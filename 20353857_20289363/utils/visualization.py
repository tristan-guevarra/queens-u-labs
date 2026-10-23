"""
ELEC 475: Lab 2 - Plotting & Visualization Utilities
"""
import os
import matplotlib.pyplot as plt
from typing import List, Dict

def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: str = "./plots/training_curves.png",
    model_name: str = "Model"
):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-o', label='Train Loss')
    plt.plot(epochs, val_losses, 'r--s', label='Val Loss')
    plt.title(f"{model_name.upper()} - Cross-Entropy Loss", fontsize=12)
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Loss", fontsize=10)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-o', label='Train Accuracy')
    plt.plot(epochs, val_accs, 'g--^', label='Val Accuracy')
    plt.title(f"{model_name.upper()} - Top-1 Accuracy (%)", fontsize=12)
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Accuracy (%)", fontsize=10)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Training curves successfully saved to: {save_path}")

def plot_ablation_summary(
    ablation_results: Dict[str, Dict[str, List[float]]],
    save_path: str = "./plots/ablation_summary.png",
    title: str = "ELEC 475: Master Ablation Study (Validation Accuracy)"
):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.figure(figsize=(10, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    styles = ['-', '--', '-.', ':', '-', '--', '-.']

    for idx, (model_label, data) in enumerate(ablation_results.items()):
        val_accs = data.get("val_accs", [])
        epochs = range(1, len(val_accs) + 1)
        color = colors[idx % len(colors)]
        style = styles[idx % len(styles)]
        plt.plot(epochs, val_accs, linestyle=style, marker='o', color=color, label=model_label, linewidth=2)

    plt.title(title, fontsize=13)
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Validation Accuracy (%)", fontsize=11)
    plt.legend(loc='lower right', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Master ablation summary plot saved to: {save_path}")



def collect_ablation_results(output_dir, dataset, seed=42):
    """Read complete handout configurations without requiring checkpoint files."""
    import math
    import re
    from pathlib import Path

    configs = [
        ("Base LeNet-5", "lenet", None, False, 5),
        ("Modernized LeNet", "lenet_modern", None, False, 5),
        ("AlexNet (No Norm)", "alexnet", "none", False, 10),
        ("AlexNet + LRN", "alexnet", "lrn", False, 10),
        ("AlexNet + BatchNorm", "alexnet", "bn", False, 10),
        ("AlexNet + BN + Aug", "alexnet", "bn", True, 15 if dataset == "cifar10" else 10),
        ("PyTorch Zoo AlexNet", "zoo_alexnet", None, False, 15),
    ]
    selected = {}
    paths = sorted(Path(output_dir).rglob("*.txt"), key=lambda p: (p.stat().st_mtime_ns, str(p)))
    for path in paths:
        raw = path.read_bytes()
        encoding = "utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
        text = raw.decode(encoding, errors="replace")
        model = re.search(r"^Model\s*:\s*(\w+) \(Norm: ([\w/]+)\)", text, re.M)
        data = re.search(r"^Dataset\s*:\s*(\w+) \(Augmentation: (True|False)\)", text, re.M)
        hp = re.search(r"Hyperparams\s*: Epochs=(\d+), BatchSize=(\d+), LR=([\deE.+-]+), Seed=(\d+)", text)
        if not (model and data and hp and "Training completed in" in text):
            continue
        epochs, batch, lr, run_seed = hp.groups()
        if (data[1].lower() != dataset or int(run_seed) != seed
                or int(batch) != 128 or float(lr) != 0.001):
            continue
        rows = []
        try:
            for line in text.splitlines():
                if re.match(r"^\s*\d+\s*\|", line):
                    fields = line.split("|")
                    if len(fields) == 8:
                        rows.append([float(v.strip()) for v in fields[:7]])
        except ValueError:
            continue
        if ([r[0] for r in rows] != list(range(1, int(epochs) + 1))
                or not all(math.isfinite(v) for row in rows for v in row)):
            continue
        for label, name, norm, augment, count in configs:
            if (model[1].lower() == name and (norm is None or model[2].lower() == norm)
                    and (data[2] == "True") == augment and int(epochs) == count):
                # A later completed retry wins; incomplete logs never replace it.
                selected[label] = {"val_accs": [r[4] for r in rows], "source": str(path.resolve())}
    return ({label: selected[label] for label, *_ in configs if label in selected},
            [label for label, *_ in configs if label not in selected])


def main():
    import argparse
    import json
    import shutil
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Plot seven-model ablation curves from completed training logs")
    parser.add_argument("--plot-ablation", action="store_true")
    parser.add_argument("--dataset", choices=["both", "cifar10", "tiny_imagenet"], default="both")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outputs-dir", type=Path, default=root / "outputs")
    parser.add_argument("--plots-dir", type=Path, default=root / "plots")
    args = parser.parse_args()
    if not args.plot_ablation:
        parser.error("specify --plot-ablation")
    # This CLI saves files and does not need a desktop GUI or Tk installation.
    plt.switch_backend("Agg")
    datasets = ["cifar10", "tiny_imagenet"] if args.dataset == "both" else [args.dataset]
    manifest = {}
    generated = 0
    args.plots_dir.mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        results, missing = collect_ablation_results(args.outputs_dir, dataset, args.seed)
        manifest[dataset] = {"seed": args.seed, "sources": {k: v["source"] for k, v in results.items()}, "missing": missing}
        print(f"{dataset}: {len(results)}/7 completed configurations (seed {args.seed})")
        for label, values in results.items():
            print(f"  {label}: {Path(values['source']).name}")
        if missing:
            print("  MISSING: " + ", ".join(missing))
        if not results:
            continue
        title = f"{dataset.upper()} - Validation accuracy (seed {args.seed}; {len(results)}/7 configurations)"
        plot_ablation_summary(results, str(args.plots_dir / f"ablation_{dataset}.png"), title)
        if dataset == "cifar10":
            shutil.copyfile(args.plots_dir / "ablation_cifar10.png", args.plots_dir / "ablation_summary.png")
            print(f"Submission plot saved to: {args.plots_dir / 'ablation_summary.png'}")
        generated += 1
    (args.plots_dir / "ablation_sources.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if not generated:
        parser.exit(1, "No completed matching training logs found; no plots generated.\n")


if __name__ == "__main__":
    main()
