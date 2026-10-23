"""Lazy exports let utility modules run with python -m without pre-importing them."""
from importlib import import_module

_MODULE_EXPORTS = {
    "dataset": ["get_dataloaders"],
    "transforms": ["get_train_transforms", "get_test_transforms"],
    "metrics": ["AverageMeter", "compute_accuracy", "ConfusionMatrixTracker", "EarlyStopping"],
    "visualization": ["plot_training_curves", "plot_ablation_summary"],
    "eda": ["run_eda"],
    "profiler": ["Timer", "ModelProfiler", "benchmark_inference", "compare_model_complexities"],
    "seed": ["seed_everything", "seed_worker", "calculate_mean_std", "format_mean_std"],
}
_EXPORTS = {name: module for module, names in _MODULE_EXPORTS.items() for name in names}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value

__all__ = [
    "get_dataloaders",
    "get_train_transforms",
    "get_test_transforms",
    "AverageMeter",
    "compute_accuracy",
    "ConfusionMatrixTracker",
    "EarlyStopping",
    "plot_training_curves",
    "plot_ablation_summary",
    "run_eda",
    "Timer",
    "ModelProfiler",
    "benchmark_inference",
    "compare_model_complexities",
    "seed_everything",
    "seed_worker",
    "calculate_mean_std",
    "format_mean_std",
]
