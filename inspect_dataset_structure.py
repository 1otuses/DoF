#!/usr/bin/env python3
"""Inspect dataset directory structure and NumPy array metadata.

By default, this script checks the two dataset folders requested by the user:
- DoF_Trajectory/diffuser/datasets/data/mpe/simple_spread/expert
- DoF_Trajectory/diffuser/datasets/data/smac/3m/Good

It prints the file tree, array shapes, dtypes, and a few lightweight stats.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_DATASETS = [
    Path("DoF_Trajectory/diffuser/datasets/data/mpe/simple_spread/expert"),
    # Path("DoF_Trajectory/diffuser/datasets/data/smac/3m/Good"),
    # Path("DoF_Trajectory/diffuser/datasets/data/mpe/simple_spread"),
]


def format_scalar(value) -> str:
    if isinstance(value, (np.floating, float)):
        return f"{float(value):.6g}"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    return str(value)


def summarize_array(array: np.ndarray) -> list[str]:
    lines = [f"shape={array.shape}", f"dtype={array.dtype}"]
    if array.size == 0:
        lines.append("empty")
        return lines

    if np.issubdtype(array.dtype, np.number):
        lines.append(f"min={format_scalar(np.min(array))}")
        lines.append(f"max={format_scalar(np.max(array))}")
        lines.append(f"mean={format_scalar(np.mean(array))}")
        if array.ndim == 1:
            lines.append(f"unique={len(np.unique(array))}")
    elif array.dtype == object:
        lines.append(f"object_items={array.size}")
    return lines


def inspect_npy_file(file_path: Path) -> None:
    try:
        array = np.load(file_path, allow_pickle=False)
    except ValueError:
        array = np.load(file_path, allow_pickle=True)

    print(f"  - {file_path.name}")
    for line in summarize_array(array):
        print(f"      {line}")


def iter_entries(path: Path) -> Iterable[Path]:
    return sorted(path.iterdir(), key=lambda entry: (entry.is_file(), entry.name.lower()))


def inspect_dataset(dataset_path: Path) -> None:
    print(f"Dataset: {dataset_path}")
    if not dataset_path.exists():
        print("  [missing]")
        return

    files = [entry for entry in iter_entries(dataset_path) if entry.is_file()]
    subdirs = [entry for entry in iter_entries(dataset_path) if entry.is_dir()]

    if subdirs:
        print("  subdirectories:")
        for subdir in subdirs:
            print(f"    - {subdir.name}/")

    if not files:
        print("  [no files]")
        return

    print("  files:")
    for file_path in files:
        if file_path.suffix == ".npy":
            inspect_npy_file(file_path)
        else:
            print(f"  - {file_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect dataset directory structures.")
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Optional dataset paths to inspect. Defaults to the two requested datasets.",
    )
    args = parser.parse_args()

    dataset_paths = args.paths or DEFAULT_DATASETS
    for index, dataset_path in enumerate(dataset_paths):
        if index:
            print()
        inspect_dataset(dataset_path)


if __name__ == "__main__":
    main()