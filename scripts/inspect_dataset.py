from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


@dataclass
class SampleStats:
    sample_id: str
    height_exists: bool
    seg_exists: bool
    tex_exists: bool
    height_shape: tuple[int, ...] | None
    seg_shape: tuple[int, ...] | None
    tex_shape: tuple[int, ...] | None
    height_dtype: str | None
    seg_dtype: str | None
    tex_dtype: str | None
    height_min: float | None
    height_max: float | None
    seg_unique_count: int | None
    seg_unique_preview: list[int] | None


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img)


def safe_shape(arr: np.ndarray | None) -> tuple[int, ...] | None:
    return tuple(arr.shape) if arr is not None else None


def safe_dtype(arr: np.ndarray | None) -> str | None:
    return str(arr.dtype) if arr is not None else None


def compute_sample_stats(row: pd.Series, data_root: Path) -> SampleStats:
    sample_id = str(row["id"])

    height_path = data_root / str(row["height_rel"])
    seg_path = data_root / str(row["seg_rel"])
    tex_path = data_root / str(row["tex_rel"]) if "tex_rel" in row and pd.notna(row["tex_rel"]) else None

    height_exists = height_path.exists()
    seg_exists = seg_path.exists()
    tex_exists = tex_path.exists() if tex_path is not None else False

    height_arr = load_image(height_path) if height_exists else None
    seg_arr = load_image(seg_path) if seg_exists else None
    tex_arr = load_image(tex_path) if tex_exists and tex_path is not None else None

    height_min = float(np.min(height_arr)) if height_arr is not None else None
    height_max = float(np.max(height_arr)) if height_arr is not None else None

    seg_unique_preview: list[int] | None = None
    seg_unique_count: int | None = None

    if seg_arr is not None:
        unique_vals = np.unique(seg_arr)
        seg_unique_count = int(len(unique_vals))
        seg_unique_preview = [int(v) for v in unique_vals[:20]]

    return SampleStats(
        sample_id=sample_id,
        height_exists=height_exists,
        seg_exists=seg_exists,
        tex_exists=tex_exists,
        height_shape=safe_shape(height_arr),
        seg_shape=safe_shape(seg_arr),
        tex_shape=safe_shape(tex_arr),
        height_dtype=safe_dtype(height_arr),
        seg_dtype=safe_dtype(seg_arr),
        tex_dtype=safe_dtype(tex_arr),
        height_min=height_min,
        height_max=height_max,
        seg_unique_count=seg_unique_count,
        seg_unique_preview=seg_unique_preview,
    )


def save_preview(
    row: pd.Series,
    data_root: Path,
    output_dir: Path,
) -> None:
    sample_id = str(row["id"])

    height_path = data_root / str(row["height_rel"])
    seg_path = data_root / str(row["seg_rel"])
    tex_path = data_root / str(row["tex_rel"]) if "tex_rel" in row and pd.notna(row["tex_rel"]) else None

    images: list[tuple[str, np.ndarray]] = []

    if tex_path is not None and tex_path.exists():
        images.append(("Texture", load_image(tex_path)))

    if seg_path.exists():
        images.append(("Segmentation", load_image(seg_path)))

    if height_path.exists():
        images.append(("Heightmap", load_image(height_path)))

    if not images:
        return

    fig, axes = plt.subplots(1, len(images), figsize=(5 * len(images), 5))
    if len(images) == 1:
        axes = [axes]

    for ax, (title, img) in zip(axes, images):
        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img)
        ax.set_title(f"{sample_id} - {title}")
        ax.axis("off")

    plt.tight_layout()
    output_path = output_dir / f"{sample_id}_preview.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_basic_summary(df_stats: pd.DataFrame) -> None:
    print("\n=== DATASET SUMMARY ===")
    print(f"Number of samples: {len(df_stats)}")

    print("\n=== FILE EXISTENCE ===")
    print(f"Height files found: {df_stats['height_exists'].sum()} / {len(df_stats)}")
    print(f"Segmentation files found: {df_stats['seg_exists'].sum()} / {len(df_stats)}")
    print(f"Texture files found: {df_stats['tex_exists'].sum()} / {len(df_stats)}")

    missing_any = df_stats[
        (~df_stats["height_exists"]) |
        (~df_stats["seg_exists"])
    ]
    print(f"Samples with missing required files: {len(missing_any)}")

    print("\n=== HEIGHTMAP VALUE RANGE ===")
    valid_height = df_stats.dropna(subset=["height_min", "height_max"])
    if not valid_height.empty:
        print(f"Global min: {valid_height['height_min'].min()}")
        print(f"Global max: {valid_height['height_max'].max()}")
    else:
        print("No valid heightmaps found.")

    print("\n=== SHAPES ===")
    print("Most common height shapes:")
    print(df_stats["height_shape"].value_counts().head(10).to_string())

    print("\nMost common segmentation shapes:")
    print(df_stats["seg_shape"].value_counts().head(10).to_string())

    print("\nMost common texture shapes:")
    print(df_stats["tex_shape"].value_counts().head(10).to_string())

    print("\n=== SEGMENTATION CLASSES ===")
    valid_seg = df_stats.dropna(subset=["seg_unique_count"])
    if not valid_seg.empty:
        print(f"Min number of unique classes: {int(valid_seg['seg_unique_count'].min())}")
        print(f"Max number of unique classes: {int(valid_seg['seg_unique_count'].max())}")
    else:
        print("No valid segmentation maps found.")


def save_reports(df_stats: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_csv = output_dir / "dataset_stats.csv"
    df_stats.to_csv(stats_csv, index=False)

    missing_csv = output_dir / "missing_files.csv"
    df_missing = df_stats[
        (~df_stats["height_exists"]) |
        (~df_stats["seg_exists"])
    ]
    df_missing.to_csv(missing_csv, index=False)

    inconsistent_shapes_csv = output_dir / "inconsistent_shapes.csv"
    df_inconsistent = df_stats[
        (df_stats["height_shape"] != df_stats["seg_shape"]) &
        df_stats["height_shape"].notna() &
        df_stats["seg_shape"].notna()
    ]
    df_inconsistent.to_csv(inconsistent_shapes_csv, index=False)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "data" / "raw"
    index_path = project_root / "data" / "interim" / "index.csv"
    output_dir = project_root / "outputs" / "inspection"

    if not index_path.exists():
        print(f"Missing file: {index_path}")
        sys.exit(1)

    df = pd.read_csv(index_path)

    required_columns = {"id", "height_rel", "seg_rel"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        print(f"Missing required columns in index.csv: {sorted(missing_columns)}")
        sys.exit(1)

    if df.empty:
        print("index.csv is empty.")
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / "empty_index_copy.csv", index=False)
        sys.exit(0)

    stats: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        sample_stats = compute_sample_stats(row, data_root)
        stats.append(sample_stats.__dict__)

    df_stats = pd.DataFrame(stats)

    output_dir.mkdir(parents=True, exist_ok=True)

    print_basic_summary(df_stats)
    save_reports(df_stats, output_dir)

    preview_count = min(10, len(df))
    for i in range(preview_count):
        save_preview(df.iloc[i], data_root, output_dir)

    print(f"\nReports saved to: {output_dir}")


if __name__ == "__main__":
    main()