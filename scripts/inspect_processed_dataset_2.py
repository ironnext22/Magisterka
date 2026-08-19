from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

SEGMENTATION_CLASSES = {
    (17, 141, 215): "Water",
    (225, 227, 155): "Grassland",
    (127, 173, 123): "Forest",
    (185, 122, 87): "Hills",
    (230, 200, 181): "Desert",
    (150, 150, 150): "Mountain",
    (193, 190, 175): "Tundra",
}

# Kolejność klas używana w tabelach i na wykresach.
CLASS_ORDER = [
    "Water",
    "Grassland",
    "Forest",
    "Hills",
    "Desert",
    "Mountain",
    "Tundra",
]

INDEX_CLASSES = {
    112: "Water",
    218: "Grassland",
    154: "Forest",
    137: "Hills",
    207: "Desert",
    150: "Mountain",
    189: "Tundra",
}

CLASS_COLORS = {
    "Water": (17, 141, 215),
    "Grassland": (225, 227, 155),
    "Forest": (127, 173, 123),
    "Hills": (185, 122, 87),
    "Desert": (230, 200, 181),
    "Mountain": (150, 150, 150),
    "Tundra": (193, 190, 175),
}


def load_image(path: Path) -> tuple[np.ndarray, str]:
    with Image.open(path) as img:
        return np.array(img), img.mode


def load_array(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img)


def relative_path(root: Path, value: str) -> Path:
    return root / str(value)


def inspect_sample(row: pd.Series, root: Path) -> None:
    sample_id = row["id"]

    height_path = relative_path(root, row["height_rel"])
    seg_path = relative_path(root, row["seg_rel"])
    tex_path = relative_path(root, row["tex_rel"]) if "tex_rel" in row and pd.notna(row["tex_rel"]) else None

    height, height_mode = load_image(height_path)
    seg, seg_mode = load_image(seg_path)

    tex = None
    tex_mode = None
    if tex_path is not None and tex_path.exists():
        tex, tex_mode = load_image(tex_path)

    print(f"\n--- SAMPLE {sample_id} ---")

    print("HEIGHTMAP")
    print("shape:", height.shape)
    print("dtype:", height.dtype)
    print("mode:", height_mode)
    print("min:", int(height.min()))
    print("max:", int(height.max()))
    print("mean:", float(height.mean()))

    print("\nSEGMENTATION")
    print("shape:", seg.shape)
    print("dtype:", seg.dtype)
    print("mode:", seg_mode)

    if seg.ndim == 3:
        unique_colors = np.unique(seg.reshape(-1, seg.shape[-1]), axis=0)
        print("unique colors count:", len(unique_colors))
        print("unique colors preview:", unique_colors[:10].tolist())
    else:
        unique_vals = np.unique(seg)
        print("unique classes count:", len(unique_vals))
        print("unique classes preview:", unique_vals[:20].tolist())

    if tex is not None:
        print("\nTEXTURE")
        print("shape:", tex.shape)
        print("dtype:", tex.dtype)
        print("mode:", tex_mode)


def count_segmentation_classes(seg: np.ndarray) -> Counter:
    counts = Counter()

    if seg.ndim == 3:
        flat = seg.reshape(-1, seg.shape[-1])
        colors, color_counts = np.unique(flat, axis=0, return_counts=True)
        for color, count in zip(colors, color_counts):
            key = tuple(int(v) for v in color[:3])
            class_name = SEGMENTATION_CLASSES.get(key, f"Unknown RGB {key}")
            counts[class_name] += int(count)
    else:
        values, value_counts = np.unique(seg, return_counts=True)
        for value, count in zip(values, value_counts):
            class_name = INDEX_CLASSES.get(int(value), f"Unknown index {int(value)}")
            counts[class_name] += int(count)

    return counts


def colorize_segmentation(seg: np.ndarray) -> np.ndarray:
    if seg.ndim == 3:
        return seg[..., :3]

    rgb = np.zeros((*seg.shape, 3), dtype=np.uint8)

    for index_value, class_name in INDEX_CLASSES.items():
        color = CLASS_COLORS[class_name]
        rgb[seg == index_value] = color

    return rgb


def visualize_sample(row: pd.Series, root: Path, out_dir: Path, filename_suffix: str = "") -> None:
    sample_id = str(row["id"])

    height = load_array(relative_path(root, row["height_rel"]))
    seg = load_array(relative_path(root, row["seg_rel"]))

    tex = None
    if "tex_rel" in row and pd.notna(row["tex_rel"]):
        tex_path = relative_path(root, row["tex_rel"])
        if tex_path.exists():
            tex = load_array(tex_path)

    images: list[tuple[str, np.ndarray, str | None]] = []
    if tex is not None:
        images.append(("Tekstura RGB", tex, None))
    images.append(("Mapa segmentacji", colorize_segmentation(seg), None))
    images.append(("Mapa wysokości", height, "terrain"))

    fig, axes = plt.subplots(1, len(images), figsize=(4.2 * len(images), 3.6))
    if len(images) == 1:
        axes = [axes]

    for ax, (title, img, cmap) in zip(axes, images):
        if img.ndim == 2:
            ax.imshow(img, cmap=cmap or "gray")
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    plt.tight_layout(pad=0.8)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / f"{sample_id}{filename_suffix}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_combined_preview(df: pd.DataFrame, root: Path, out_dir: Path, sample_indices: list[int]) -> None:
    selected = [i for i in sample_indices if 0 <= i < len(df)]
    if not selected:
        return

    rows = len(selected)
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.5 * rows))
    if rows == 1:
        axes = np.array([axes])

    column_titles = ["Tekstura", "Mapa segmentacji", "Mapa wysokości"]

    for r, idx in enumerate(selected):
        row = df.iloc[idx]
        height = load_array(relative_path(root, row["height_rel"]))
        seg = load_array(relative_path(root, row["seg_rel"]))
        tex = load_array(relative_path(root, row["tex_rel"])) if "tex_rel" in row and pd.notna(row["tex_rel"]) else None

        imgs = [tex, colorize_segmentation(seg), height]
        cmaps = [None, None, "terrain"]

        for c in range(cols):
            ax = axes[r, c]
            img = imgs[c]
            if img is None:
                ax.axis("off")
                continue
            if img.ndim == 2:
                ax.imshow(img, cmap=cmaps[c] or "gray")
            else:
                ax.imshow(img)
            if r == 0:
                ax.set_title(column_titles[c], fontsize=12)
            ax.axis("off")

    plt.tight_layout(pad=0.8)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "filtered_samples.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_height_histogram(values: np.ndarray, out_dir: Path) -> None:
    plt.figure(figsize=(7, 4))
    plt.hist(values, bins=80)
    plt.title("Histogram wartości pikseli map wysokości")
    plt.xlabel("Wartość piksela")
    plt.ylabel("Liczba pikseli")
    plt.tight_layout()
    plt.savefig(out_dir / "height_histogram.png", dpi=180, bbox_inches="tight")
    plt.close()


def save_class_distribution(class_counts: Counter, out_dir: Path) -> pd.DataFrame:
    total = sum(class_counts.values())
    rows = []

    ordered_names = CLASS_ORDER + sorted(
        name for name in class_counts.keys() if name not in CLASS_ORDER
    )

    for class_name in ordered_names:
        count = class_counts.get(class_name, 0)
        rows.append({
            "class": class_name,
            "pixels": count,
            "percentage": 100.0 * count / total if total else 0.0,
        })

    df_classes = pd.DataFrame(rows)
    df_classes.to_csv(out_dir / "segmentation_class_distribution.csv", index=False)

    plt.figure(figsize=(8, 4))
    plt.bar(df_classes["class"], df_classes["percentage"])
    plt.title("Udział klas segmentacji w zbiorze")
    plt.xlabel("Klasa")
    plt.ylabel("Udział pikseli [%]")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "segmentation_class_distribution.png", dpi=180, bbox_inches="tight")
    plt.close()

    return df_classes


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    processed_root = project_root / "data" / "processed"
    index_path = processed_root / "index.csv"
    out_dir = project_root / "outputs" / "processed_inspection"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(index_path)

    print("\n=== DATASET SIZE ===")
    print(len(df))

    shapes = []
    missing_rows = []
    height_min_values = []
    height_max_values = []
    height_mean_values = []
    height_samples_for_hist = []
    class_counts = Counter()

    sample_stride = max(1, len(df) // 500)

    for i, row in df.iterrows():
        height_path = relative_path(processed_root, row["height_rel"])
        seg_path = relative_path(processed_root, row["seg_rel"])
        tex_path = relative_path(processed_root, row["tex_rel"]) if "tex_rel" in row and pd.notna(row["tex_rel"]) else None

        missing = []
        if not height_path.exists():
            missing.append("heightmap")
        if not seg_path.exists():
            missing.append("segmentation")
        if tex_path is not None and not tex_path.exists():
            missing.append("texture")

        if missing:
            missing_rows.append({"id": row["id"], "missing": ", ".join(missing)})
            continue

        height = load_array(height_path)
        seg = load_array(seg_path)

        shapes.append(height.shape)
        height_min_values.append(float(height.min()))
        height_max_values.append(float(height.max()))
        height_mean_values.append(float(height.mean()))

        if i % sample_stride == 0:
            height_samples_for_hist.append(height.reshape(-1))

        class_counts.update(count_segmentation_classes(seg))

    print("\n=== HEIGHTMAP SHAPES ===")
    print(pd.Series(shapes).value_counts())

    print("\n=== DATA COMPLETENESS ===")
    print("Samples in index.csv:", len(df))
    print("Missing or incomplete samples:", len(missing_rows))
    print("Complete samples:", len(df) - len(missing_rows))

    completeness = pd.DataFrame([
        {"stage": "Próbki w index.csv", "samples": len(df)},
        {"stage": "Kompletne zestawy danych", "samples": len(df) - len(missing_rows)},
        {"stage": "Odrzucone / niekompletne próbki", "samples": len(missing_rows)},
    ])
    completeness.to_csv(out_dir / "data_completeness.csv", index=False)

    if missing_rows:
        pd.DataFrame(missing_rows).to_csv(out_dir / "missing_processed_files.csv", index=False)

    print("\n=== HEIGHTMAP STATISTICS ===")
    height_stats = pd.DataFrame([
        {
            "global_min": min(height_min_values) if height_min_values else None,
            "global_max": max(height_max_values) if height_max_values else None,
            "mean_of_sample_means": float(np.mean(height_mean_values)) if height_mean_values else None,
            "median_of_sample_means": float(np.median(height_mean_values)) if height_mean_values else None,
        }
    ])
    print(height_stats.to_string(index=False))
    height_stats.to_csv(out_dir / "height_statistics.csv", index=False)

    if height_samples_for_hist:
        height_values = np.concatenate(height_samples_for_hist)
        save_height_histogram(height_values, out_dir)

    print("\n=== SEGMENTATION CLASS DISTRIBUTION ===")
    df_classes = save_class_distribution(class_counts, out_dir)
    print(df_classes.to_string(index=False))

    print("\n=== SAMPLE INSPECTION ===")
    sample_count = min(5, len(df))
    for i in range(sample_count):
        inspect_sample(df.iloc[i], processed_root)

    print("\n=== SAVING VISUAL PREVIEWS ===")
    for i in range(sample_count):
        visualize_sample(df.iloc[i], processed_root, out_dir, filename_suffix="_preview")

    if len(df) >= 3:
        preview_indices = [
            0,
            len(df) // 2,
            len(df) - 1,
        ]
    else:
        preview_indices = list(range(len(df)))

    save_combined_preview(df, processed_root, out_dir, preview_indices)

    print("\nSaved reports and figures to:", out_dir)
    print("Generated files:")
    for path in sorted(out_dir.glob("*")):
        print(" -", path.name)


if __name__ == "__main__":
    main()
