from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


def load_image(path: Path):
    img = Image.open(path)
    return np.array(img), img.mode


def inspect_sample(row, root):
    sample_id = row["id"]

    height_path = root / row["height_rel"]
    seg_path = root / row["seg_rel"]
    tex_path = root / row["tex_rel"] if "tex_rel" in row else None

    height, height_mode = load_image(height_path)
    seg, seg_mode = load_image(seg_path)

    tex = None
    tex_mode = None
    if tex_path and tex_path.exists():
        tex, tex_mode = load_image(tex_path)

    print(f"\n--- SAMPLE {sample_id} ---")

    print("HEIGHTMAP")
    print("shape:", height.shape)
    print("dtype:", height.dtype)
    print("mode:", height_mode)
    print("min:", height.min())
    print("max:", height.max())

    print("\nSEGMENTATION")
    print("shape:", seg.shape)
    print("dtype:", seg.dtype)
    print("mode:", seg_mode)
    print("unique classes:", np.unique(seg)[:20])

    if tex is not None:
        print("\nTEXTURE")
        print("shape:", tex.shape)
        print("dtype:", tex.dtype)
        print("mode:", tex_mode)


def visualize_sample(row, root, out_dir):
    sample_id = row["id"]

    height = np.array(Image.open(root / row["height_rel"]))
    seg = np.array(Image.open(root / row["seg_rel"]))

    tex = None
    if "tex_rel" in row:
        tex_path = root / row["tex_rel"]
        if tex_path.exists():
            tex = np.array(Image.open(tex_path))

    cols = 2 if tex is None else 3
    fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 5))

    if cols == 2:
        axes = [axes[0], axes[1]]

    if tex is not None:
        axes[0].imshow(tex)
        axes[0].set_title("Texture")
        axes[0].axis("off")

        axes[1].imshow(seg)
        axes[1].set_title("Segmentation")
        axes[1].axis("off")

        axes[2].imshow(height, cmap="terrain")
        axes[2].set_title("Heightmap")
        axes[2].axis("off")

    else:
        axes[0].imshow(seg)
        axes[0].set_title("Segmentation")
        axes[0].axis("off")

        axes[1].imshow(height, cmap="terrain")
        axes[1].set_title("Heightmap")
        axes[1].axis("off")

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / f"{sample_id}.png")
    plt.close()


def main():

    project_root = Path(__file__).resolve().parents[1]

    processed_root = project_root / "data" / "processed"
    index_path = processed_root / "index.csv"

    df = pd.read_csv(index_path)

    print("\n=== DATASET SIZE ===")
    print(len(df))

    shapes = []

    for i, row in df.iterrows():
        height = np.array(Image.open(processed_root / row["height_rel"]))
        shapes.append(height.shape)

    print("\n=== HEIGHTMAP SHAPES ===")
    print(pd.Series(shapes).value_counts())

    print("\n=== SAMPLE INSPECTION ===")

    sample_count = min(5, len(df))

    for i in range(sample_count):
        inspect_sample(df.iloc[i], processed_root)

    print("\n=== SAVING VISUAL PREVIEWS ===")

    out_dir = project_root / "outputs" / "processed_inspection"

    for i in range(sample_count):
        visualize_sample(df.iloc[i], processed_root, out_dir)

    print("\nSaved previews to:", out_dir)


if __name__ == "__main__":
    main()