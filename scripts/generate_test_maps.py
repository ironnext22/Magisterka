from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from projekt.generation.procedural_maps import (
    ProceduralMapConfig,
    generate_maps,
    save_segmentation_png,
    save_texture_png,
)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "outputs" / "generated_inputs"

    seed = 100

    config = ProceduralMapConfig(
        width=256,
        height=256,

        sea_level=0.42,
        beach_width=3,

        grass_level=0.52,
        rock_level=0.68,
        snow_level=0.82,

        continent_strength=0.65,
        mountain_strength=0.35,
    )

    height_seed_map, segmentation_map, texture_map = generate_maps(seed, config)

    seg_path = output_dir / f"segmentation_seed_{seed}.png"
    tex_path = output_dir / f"texture_seed_{seed}.png"

    save_segmentation_png(segmentation_map, seg_path)
    save_texture_png(texture_map, tex_path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(height_seed_map, cmap="terrain")
    axes[0].set_title("Height seed map")
    axes[0].axis("off")

    axes[1].imshow(segmentation_map, cmap="gray")
    axes[1].set_title("Segmentation classes")
    axes[1].axis("off")

    axes[2].imshow(texture_map)
    axes[2].set_title("Generated texture")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

    print("Saved segmentation to:", seg_path)
    print("Saved texture to:", tex_path)


if __name__ == "__main__":
    main()