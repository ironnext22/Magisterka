from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from projekt.generation.procedural_maps import (
    ProceduralMapConfig,
    generate_maps,
    save_segmentation_png,
    save_texture_png,
)
from projekt.models.unet import UNetTerrainModel
from projekt.visualization.panda_viewer_2 import PandaTerrainComparisonViewer


def load_generated_input_tensor(
    texture_path: Path,
    segmentation_path: Path,
    device: torch.device,
) -> torch.Tensor:
    with Image.open(texture_path) as img:
        texture = np.array(img.convert("RGB"), dtype=np.float32) / 255.0

    with Image.open(segmentation_path) as img:
        segmentation = np.array(img.convert("L"), dtype=np.float32) / 255.0

    texture = np.transpose(texture, (2, 0, 1))          # [3, H, W]
    segmentation = np.expand_dims(segmentation, axis=0) # [1, H, W]

    x = np.concatenate([texture, segmentation], axis=0) # [4, H, W]
    x = torch.from_numpy(x).unsqueeze(0).float().to(device)  # [1, 4, H, W]

    return x


def save_heightmap_png(height_tensor: torch.Tensor, output_path: Path) -> None:
    if height_tensor.ndim == 3:
        height_tensor = height_tensor[0]

    arr = height_tensor.detach().cpu().numpy()
    arr = np.clip(arr, 0.0, 1.0)
    arr = arr - arr.min()
    arr = arr / (arr.max() + 1e-8)

    arr = (arr * 255.0).astype(np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="L").save(output_path)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    outputs_root = project_root / "outputs"
    generated_dir = outputs_root / "generated_inputs"
    prediction_dir = outputs_root / "generated_predictions"

    model_path = project_root / "outputs" / "unet_segmentation" / "unet_segmentation.pt"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Nie znaleziono modelu: {model_path}\n"
            "Najpierw wytrenuj model i zapisz plik wag."
        )

    seed = 42

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

    print(f"Generating procedural maps for seed={seed}...")

    noise_map, segmentation_map, texture_map = generate_maps(seed=seed, config=config)

    texture_path = generated_dir / f"texture_seed_{seed}.png"
    segmentation_path = generated_dir / f"segmentation_seed_{seed}.png"
    predicted_heightmap_path = prediction_dir / f"predicted_heightmap_seed_{seed}.png"

    save_texture_png(texture_map, texture_path)
    save_segmentation_png(segmentation_map, segmentation_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = UNetTerrainModel(in_channels=4, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    x = load_generated_input_tensor(
        texture_path=texture_path,
        segmentation_path=segmentation_path,
        device=device,
    )

    with torch.no_grad():
        y_pred = model(x)
        y_pred = torch.clamp(y_pred, 0.0, 1.0)[0]  # [1, H, W]

    save_heightmap_png(y_pred, predicted_heightmap_path)

    print("Generated texture:", texture_path)
    print("Generated segmentation:", segmentation_path)
    print("Predicted heightmap:", predicted_heightmap_path)

    app = PandaTerrainComparisonViewer(
        heightmap_path_1=predicted_heightmap_path,
        heightmap_path_2=predicted_heightmap_path,
        texture_path_1=texture_path,
        texture_path_2=texture_path,
        z_scale=40.0,
        terrain_spacing=320.0,
    )
    app.run()


if __name__ == "__main__":
    main()