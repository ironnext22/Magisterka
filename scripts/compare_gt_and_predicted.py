from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from projekt.data.PytorchDataset import TerrainDataset
from projekt.models.unet import UNetTerrainModel
from projekt.visualization.panda_viewer_2 import PandaTerrainComparisonViewer


def save_heightmap_png(height_tensor: torch.Tensor, output_path: Path) -> None:
    if height_tensor.ndim == 3:
        height_tensor = height_tensor[0]

    arr = height_tensor.detach().cpu().numpy()
    arr = arr - arr.min()
    arr = arr / (arr.max() + 1e-8)
    arr = (arr * 255.0).astype(np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="L").save(output_path)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    processed_root = project_root / "data" / "processed"
    index_path = processed_root / "index.csv"

    model_path = project_root / "outputs" / "unet_segmentation" / "unet_segmentation.pt"
    predictions_dir = project_root / "outputs" / "predictions"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TerrainDataset(
        index_path=index_path,
        data_root=processed_root,
        use_texture=True,
        use_segmentation=True,
        normalize_texture=True,
        normalize_heightmap=True,
        segmentation_mode="normalized",
    )

    df = pd.read_csv(index_path)

    sample_idx = 0
    x, _ = dataset[sample_idx]
    x = x.unsqueeze(0).to(device)

    row = df.iloc[sample_idx]

    gt_heightmap_path = processed_root / str(row["height_rel"])
    texture_path = processed_root / str(row["tex_rel"])

    pred_heightmap_path = predictions_dir / f"predicted_heightmap_{row['id']}.png"

    model = UNetTerrainModel(in_channels=4, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        y_pred = model(x)
        y_pred = torch.clamp(y_pred, 0.0, 1.0)[0]

    save_heightmap_png(y_pred, pred_heightmap_path)

    print("Ground truth:", gt_heightmap_path)
    print("Prediction:", pred_heightmap_path)
    print("Texture:", texture_path)

    app = PandaTerrainComparisonViewer(
        heightmap_path_1=gt_heightmap_path,
        heightmap_path_2=pred_heightmap_path,
        texture_path_1=texture_path,
        texture_path_2=texture_path,
        z_scale=40.0,
        terrain_spacing=320.0,
    )
    app.run()


if __name__ == "__main__":
    main()