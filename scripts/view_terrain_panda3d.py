from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from projekt.data.PytorchDataset import TerrainDataset
from projekt.models.baseline_cnn import BaselineTerrainCNN
from projekt.visualization.panda_viewer import PandaTerrainViewer
from projekt.models.unet import UNetTerrainModel

def save_heightmap_png(height_tensor: torch.Tensor, output_path: Path) -> None:
    """
    Save predicted heightmap tensor [1, H, W] or [H, W] as grayscale PNG.
    Expected input range: [0, 1]
    """
    if height_tensor.ndim == 3:
        height_tensor = height_tensor[0]

    arr = height_tensor.detach().cpu().numpy()
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).astype(np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="L").save(output_path)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    processed_root = project_root / "data" / "processed"
    index_path = processed_root / "index.csv"

    model_path = project_root / "outputs" / "unet_segmentation" / "unet_segmentation.pt"
    #model_path = project_root / "outputs" / "baseline" / "baseline_cnn.pt"
    predicted_dir = project_root / "outputs" / "predictions"
    predicted_heightmap_path = predicted_dir / "predicted_heightmap.png"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Nie znaleziono modelu: {model_path}\n"
            "Najpierw uruchom trening i zapisz baseline_cnn.pt."
        )

    dataset = TerrainDataset(
        index_path=index_path,
        data_root=processed_root,
        use_texture=True,
        use_segmentation=True,
        normalize_texture=True,
        normalize_heightmap=True,
        segmentation_mode="normalized",
    )

    sample_idx = 0
    x, _ = dataset[sample_idx]
    x = x.unsqueeze(0).to(device)

    df = pd.read_csv(index_path)
    row = df.iloc[sample_idx]

    texture_path = None
    if "tex_rel" in df.columns and pd.notna(row["tex_rel"]):
        texture_path = processed_root / str(row["tex_rel"])

    model =  UNetTerrainModel(in_channels=4, out_channels=1).to(device)
    #model = BaselineTerrainCNN(in_channels=4, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        y_pred = model(x)[0]  # [1, H, W]

    save_heightmap_png(y_pred, predicted_heightmap_path)

    print("Sample ID:", row["id"])
    print("Texture:", texture_path)
    print("Predicted heightmap:", predicted_heightmap_path)

    app = PandaTerrainViewer(
        heightmap_path=predicted_heightmap_path,
        texture_path=texture_path,
        z_scale=40.0,
    )
    app.run()


if __name__ == "__main__":
    main()