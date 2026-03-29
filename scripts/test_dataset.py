from __future__ import annotations

from pathlib import Path

from projekt.data.dataloader import create_dataloader
from projekt.data.PytorchDataset import TerrainDataset


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    processed_root = project_root / "data" / "processed"
    index_path = processed_root / "index.csv"

    dataset = TerrainDataset(
        index_path=index_path,
        data_root=processed_root,
        use_texture=True,
        use_segmentation=True,
        normalize_texture=True,
        normalize_heightmap=True,
        segmentation_mode="normalized",
    )

    print(f"Dataset size: {len(dataset)}")

    x, y = dataset[0]
    print("Single sample:")
    print("X shape:", x.shape)
    print("Y shape:", y.shape)
    print("X dtype:", x.dtype)
    print("Y dtype:", y.dtype)
    print("X min/max:", float(x.min()), float(x.max()))
    print("Y min/max:", float(y.min()), float(y.max()))

    dataloader = create_dataloader(
        index_path=index_path,
        data_root=processed_root,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    batch_x, batch_y = next(iter(dataloader))

    print("\nBatch:")
    print("batch_x shape:", batch_x.shape)
    print("batch_y shape:", batch_y.shape)
    print("batch_x dtype:", batch_x.dtype)
    print("batch_y dtype:", batch_y.dtype)


if __name__ == "__main__":
    main()