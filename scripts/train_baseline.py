from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from projekt.data.PytorchDataset import TerrainDataset
from projekt.models.baseline_cnn import BaselineTerrainCNN
from projekt.models.unet import UNetTerrainModel
from projekt.training.train_utils import (
    save_prediction_preview,
    train_one_epoch,
    validate_one_epoch,
)

import time


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    processed_root = project_root / "data" / "processed"
    index_path = processed_root / "index.csv"

    #outputs_root = project_root / "outputs" / "baseline"
    outputs_root = project_root / "outputs" / "unet_segmentation"
    outputs_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    full_dataset = TerrainDataset(
        index_path=index_path,
        data_root=processed_root,
        use_texture=True,
        use_segmentation=True,
        normalize_texture=True,
        normalize_heightmap=True,
        segmentation_mode="normalized",
    )

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    #model = BaselineTerrainCNN(in_channels=4, out_channels=1).to(device)
    model = UNetTerrainModel(in_channels=4, out_channels=1).to(device)
    #criterion = nn.MSELoss()
    criterion = nn.SmoothL1Loss(beta=0.02)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    gradient_weight = 0.1
    num_epochs = 50
    history: list[dict[str, float]] = []

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        train_loss, train_mse, train_grad = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            gradient_weight=gradient_weight,
        )

        val_loss, val_mse, val_grad = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            gradient_weight=gradient_weight,
        )

        epoch_time = time.time() - epoch_start
        remaining = epoch_time * (num_epochs - epoch - 1)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_mse":train_mse,
                "train_grad":train_grad,
                "val_loss": val_loss,
                "val_mse":val_mse,
                "val_grad":val_grad,
            }
        )

        print(
            f"Epoch {epoch:02d}/{num_epochs} | "
            f"train_loss={train_loss:.6f} | train_mse={train_mse:.6f} | train_grad={train_grad:.6f} | "
            f"val_loss={val_loss:.6f} | val_mse={val_mse:.6f} | val_grad={val_grad:.6f} | "
            f"time={epoch_time:.2f}s | "
            f"ETA={remaining / 60:.1f}min"
        )

        save_prediction_preview(
            model=model,
            dataloader=val_loader,
            device=device,
            output_path=outputs_root / f"preview_epoch_{epoch:02d}.png",
        )

    #(model.state_dict(), outputs_root / "baseline_cnn.pt")
    torch.save(model.state_dict(), outputs_root / "unet_segmentation.pt")

    history_path = outputs_root / "training_history.csv"
    with open(history_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,train_mse,train_grad,val_loss,val_mse,val_grad\n")
        for row in history:
            f.write(f"{row['epoch']},{row['train_loss']},{row['train_mse']},{row['train_grad']},{row['val_loss']},{row['val_mse']},{row['val_grad']}\n")

    print("Training completed.")
    #print(f"Model saved to: {outputs_root / 'baseline_cnn.pt'}")
    print(f"Model saved to: {outputs_root / 'unet_segmentation.pt'}")
    print(f"History saved to: {history_path}")


if __name__ == "__main__":
    main()