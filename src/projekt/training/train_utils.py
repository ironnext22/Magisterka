from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch

def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]

    dx_target = target[:, :, :, 1:] - target[:, :, :, :-1]
    dy_target = target[:, :, 1:, :] - target[:, :, :-1, :]

    loss_x = torch.mean(torch.abs(dx_pred - dx_target))
    loss_y = torch.mean(torch.abs(dy_pred - dy_target))

    return loss_x + loss_y

def tv_loss(pred: torch.Tensor) -> torch.Tensor:
    dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    return torch.mean(torch.abs(dx)) + torch.mean(torch.abs(dy))

def train_one_epoch(
    model: torch.nn.Module,
    dataloader,
    criterion,
    optimizer,
    device: torch.device,
    gradient_weight: float = 0.01,
) -> tuple[float, float, float]:
    model.train()
    running_loss = 0.0
    running_mse = 0.0
    running_grad = 0.0

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        y_pred = model(x)
        y_pred = torch.clamp(y_pred, 0.0, 1.0)

        mse = criterion(y_pred, y)
        grad = gradient_loss(y_pred, y)
        loss = mse + gradient_weight * grad

        loss.backward()
        optimizer.step()

        batch_size = x.size(0)
        running_loss += loss.item() * batch_size
        running_mse += mse.item() * batch_size
        running_grad += grad.item() * batch_size

    dataset_size = len(dataloader.dataset)
    return (
        running_loss / dataset_size,
        running_mse / dataset_size,
        running_grad / dataset_size,
    )


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    dataloader,
    criterion,
    device: torch.device,
    gradient_weight: float = 0.01,
) -> tuple[float, float, float]:
    model.eval()
    running_loss = 0.0
    running_mse = 0.0
    running_grad = 0.0

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        y_pred = model(x)
        y_pred = torch.clamp(y_pred, 0.0, 1.0)

        mse = criterion(y_pred, y)
        grad = gradient_loss(y_pred, y)
        loss = mse + gradient_weight * grad

        batch_size = x.size(0)
        running_loss += loss.item() * batch_size
        running_mse += mse.item() * batch_size
        running_grad += grad.item() * batch_size

    dataset_size = len(dataloader.dataset)
    return (
        running_loss / dataset_size,
        running_mse / dataset_size,
        running_grad / dataset_size,
    )


@torch.no_grad()
def save_prediction_preview(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    output_path: str | Path,
) -> None:
    model.eval()

    x, y = next(iter(dataloader))
    x = x.to(device)
    y = y.to(device)

    y_pred = model(x)

    x_cpu = x[0].cpu()
    y_cpu = y[0, 0].cpu().numpy()
    y_pred_cpu = torch.clamp(y_pred[0, 0], 0, 1).cpu().numpy()


    texture = x_cpu[:3].permute(1, 2, 0).numpy()
    segmentation = x_cpu[3].numpy()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(texture)
    axes[0].set_title("Texture")
    axes[0].axis("off")

    axes[1].imshow(segmentation, cmap="gray")
    axes[1].set_title("Segmentation")
    axes[1].axis("off")

    axes[2].imshow(y_cpu, cmap="terrain")
    axes[2].set_title("Target heightmap")
    axes[2].axis("off")

    axes[3].imshow(y_pred_cpu, cmap="terrain")
    axes[3].set_title("Predicted heightmap")
    axes[3].axis("off")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)