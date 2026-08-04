from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
import matplotlib.pyplot as plt

from projekt.data.PytorchDataset import TerrainDataset
from projekt.models.baseline_cnn import BaselineTerrainCNN
from projekt.models.unet import UNetTerrainModel
from projekt.training.train_utils import (
    save_prediction_preview,
    train_one_epoch,
    validate_one_epoch,
)


# ============================================================
# Konfiguracja
# ============================================================

@dataclass(frozen=True)
class TrainingConfig:
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15

    split_seed: int = 42
    dataloader_seed: int = 42

    batch_size: int = 4
    num_workers: int = 0

    num_epochs: int = 100
    learning_rate: float = 3e-4

    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 1e-5

    gradient_weight: float = 0.1


@dataclass(frozen=True)
class ModelConfig:
    name: str
    output_directory: str
    checkpoint_filename: str
    constructor: Callable[[], nn.Module]


# ============================================================
# Powtarzalność
# ============================================================

def set_reproducibility(seed: int) -> None:
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ustawienia zwiększające powtarzalność.
    # Mogą nieznacznie obniżyć wydajność.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Podział danych
# ============================================================

def calculate_split_sizes(
    dataset_size: int,
    config: TrainingConfig,
) -> tuple[int, int, int]:
    ratio_sum = (
        config.train_ratio
        + config.validation_ratio
        + config.test_ratio
    )

    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError(
            f"Suma proporcji podziału musi wynosić 1.0, "
            f"otrzymano {ratio_sum}."
        )

    train_size = int(dataset_size * config.train_ratio)
    validation_size = int(
        dataset_size * config.validation_ratio
    )

    # Resztę przypisujemy do testu, aby suma zawsze była
    # dokładnie równa liczbie próbek.
    test_size = (
        dataset_size
        - train_size
        - validation_size
    )

    return train_size, validation_size, test_size


def create_dataset_splits(
    dataset: TerrainDataset,
    config: TrainingConfig,
) -> tuple[Subset, Subset, Subset]:
    train_size, validation_size, test_size = (
        calculate_split_sizes(len(dataset), config)
    )

    train_dataset, validation_dataset, test_dataset = random_split(
        dataset,
        [train_size, validation_size, test_size],
        generator=torch.Generator().manual_seed(
            config.split_seed
        ),
    )

    return train_dataset, validation_dataset, test_dataset

def save_training_plots(
    history: list[dict[str, float | int | bool]],
    output_directory: Path,
    model_name: str,
) -> None:
    if not history:
        return

    dataframe = pd.DataFrame(history)

    plots_directory = output_directory / "plots"
    plots_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Całkowita funkcja straty
    # ========================================================

    plt.figure(figsize=(9, 5))

    plt.plot(
        dataframe["epoch"],
        dataframe["train_loss"],
        label="Zbiór treningowy",
    )

    plt.plot(
        dataframe["epoch"],
        dataframe["validation_loss"],
        label="Zbiór walidacyjny",
    )

    plt.xlabel("Epoka")
    plt.ylabel("Całkowita funkcja straty")
    plt.title(
        f"Przebieg funkcji straty — {model_name}"
    )
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        plots_directory / "loss_history.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    # ========================================================
    # Podstawowy składnik Smooth L1
    # W kodzie zmienna historycznie nazywa się train_mse,
    # ale criterion to SmoothL1Loss.
    # ========================================================

    plt.figure(figsize=(9, 5))

    plt.plot(
        dataframe["epoch"],
        dataframe["train_mse"],
        label="Zbiór treningowy",
    )

    plt.plot(
        dataframe["epoch"],
        dataframe["validation_mse"],
        label="Zbiór walidacyjny",
    )

    plt.xlabel("Epoka")
    plt.ylabel("MSE")
    plt.title(f"Przebieg błędu MSE — {model_name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        plots_directory / "mse_history.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    # ========================================================
    # Gradient loss
    # ========================================================

    plt.figure(figsize=(9, 5))

    plt.plot(
        dataframe["epoch"],
        dataframe["train_gradient_loss"],
        label="Zbiór treningowy",
    )

    plt.plot(
        dataframe["epoch"],
        dataframe["validation_gradient_loss"],
        label="Zbiór walidacyjny",
    )

    plt.xlabel("Epoka")
    plt.ylabel("Gradient loss")
    plt.title(f"Przebieg składnika gradientowego — {model_name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        plots_directory / "gradient_loss_history.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    # ========================================================
    # Learning rate
    # ========================================================

    if "learning_rate" in dataframe.columns:
        plt.figure(figsize=(9, 5))

        plt.plot(
            dataframe["epoch"],
            dataframe["learning_rate"],
        )

        plt.xlabel("Epoka")
        plt.ylabel("Learning rate")
        plt.title(
            f"Zmiana współczynnika uczenia — "
            f"{model_name}"
        )
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            plots_directory / "learning_rate_history.png",
            dpi=200,
            bbox_inches="tight",
        )

        plt.close()

def save_split_indices(
    train_dataset: Subset,
    validation_dataset: Subset,
    test_dataset: Subset,
    output_path: Path,
    config: TrainingConfig,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    split_data = {
        "split_seed": config.split_seed,
        "train_ratio": config.train_ratio,
        "validation_ratio": config.validation_ratio,
        "test_ratio": config.test_ratio,
        "train_indices": [
            int(index) for index in train_dataset.indices
        ],
        "validation_indices": [
            int(index) for index in validation_dataset.indices
        ],
        "test_indices": [
            int(index) for index in test_dataset.indices
        ],
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            split_data,
            file,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# DataLoadery
# ============================================================

def create_dataloaders(
    train_dataset: Subset,
    validation_dataset: Subset,
    test_dataset: Subset,
    config: TrainingConfig,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_generator = torch.Generator().manual_seed(
        config.dataloader_seed
    )

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        generator=train_generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, validation_loader, test_loader


# ============================================================
# Zapis historii
# ============================================================

def save_history(
    history: list[dict[str, float | int]],
    output_path: Path,
) -> None:
    dataframe = pd.DataFrame(history)
    dataframe.to_csv(output_path, index=False)


# ============================================================
# Trening pojedynczego modelu
# ============================================================

def train_model(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    outputs_root: Path,
) -> dict[str, float | str]:
    model_output_directory = (
        outputs_root / model_config.output_directory
    )
    model_output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = model_config.constructor().to(device)
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    print(f"Trainable parameters: {trainable_parameters}")



    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config.learning_rate,
    )

    checkpoint_path = (
        model_output_directory
        / model_config.checkpoint_filename
    )

    history_path = (
        model_output_directory
        / "training_history.csv"
    )

    history: list[dict[str, float | int]] = []

    best_validation_loss = float("inf")
    best_epoch = 0

    epochs_without_improvement = 0
    stopped_epoch = training_config.num_epochs

    print("\n" + "=" * 70)
    print(f"TRAINING MODEL: {model_config.name}")
    print("=" * 70)

    for epoch in range(1, training_config.num_epochs + 1):
        epoch_start = time.perf_counter()

        train_loss, train_mse, train_grad = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            gradient_weight=training_config.gradient_weight,
        )

        validation_loss, validation_mse, validation_grad = (
            validate_one_epoch(
                model=model,
                dataloader=validation_loader,
                criterion=criterion,
                device=device,
                gradient_weight=(
                    training_config.gradient_weight
                ),
            )
        )

        epoch_time = time.perf_counter() - epoch_start
        remaining_epochs = (
            training_config.num_epochs - epoch
        )
        estimated_remaining_seconds = (
            epoch_time * remaining_epochs
        )
        
        improvement = best_validation_loss - validation_loss

        is_best_epoch = improvement > training_config.early_stopping_min_delta

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_mse": train_mse,
                "train_gradient_loss": train_grad,
                "validation_loss": validation_loss,
                "validation_mse": validation_mse,
                "validation_gradient_loss": validation_grad,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epoch_time_seconds": epoch_time,
                "is_best_epoch": is_best_epoch,
                "epochs_without_improvement": (epochs_without_improvement),
            }
        )

        print(
            f"Epoch {epoch:02d}/"
            f"{training_config.num_epochs} | "
            f"train_loss={train_loss:.6f} | "
            f"train_mse={train_mse:.6f} | "
            f"train_grad={train_grad:.6f} | "
            f"val_loss={validation_loss:.6f} | "
            f"val_mse={validation_mse:.6f} | "
            f"val_grad={validation_grad:.6f} | "
            f"time={epoch_time:.2f}s | "
            f"ETA={estimated_remaining_seconds / 60:.1f}min"
        )

        improvement = best_validation_loss - validation_loss

        if improvement > training_config.early_stopping_min_delta:
            best_validation_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                model.state_dict(),
                checkpoint_path,
            )

            print(
                f"  -> Saved best checkpoint "
                f"(epoch={epoch}, "
                f"val_loss={validation_loss:.6f})"
            )
        else:
            epochs_without_improvement += 1

            print(
                f"  -> No validation improvement: "
                f"{epochs_without_improvement}/"
                f"{training_config.early_stopping_patience}"
            )

        save_prediction_preview(
            model=model,
            dataloader=validation_loader,
            device=device,
            output_path=(
                model_output_directory
                / f"preview_epoch_{epoch:02d}.png"
            ),
        )

        if epochs_without_improvement >= training_config.early_stopping_patience:
            stopped_epoch = epoch

            print("\nEarly stopping activated.")
            print(
                f"No validation improvement for "
                f"{training_config.early_stopping_patience} "
                f"consecutive epochs."
            )
            print(f"Training stopped at epoch {epoch}. Best epoch: {best_epoch}.")

            break

    save_history(history, history_path)
    save_training_plots(
        history=history,
        output_directory=model_output_directory,
        model_name=model_config.name,
    )

    # ========================================================
    # Test — dopiero po zakończeniu treningu
    # ========================================================

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device,
        )
    )
    model.eval()

    test_loss, test_mse, test_grad = validate_one_epoch(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        gradient_weight=training_config.gradient_weight,
    )

    test_results = {
        "model": model_config.name,
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "best_validation_loss": best_validation_loss,
        "test_loss": test_loss,
        "test_mse": test_mse,
        "test_gradient_loss": test_grad,
        "checkpoint": str(checkpoint_path),
    }

    pd.DataFrame([test_results]).to_csv(
        model_output_directory / "test_results.csv",
        index=False,
    )

    save_prediction_preview(
        model=model,
        dataloader=test_loader,
        device=device,
        output_path=(
            model_output_directory
            / "test_prediction_preview.png"
        ),
    )

    print("\nTraining completed.")
    print(f"Best epoch: {best_epoch}")
    print(
        f"Best validation loss: "
        f"{best_validation_loss:.6f}"
    )
    print(f"Test loss: {test_loss:.6f}")
    print(f"Test MSE: {test_mse:.6f}")
    print(f"Test gradient loss: {test_grad:.6f}")
    print(f"Model saved to: {checkpoint_path}")
    print(f"History saved to: {history_path}")

    return test_results


# ============================================================
# Główna funkcja
# ============================================================

def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    processed_root = project_root / "data" / "processed"
    index_path = processed_root / "index.csv"

    outputs_root = project_root / "outputs"
    splits_path = (
        outputs_root
        / "dataset_splits"
        / "split_70_15_15_seed_42.json"
    )

    training_config = TrainingConfig()

    set_reproducibility(training_config.split_seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(device),
        )

    if not index_path.exists():
        raise FileNotFoundError(
            f"Nie znaleziono pliku indeksu: {index_path}"
        )

    full_dataset = TerrainDataset(
        index_path=index_path,
        data_root=processed_root,
        use_texture=True,
        use_segmentation=True,
        normalize_texture=True,
        normalize_heightmap=True,
        segmentation_mode="normalized",
    )

    (
        train_dataset,
        validation_dataset,
        test_dataset,
    ) = create_dataset_splits(
        dataset=full_dataset,
        config=training_config,
    )

    print("\nDATASET SPLIT")
    print("-------------")
    print(f"All samples:        {len(full_dataset)}")
    print(f"Training samples:   {len(train_dataset)}")
    print(
        f"Validation samples: "
        f"{len(validation_dataset)}"
    )
    print(f"Test samples:       {len(test_dataset)}")

    save_split_indices(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
        output_path=splits_path,
        config=training_config,
    )

    print(f"Split indices saved to: {splits_path}")

    # (
    #     train_loader,
    #     validation_loader,
    #     test_loader,
    # ) = create_dataloaders(
    #     train_dataset=train_dataset,
    #     validation_dataset=validation_dataset,
    #     test_dataset=test_dataset,
    #     config=training_config,
    #     device=device,
    # )

    model_configs = [
        ModelConfig(
            name="Baseline CNN",
            output_directory="baseline",
            checkpoint_filename="baseline_cnn.pt",
            constructor=lambda: BaselineTerrainCNN(
                in_channels=4,
                out_channels=1,
            ),
        ),
        ModelConfig(
            name="U-Net",
            output_directory="unet_segmentation",
            checkpoint_filename="unet_segmentation.pt",
            constructor=lambda: UNetTerrainModel(
                in_channels=4,
                out_channels=1,
            ),
        ),
    ]

    all_test_results = []

    for model_config in model_configs:
        # Każdy model otrzymuje tę samą inicjalizację generatorów
        # oraz tę samą kolejność paczek treningowych.
        set_reproducibility(training_config.split_seed)

        (
            train_loader,
            validation_loader,
            test_loader,
        ) = create_dataloaders(
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            test_dataset=test_dataset,
            config=training_config,
            device=device,
        )

        test_result = train_model(
            model_config=model_config,
            training_config=training_config,
            train_loader=train_loader,
            validation_loader=validation_loader,
            test_loader=test_loader,
            device=device,
            outputs_root=outputs_root,
        )

        all_test_results.append(test_result)

    comparison_path = (
        outputs_root
        / "test_results_comparison.csv"
    )

    pd.DataFrame(all_test_results).to_csv(
        comparison_path,
        index=False,
    )

    print("\n" + "=" * 70)
    print("TRAINING OF BOTH MODELS COMPLETED")
    print("=" * 70)
    print(f"Comparison saved to: {comparison_path}")


if __name__ == "__main__":
    main()