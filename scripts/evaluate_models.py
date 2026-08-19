from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader, Dataset, Subset

from projekt.data.PytorchDataset import TerrainDataset
from projekt.models.baseline_cnn import BaselineTerrainCNN
from projekt.models.unet import UNetTerrainModel
from scipy.stats import gaussian_kde

# ============================================================
# Konfiguracja modeli
# ============================================================

@dataclass(frozen=True)
class ModelConfig:
    name: str
    checkpoint_path: Path
    constructor: Callable[[], torch.nn.Module]


@dataclass(frozen=True)
class EvaluationConfig:
    batch_size: int = 1
    num_workers: int = 0
    split_seed: int = 42
    warmup_batches: int = 10
    preview_count: int = 5
    clamp_predictions: bool = True


# ============================================================
# Funkcje metryk
# ============================================================

def calculate_mse(prediction: np.ndarray, target: np.ndarray) -> float:
    """Średni błąd kwadratowy."""
    difference = prediction.astype(np.float64) - target.astype(np.float64)
    return float(np.mean(difference ** 2))


def calculate_psnr(
    prediction: np.ndarray,
    target: np.ndarray,
    data_range: float = 1.0,
) -> float:
    """Peak Signal-to-Noise Ratio."""
    mse = calculate_mse(prediction, target)

    if mse == 0:
        return float("inf")

    return float(10.0 * math.log10((data_range ** 2) / mse))


def calculate_ssim(
    prediction: np.ndarray,
    target: np.ndarray,
    data_range: float = 1.0,
) -> float:
    """Structural Similarity Index."""
    return float(
        structural_similarity(
            target,
            prediction,
            data_range=data_range,
        )
    )


# ============================================================
# Funkcje pomocnicze
# ============================================================

def synchronize_device(device: torch.device) -> None:
    """Synchronizuje GPU przed i po pomiarze czasu."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Zamienia tensor heightmapy na tablicę 2D.

    Obsługiwane kształty:
    [1, H, W]
    [H, W]
    """
    array = tensor.detach().cpu().float().numpy()

    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]

    if array.ndim != 2:
        raise ValueError(
            f"Oczekiwano mapy 2D lub [1,H,W], otrzymano: {array.shape}"
        )

    return array


def save_grayscale_image(array: np.ndarray, output_path: Path) -> None:
    """Zapisuje tablicę z zakresu 0–1 jako PNG."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    normalized = np.clip(array, 0.0, 1.0)
    image = (normalized * 255.0).round().astype(np.uint8)

    Image.fromarray(image, mode="L").save(output_path)


def get_sample_identifier(
    dataset: Dataset,
    subset_index: int,
) -> str:
    if isinstance(dataset, Subset):
        original_index = int(dataset.indices[subset_index])
        source_dataset = dataset.dataset
    else:
        original_index = subset_index
        source_dataset = dataset

    dataframe = getattr(source_dataset, "df", None)

    if dataframe is not None and "id" in dataframe.columns:
        return str(dataframe.iloc[original_index]["id"])

    return str(original_index)


def load_test_subset(
    full_dataset: Dataset,
    split_path: Path,
) -> Subset:
    if not split_path.exists():
        raise FileNotFoundError(
            f"Nie znaleziono pliku podziału danych: {split_path}"
        )

    with open(split_path, "r", encoding="utf-8") as file:
        split_data = json.load(file)

    if "test_indices" not in split_data:
        raise KeyError(
            f"Plik {split_path} nie zawiera pola 'test_indices'."
        )

    test_indices = [
        int(index)
        for index in split_data["test_indices"]
    ]

    if not test_indices:
        raise ValueError("Lista indeksów testowych jest pusta.")

    if max(test_indices) >= len(full_dataset):
        raise IndexError(
            "Co najmniej jeden indeks testowy przekracza "
            "rozmiar aktualnego zbioru danych."
        )

    return Subset(full_dataset, test_indices)


def load_model(
    model_config: ModelConfig,
    device: torch.device,
) -> torch.nn.Module:
    if not model_config.checkpoint_path.exists():
        raise FileNotFoundError(
            f"Nie znaleziono checkpointu modelu "
            f"{model_config.name}: {model_config.checkpoint_path}"
        )

    model = model_config.constructor().to(device)

    state_dict = torch.load(
        model_config.checkpoint_path,
        map_location=device,
    )
    if (
        isinstance(state_dict, dict)
        and "model_state_dict" in state_dict
    ):
        state_dict = state_dict["model_state_dict"]

    model.load_state_dict(state_dict)
    model.eval()

    return model


# ============================================================
# Ewaluacja pojedynczego modelu
# ============================================================

class ModelEvaluator:
    def __init__(
        self,
        model_config: ModelConfig,
        dataset: Dataset,
        dataloader: DataLoader,
        device: torch.device,
        output_dir: Path,
        evaluation_config: EvaluationConfig,
    ) -> None:
        self.model_config = model_config
        self.dataset = dataset
        self.dataloader = dataloader
        self.device = device
        self.output_dir = output_dir
        self.config = evaluation_config

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = load_model(model_config, device)

    def warmup(self) -> None:
        if self.config.warmup_batches <= 0:
            return

        print(
            f"[{self.model_config.name}] "
            f"Warm-up: {self.config.warmup_batches} iteracji"
        )

        with torch.inference_mode():
            for batch_index, batch in enumerate(self.dataloader):
                if batch_index >= self.config.warmup_batches:
                    break

                inputs, _ = batch
                inputs = inputs.to(
                    self.device,
                    non_blocking=True,
                )

                _ = self.model(inputs)

        synchronize_device(self.device)

    def evaluate(self) -> pd.DataFrame:
        self.warmup()

        results: list[dict[str, float | int | str]] = []

        prediction_dir = self.output_dir / "predictions"
        prediction_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"\n[{self.model_config.name}] "
            f"Liczba próbek: {len(self.dataset)}"
        )

        global_sample_index = 0

        with torch.inference_mode():
            for batch_index, batch in enumerate(self.dataloader):
                inputs, targets = batch

                inputs = inputs.to(
                    self.device,
                    non_blocking=True,
                )
                targets = targets.to(
                    self.device,
                    non_blocking=True,
                )

                synchronize_device(self.device)
                start_time = time.perf_counter()

                predictions = self.model(inputs)

                synchronize_device(self.device)
                elapsed_seconds = time.perf_counter() - start_time

                if self.config.clamp_predictions:
                    predictions = torch.clamp(
                        predictions,
                        0.0,
                        1.0,
                    )

                batch_size = inputs.shape[0]
                time_per_sample_ms = (
                    elapsed_seconds * 1000.0 / batch_size
                )

                for local_index in range(batch_size):
                    prediction = tensor_to_numpy(
                        predictions[local_index]
                    )
                    target = tensor_to_numpy(
                        targets[local_index]
                    )

                    sample_id = get_sample_identifier(
                        self.dataset,
                        global_sample_index,
                    )

                    mse = calculate_mse(prediction, target)
                    psnr = calculate_psnr(
                        prediction,
                        target,
                        data_range=1.0,
                    )
                    ssim = calculate_ssim(
                        prediction,
                        target,
                        data_range=1.0,
                    )

                    results.append(
                        {
                            "model": self.model_config.name,
                            "sample_index": global_sample_index,
                            "sample_id": sample_id,
                            "mse": mse,
                            "psnr": psnr,
                            "ssim": ssim,
                            "time_ms": time_per_sample_ms,
                        }
                    )

                    if (
                        global_sample_index
                        < self.config.preview_count
                    ):
                        save_grayscale_image(
                            prediction,
                            prediction_dir
                            / f"{sample_id}_prediction.png",
                        )

                        save_grayscale_image(
                            target,
                            prediction_dir
                            / f"{sample_id}_ground_truth.png",
                        )

                    global_sample_index += 1

                if (batch_index + 1) % 50 == 0:
                    print(
                        f"[{self.model_config.name}] "
                        f"Przetworzono "
                        f"{global_sample_index}/{len(self.dataset)}"
                    )

        dataframe = pd.DataFrame(results)

        metrics_path = self.output_dir / "metrics_per_sample.csv"
        dataframe.to_csv(metrics_path, index=False)

        print(
            f"[{self.model_config.name}] "
            f"Zapisano wyniki: {metrics_path}"
        )

        return dataframe


# ============================================================
# Statystyki
# ============================================================

def save_mse_ecdf(
    metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 5))

    for model_name, model_data in metrics.groupby("model"):
        values = (
            model_data["mse"]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .sort_values()
            .to_numpy()
        )

        if len(values) == 0:
            continue

        cumulative_probability = (
            np.arange(1, len(values) + 1)
            / len(values)
        )

        plt.step(
            values,
            cumulative_probability,
            where="post",
            label=model_name,
        )

    plt.title("Dystrybuanta empiryczna wartości MSE")
    plt.xlabel("MSE")
    plt.ylabel("Odsetek próbek")
    plt.xscale("log")
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        output_path,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close()

def calculate_summary(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    metric_columns = [
        "mse",
        "psnr",
        "ssim",
        "time_ms",
    ]

    summary_rows: list[dict[str, float | str | int]] = []

    for model_name, model_data in metrics.groupby("model"):
        for metric_name in metric_columns:
            values = model_data[metric_name].replace(
                [np.inf, -np.inf],
                np.nan,
            ).dropna()

            if values.empty:
                continue
            count = int(values.count())
            std = float(values.std(ddof=1))

            ci95 = 1.96 * std / math.sqrt(count) if count > 1 else 0.0
            summary_rows.append(
                {
                    "model": model_name,
                    "metric": metric_name,
                    "count": count,
                    "mean": float(values.mean()),
                    "std": std,
                    "ci95": ci95,
                    "median": float(values.median()),
                    "min": float(values.min()),
                    "q1": float(values.quantile(0.25)),
                    "q3": float(values.quantile(0.75)),
                    "p95": float(values.quantile(0.95)),
                    "max": float(values.max()),
                }
            )

    return pd.DataFrame(summary_rows)


def create_compact_summary(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    for model_name in summary["model"].unique():
        model_summary = summary[
            summary["model"] == model_name
        ]

        output_row: dict[str, str] = {
            "model": model_name,
        }

        for metric in ["mse", "psnr", "ssim", "time_ms"]:
            metric_row = model_summary[
                model_summary["metric"] == metric
            ]

            if metric_row.empty:
                output_row[metric] = ""
                continue

            mean_value = float(metric_row.iloc[0]["mean"])
            std_value = float(metric_row.iloc[0]["std"])

            if metric == "mse":
                output_row[metric] = (
                    f"{mean_value:.6f} ± {std_value:.6f}"
                )
            elif metric == "time_ms":
                output_row[metric] = (
                    f"{mean_value:.3f} ± {std_value:.3f}"
                )
            else:
                output_row[metric] = (
                    f"{mean_value:.4f} ± {std_value:.4f}"
                )

        rows.append(output_row)

    return pd.DataFrame(rows)

def select_representative_samples(
    metrics: pd.DataFrame,
) -> list[int]:
    sample_difficulty = (
        metrics.groupby("sample_index", as_index=False)["mse"]
        .mean()
        .sort_values("mse")
        .reset_index(drop=True)
    )

    if sample_difficulty.empty:
        return []

    best_index = int(
        sample_difficulty.iloc[0]["sample_index"]
    )

    median_position = len(sample_difficulty) // 2
    median_index = int(
        sample_difficulty.iloc[median_position]["sample_index"]
    )

    worst_index = int(
        sample_difficulty.iloc[-1]["sample_index"]
    )

    return [
        best_index,
        median_index,
        worst_index,
    ]

# ============================================================
# Wykresy
# ============================================================

def save_histogram(
    metrics: pd.DataFrame,
    metric_name: str,
    output_path: Path,
    title: str,
    xlabel: str,
) -> None:
    plt.figure(figsize=(8, 5))

    for model_name, model_data in metrics.groupby("model"):
        values = model_data[metric_name].replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()

        plt.hist(
            values,
            bins=40,
            alpha=0.55,
            label=model_name,
        )

        if metric_name == "mse":
            plt.xscale("log")

        if metric_name == "time_ms":
            plt.xlim(0, np.percentile(values, 99))

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Liczba próbek")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

def save_time_histogram(
    metrics: pd.DataFrame,
    output_path: Path,
):
    models = list(metrics["model"].unique())

    fig, axes = plt.subplots(
        1,
        len(models),
        figsize=(10,4),
        sharey=True
    )

    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):

        values = (
            metrics.loc[
                metrics["model"] == model,
                "time_ms"
            ]
            .replace([np.inf,-np.inf], np.nan)
            .dropna()
        )

        ax.hist(
            values,
            bins=30,
            color="tab:blue",
            edgecolor="black",
            alpha=0.75
        )

        ax.set_title(model)
        ax.set_xlabel("Czas [ms]")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Liczba próbek")

    plt.suptitle("Histogram czasu generowania map wysokości")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=250,
        bbox_inches="tight"
    )

    plt.close()

def save_time_kde(
    metrics: pd.DataFrame,
    output_path: Path,
):

    plt.figure(figsize=(8,5))

    for model, data in metrics.groupby("model"):

        values = (
            data["time_ms"]
            .replace([np.inf,-np.inf], np.nan)
            .dropna()
            .to_numpy()
        )

        kde = gaussian_kde(values)

        x = np.linspace(
            values.min(),
            values.max(),
            300
        )

        plt.plot(
            x,
            kde(x),
            linewidth=2,
            label=model
        )

    plt.xlabel("Czas [ms]")
    plt.ylabel("Gęstość")
    plt.title("Estymacja gęstości rozkładu czasu generowania")
    plt.grid(alpha=0.25)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=250,
        bbox_inches="tight"
    )

    plt.close()

def save_boxplot(
    metrics: pd.DataFrame,
    metric_name: str,
    output_path: Path,
    title: str,
    ylabel: str,
) -> None:
    model_names = list(metrics["model"].unique())

    values = [
        metrics.loc[
            metrics["model"] == model_name,
            metric_name,
        ]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy()
        for model_name in model_names
    ]

    plt.figure(figsize=(7, 5))
    plt.boxplot(
        values,
        tick_labels=model_names,
        showmeans=True,
        whis=(5,95)
    )
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()


def save_mean_comparison_chart(
    summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    plot_config = {
        "mse": (
            "Średnia wartość MSE",
            "MSE",
            "mean_mse.png",
        ),
        "psnr": (
            "Średnia wartość PSNR",
            "PSNR [dB]",
            "mean_psnr.png",
        ),
        "ssim": (
            "Średnia wartość SSIM",
            "SSIM",
            "mean_ssim.png",
        ),
        "time_ms": (
            "Średni czas generowania pojedynczej mapy",
            "Czas [ms]",
            "mean_inference_time.png",
        ),
    }

    for metric_name, (
        title,
        ylabel,
        filename,
    ) in plot_config.items():
        metric_summary = summary[
            summary["metric"] == metric_name
        ]

        if metric_summary.empty:
            continue

        plt.figure(figsize=(7, 5))

        bars = plt.bar(
            metric_summary["model"],
            metric_summary["mean"],
            yerr=metric_summary["ci95"],
            capsize=6,
        )
        plt.figtext(
            0.5,
            0.01,
            "Słupki błędów przedstawiają 95% przedział ufności średniej.",
            ha="center",
            fontsize=9,
        )
        plt.title(title)
        plt.ylabel(ylabel)
        
        for bar, value in zip(
            bars,
            metric_summary["mean"],
        ):
            value_label = "",
            if metric_name == "mse":
                value_label = f"{value:.4f}"
            elif metric_name == "time_ms":
                value_label = f"{value:.2f} ms"
            elif metric_name == "psnr":
                value_label = f"{value:.2f} dB"
            else:
                value_label = f"{value:.3f}"
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                value_label,
                ha="center",
                va="bottom",
            )

        plt.tight_layout(rect=(0, 0.04, 1, 1))
        plt.savefig(
            output_dir / filename,
            dpi=200,
            bbox_inches="tight",
        )
        plt.close()


def save_metric_plots(
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_config = {
        "mse": (
            "Rozkład wartości MSE",
            "MSE",
        ),
        "psnr": (
            "Rozkład wartości PSNR",
            "PSNR [dB]",
        ),
        "ssim": (
            "Rozkład wartości SSIM",
            "SSIM",
        ),
        "time_ms": (
            "Rozkład czasu generowania map wysokości",
            "Czas [ms]",
        ),
    }

    for metric_name, (
        title,
        axis_label,
    ) in metric_config.items():

        if metric_name != "time_ms":
            save_histogram(
                metrics=metrics,
                metric_name=metric_name,
                output_path=output_dir
                / f"histogram_{metric_name}.png",
                title=title,
                xlabel=axis_label,
            )
        else:
            save_time_histogram(metrics, output_dir / "histogram_time_ms.png")

            save_time_kde(metrics, output_dir / "kde_time_ms.png")

        save_boxplot(
            metrics=metrics,
            metric_name=metric_name,
            output_path=output_dir
            / f"boxplot_{metric_name}.png",
            title=f"Porównanie modeli – {metric_name.upper()}",
            ylabel=axis_label,
        )

    save_mean_comparison_chart(
        summary,
        output_dir,
    )

    save_mse_ecdf(
        metrics=metrics,
        output_path=output_dir / "ecdf_mse.png",
    )


def create_prediction_comparison(
    models: dict[str, torch.nn.Module],
    dataset: Dataset,
    device: torch.device,
    output_path: Path,
    sample_indices: list[int],
) -> None:
    valid_indices = [
        index
        for index in sample_indices
        if 0 <= index < len(dataset)
    ]

    if not valid_indices:
        return

    model_names = list(models.keys())

    column_titles = ["Mapa referencyjna"]

    for model_name in model_names:
        column_titles.extend(
            [
                model_name,
                f"Błąd bezwzględny — {model_name}",
            ]
        )

    row_count = len(valid_indices)
    column_count = len(column_titles)

    fig = plt.figure(
        figsize=(18, 10),
        constrained_layout=True,
    )

    grid = fig.add_gridspec(
        nrows=row_count,
        ncols=column_count + 1,
        width_ratios=[1] * column_count + [0.045],
        wspace=0.04,
        hspace=0.04,
    )

    axes = np.empty(
        (row_count, column_count),
        dtype=object,
    )

    for row_index in range(row_count):
        for column_index in range(column_count):
            axes[row_index, column_index] = fig.add_subplot(
                grid[row_index, column_index]
            )

    # Osobna kolumna na wspólny colorbar
    colorbar_axis = fig.add_subplot(grid[:, -1])

    row_labels = [
        "Próbka łatwa",
        "Próbka typowa",
        "Próbka trudna",
    ]

    for row_position, sample_index in enumerate(valid_indices):
        inputs, target = dataset[sample_index]

        input_batch = inputs.unsqueeze(0).to(device)
        target_array = tensor_to_numpy(target)

        reference_axis = axes[row_position, 0]
        reference_axis.imshow(
            target_array,
            cmap="terrain",
            vmin=0.0,
            vmax=1.0,
        )
        reference_axis.axis("off")

        if row_position == 0:
            reference_axis.set_title(column_titles[0])

        if row_position < len(row_labels):
            reference_axis.set_ylabel(
                row_labels[row_position],
                fontsize=11,
            )

        column_position = 1

        for model_name in model_names:
            model = models[model_name]

            with torch.inference_mode():
                prediction = model(input_batch)

                prediction = torch.clamp(
                    prediction,
                    0.0,
                    1.0,
                )[0]

            prediction_array = tensor_to_numpy(prediction)

            absolute_error = np.abs(
                prediction_array - target_array
            )

            prediction_axis = axes[
                row_position,
                column_position,
            ]

            prediction_axis.imshow(
                prediction_array,
                cmap="terrain",
                vmin=0.0,
                vmax=1.0,
            )
            prediction_axis.axis("off")

            error_axis = axes[
                row_position,
                column_position + 1,
            ]

            error_image = error_axis.imshow(
                absolute_error,
                cmap="inferno",
                vmin=0.0,
                vmax=1.0,
            )
            error_axis.axis("off")

            if row_position == 0:
                prediction_axis.set_title(model_name)
                error_axis.set_title(
                    f"Błąd bezwzględny\n{model_name}"
                )

            column_position += 2

    colorbar = fig.colorbar(
        error_image,
        cax=colorbar_axis,
    )

    colorbar.set_label(
        "Błąd bezwzględny",
        rotation=90,
        labelpad=12,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # plt.tight_layout(rect=(0, 0, 0.95, 1))

    plt.savefig(
        output_path,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Informacje o środowisku
# ============================================================

def save_environment_information(
    output_path: Path,
    device: torch.device,
) -> None:
    information = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "device": str(device),
    }

    if torch.cuda.is_available():
        information["gpu_name"] = torch.cuda.get_device_name(device)
        information["gpu_memory_bytes"] = (
            torch.cuda.get_device_properties(device).total_memory
        )

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            information,
            file,
            indent=4,
            ensure_ascii=False,
        )

def save_latex_comparison_table(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    metric_labels = {
        "mse": "MSE $\\downarrow$",
        "psnr": "PSNR [dB] $\\uparrow$",
        "ssim": "SSIM $\\uparrow$",
        "time_ms": "Czas [ms] $\\downarrow$",
    }

    models = list(summary["model"].unique())

    values: dict[str, dict[str, tuple[float, float]]] = {}

    for model_name in models:
        values[model_name] = {}

        for metric_name in metric_labels:
            row = summary[
                (summary["model"] == model_name)
                & (summary["metric"] == metric_name)
            ]

            if row.empty:
                continue

            values[model_name][metric_name] = (
                float(row.iloc[0]["mean"]),
                float(row.iloc[0]["std"]),
            )

    best_models: dict[str, str] = {}

    for metric_name in metric_labels:
        metric_means = {
            model_name: values[model_name][metric_name][0]
            for model_name in models
            if metric_name in values[model_name]
        }

        if metric_name in {"mse", "time_ms"}:
            best_models[metric_name] = min(
                metric_means,
                key=metric_means.get,
            )
        else:
            best_models[metric_name] = max(
                metric_means,
                key=metric_means.get,
            )

    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Porównanie jakości i czasu inferencji modeli}",
        "\\label{tab:model_comparison}",
        "\\begin{tabular}{lcccc}",
        "\\hline",
        (
            "\\textbf{Model} & "
            "\\textbf{MSE $\\downarrow$} & "
            "\\textbf{PSNR [dB] $\\uparrow$} & "
            "\\textbf{SSIM $\\uparrow$} & "
            "\\textbf{Czas [ms] $\\downarrow$} \\\\"
        ),
        "\\hline",
    ]

    for model_name in models:
        formatted_values = []

        for metric_name in metric_labels:
            mean_value, std_value = values[model_name][metric_name]

            if metric_name == "mse":
                formatted = (
                    f"{mean_value:.4f} $\\pm$ {std_value:.4f}"
                )
            elif metric_name == "time_ms":
                formatted = (
                    f"{mean_value:.2f} $\\pm$ {std_value:.2f}"
                )
            else:
                formatted = (
                    f"{mean_value:.3f} $\\pm$ {std_value:.3f}"
                )

            if best_models[metric_name] == model_name:
                formatted = f"\\textbf{{{formatted}}}"

            formatted_values.append(formatted)

        lines.append(
            f"{model_name} & "
            + " & ".join(formatted_values)
            + " \\\\"
        )

    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

# ============================================================
# Argumenty programu
# ============================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ewaluacja modeli Baseline CNN i U-Net "
            "na wspólnym, niezależnym zbiorze testowym."
        )
    )

    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=None,
        help="Ścieżka do checkpointu Baseline CNN.",
    )

    parser.add_argument(
        "--unet-checkpoint",
        type=Path,
        default=None,
        help="Ścieżka do checkpointu U-Net.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Batch size podczas ewaluacji. "
            "Dla pomiaru czasu pojedynczej mapy zalecana wartość: 1."
        ),
    )

    parser.add_argument(
        "--preview-count",
        type=int,
        default=5,
        help="Liczba zapisywanych przykładowych predykcji.",
    )

    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=10,
        help="Liczba iteracji rozgrzewkowych przed pomiarem czasu.",
    )

    return parser.parse_args()

def save_training_history_plots(
    baseline_history_path: Path,
    unet_history_path: Path,
    output_dir: Path,
) -> None:
    history_paths = {
        "Baseline CNN": baseline_history_path,
        "U-Net": unet_history_path,
    }

    histories: dict[str, pd.DataFrame] = {}

    for model_name, history_path in history_paths.items():
        if not history_path.exists():
            print(
                f"Pominięto historię {model_name}: "
                f"brak pliku {history_path}"
            )
            continue

        histories[model_name] = pd.read_csv(history_path)

    if not histories:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    plot_definitions = [
        (
            "train_loss",
            "validation_loss",
            "Całkowita funkcja straty",
            "Wartość funkcji straty",
            "training_loss_comparison.png",
        ),
        (
            "train_mse",
            "validation_mse",
            "Przebieg błędu MSE",
            "MSE",
            "training_mse_comparison.png",
        ),
        (
            "train_gradient_loss",
            "validation_gradient_loss",
            "Przebieg składnika gradientowego",
            "Gradient loss",
            "training_gradient_comparison.png",
        ),
    ]

    for (
        train_column,
        validation_column,
        title,
        ylabel,
        filename,
    ) in plot_definitions:
        plt.figure(figsize=(9, 5.5))

        plotted_anything = False

        for model_name, dataframe in histories.items():
            # Obsługa starszych nazw kolumn.
            actual_train_column = train_column
            actual_validation_column = validation_column

            if train_column == "train_gradient_loss":
                if train_column not in dataframe.columns:
                    actual_train_column = "train_grad"

                if validation_column not in dataframe.columns:
                    actual_validation_column = "validation_grad"

            required_columns = {
                "epoch",
                actual_train_column,
                actual_validation_column,
            }

            if not required_columns.issubset(
                dataframe.columns
            ):
                print(
                    f"Pominięto wykres {title} dla "
                    f"{model_name}: brak wymaganych kolumn."
                )
                continue

            plt.plot(
                dataframe["epoch"],
                dataframe[actual_train_column],
                label=f"{model_name} — trening",
            )

            plt.plot(
                dataframe["epoch"],
                dataframe[actual_validation_column],
                linestyle="--",
                label=f"{model_name} — walidacja",
            )

            plotted_anything = True

        if not plotted_anything:
            plt.close()
            continue

        plt.title(title)
        plt.xlabel("Epoka")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()

        plt.savefig(
            output_dir / filename,
            dpi=250,
            bbox_inches="tight",
        )

        plt.close()

# ============================================================
# Główna funkcja
# ============================================================

def main() -> None:
    arguments = parse_arguments()

    project_root = Path(__file__).resolve().parents[1]

    processed_root = project_root / "data" / "processed"
    index_path = processed_root / "index.csv"
    split_path = project_root / "outputs" / "dataset_splits" / "split_70_15_15_seed_42.json"


    output_root = project_root / "outputs" / "evaluation"
    output_root.mkdir(parents=True, exist_ok=True)

    baseline_history_path = project_root / "outputs" / "baseline" / "training_history.csv"
    unet_history_path = project_root / "outputs" / "unet_segmentation" / "training_history.csv"

    baseline_checkpoint = (
        arguments.baseline_checkpoint
        if arguments.baseline_checkpoint is not None
        else project_root
        / "outputs"
        / "baseline"
        / "baseline_cnn.pt"
    )

    unet_checkpoint = (
        arguments.unet_checkpoint
        if arguments.unet_checkpoint is not None
        else project_root
        / "outputs"
        / "unet_segmentation"
        / "unet_segmentation.pt"
    )

    if not index_path.exists():
        raise FileNotFoundError(
            f"Nie znaleziono pliku indeksu: {index_path}"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    evaluation_config = EvaluationConfig(
        batch_size=arguments.batch_size,
        preview_count=arguments.preview_count,
        warmup_batches=arguments.warmup_batches,
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

    evaluation_dataset = load_test_subset(
        full_dataset=full_dataset,
        split_path=split_path,
    )

    dataloader = DataLoader(
        evaluation_dataset,
        batch_size=evaluation_config.batch_size,
        shuffle=False,
        num_workers=evaluation_config.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model_configs = [
        ModelConfig(
            name="Baseline CNN",
            checkpoint_path=baseline_checkpoint,
            constructor=lambda: BaselineTerrainCNN(
                in_channels=4,
                out_channels=1,
            ),
        ),
        ModelConfig(
            name="U-Net",
            checkpoint_path=unet_checkpoint,
            constructor=lambda: UNetTerrainModel(
                in_channels=4,
                out_channels=1,
            ),
        ),
    ]

    all_metrics: list[pd.DataFrame] = []
    loaded_models: dict[str, torch.nn.Module] = {}

    for model_config in model_configs:
        model_output_dir = (
            output_root
            / model_config.name.lower().replace(" ", "_")
        )

        evaluator = ModelEvaluator(
            model_config=model_config,
            dataset=evaluation_dataset,
            dataloader=dataloader,
            device=device,
            output_dir=model_output_dir,
            evaluation_config=evaluation_config,
        )

        model_metrics = evaluator.evaluate()
        all_metrics.append(model_metrics)

        loaded_models[model_config.name] = evaluator.model

    combined_metrics = pd.concat(
        all_metrics,
        ignore_index=True,
    )

    combined_metrics_path = (
        output_root / "all_models_metrics.csv"
    )
    combined_metrics.to_csv(
        combined_metrics_path,
        index=False,
    )

    summary = calculate_summary(combined_metrics)
    summary.to_csv(
        output_root / "summary_statistics.csv",
        index=False,
    )

    save_latex_comparison_table(
        summary=summary,
        output_path=output_root / "comparison_table.tex",
    )

    compact_summary = create_compact_summary(summary)
    compact_summary.to_csv(
        output_root / "comparison_table.csv",
        index=False,
    )

    plots_dir = output_root / "plots"

    save_metric_plots(
        metrics=combined_metrics,
        summary=summary,
        output_dir=plots_dir,
    )

    save_training_history_plots(
        baseline_history_path=baseline_history_path,
        unet_history_path=unet_history_path,
        output_dir=plots_dir / "training",
    )

    comparison_indices = select_representative_samples(combined_metrics)

    print(f"Wybrane próbki porównawcze (łatwa, typowa, trudna): {comparison_indices}")

    create_prediction_comparison(
        models=loaded_models,
        dataset=evaluation_dataset,
        device=device,
        output_path=plots_dir
        / "prediction_comparison.png",
        sample_indices=comparison_indices,
    )

    save_environment_information(
        output_root / "environment.json",
        device,
    )

    print("\n====================================")
    print("EWALUACJA ZAKOŃCZONA")
    print("====================================")
    print(f"Liczba próbek: {len(evaluation_dataset)}")
    print(f"Wyniki: {combined_metrics_path}")
    print("\nTabela zbiorcza:")
    print(compact_summary.to_string(index=False))
    print(f"\nPliki zapisano w: {output_root}")


if __name__ == "__main__":
    main()