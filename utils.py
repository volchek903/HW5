from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision.utils import save_image


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_directories(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def initialize_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def save_sample_grid(
    generator: torch.nn.Module,
    noise: torch.Tensor,
    output_path: Path,
    device: torch.device,
    nrow: int = 4,
) -> None:
    was_training = generator.training
    generator.eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        samples = generator(noise.to(device)).cpu()

    save_image(samples, output_path, nrow=nrow, normalize=True, value_range=(-1, 1))

    if was_training:
        generator.train()


def plot_losses(
    generator_losses: Sequence[float],
    discriminator_losses: Sequence[float],
    output_path: Path,
) -> None:
    epochs = range(1, len(generator_losses) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, generator_losses, label="Generator loss", linewidth=2)
    plt.plot(epochs, discriminator_losses, label="Discriminator loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GAN Training Losses")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def save_training_history(
    history_rows: Sequence[dict[str, float | int]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["epoch", "generator_loss", "discriminator_loss"]

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history_rows)


def save_training_summary(summary: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2, ensure_ascii=False, default=str)


def create_progress_figure(
    sample_records: Sequence[tuple[int, Path]],
    output_path: Path,
) -> None:
    if not sample_records:
        return

    columns = min(3, len(sample_records))
    rows = (len(sample_records) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows))

    if hasattr(axes, "ravel"):
        axes_list = list(axes.ravel())
    else:
        axes_list = [axes]

    for axis in axes_list:
        axis.axis("off")

    for axis, (epoch, sample_path) in zip(axes_list, sample_records):
        image = plt.imread(sample_path)
        axis.imshow(image)
        axis.set_title(f"Epoch {epoch:03d}")
        axis.axis("off")

    figure.suptitle("Evolution of Generated Fashion-MNIST Samples", fontsize=14)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path)
    plt.close(figure)
