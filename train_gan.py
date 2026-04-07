from __future__ import annotations

from time import perf_counter

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import config
from models import Discriminator, Generator
from utils import (
    create_progress_figure,
    ensure_directories,
    initialize_weights,
    plot_losses,
    save_sample_grid,
    save_training_history,
    save_training_summary,
    set_seed,
)


def get_dataloader() -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    dataset = datasets.FashionMNIST(
        root=config.data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def save_training_artifacts(
    generator: Generator,
    discriminator: Discriminator,
    generator_losses: list[float],
    discriminator_losses: list[float],
    history_rows: list[dict[str, float | int]],
    sample_records: list[tuple[int, str]],
    fixed_noise: torch.Tensor,
    device: torch.device,
    total_training_time: float,
) -> None:
    torch.save(generator.state_dict(), config.generator_checkpoint)
    torch.save(discriminator.state_dict(), config.discriminator_checkpoint)
    plot_losses(generator_losses, discriminator_losses, config.loss_plot_path)
    save_training_history(history_rows, config.history_path)

    final_sample_path = config.generated_dir / "final_training_samples.png"
    save_sample_grid(generator, fixed_noise, final_sample_path, device=device)
    progress_records = [(epoch, config.samples_dir / filename) for epoch, filename in sample_records]
    create_progress_figure(progress_records, config.progress_plot_path)

    summary = {
        "device": str(device),
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "latent_dim": config.latent_dim,
        "learning_rate": config.learning_rate,
        "sample_interval": config.sample_interval,
        "random_seed": config.random_seed,
        "training_time_seconds": round(total_training_time, 2),
        "final_generator_loss": round(generator_losses[-1], 4),
        "final_discriminator_loss": round(discriminator_losses[-1], 4),
        "generator_checkpoint": str(config.generator_checkpoint),
        "discriminator_checkpoint": str(config.discriminator_checkpoint),
        "loss_plot": str(config.loss_plot_path),
        "progress_plot": str(config.progress_plot_path),
        "history_csv": str(config.history_path),
        "saved_samples": [
            {"epoch": epoch, "path": str(config.samples_dir / filename)}
            for epoch, filename in sample_records
        ],
    }
    save_training_summary(summary, config.summary_path)

    print(f"Чекпоинт генератора сохранён в: {config.generator_checkpoint}")
    print(f"Чекпоинт дискриминатора сохранён в: {config.discriminator_checkpoint}")
    print(f"График потерь сохранён в: {config.loss_plot_path}")
    print(f"Сводная картинка прогресса сохранена в: {config.progress_plot_path}")
    print(f"История обучения сохранена в: {config.history_path}")
    print(f"Сводка обучения сохранена в: {config.summary_path}")
    print(f"Итоговая сетка примеров сохранена в: {final_sample_path}")


def train() -> None:
    set_seed(config.random_seed)
    ensure_directories(
        [
            config.data_dir,
            config.samples_dir,
            config.checkpoints_dir,
            config.plots_dir,
            config.generated_dir,
        ]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = get_dataloader()

    generator = Generator(config.latent_dim).to(device)
    discriminator = Discriminator().to(device)
    generator.apply(initialize_weights)
    discriminator.apply(initialize_weights)

    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(
        generator.parameters(),
        lr=config.learning_rate,
        betas=config.adam_betas,
    )
    optimizer_d = optim.Adam(
        discriminator.parameters(),
        lr=config.learning_rate,
        betas=config.adam_betas,
    )

    fixed_noise = torch.randn(config.num_sample_images, config.latent_dim, device=device)
    generator_losses: list[float] = []
    discriminator_losses: list[float] = []
    history_rows: list[dict[str, float | int]] = []
    sample_records: list[tuple[int, str]] = []
    num_batches = len(dataloader)
    training_start_time = perf_counter()

    print(f"Используемое устройство: {device}")
    print(f"Размер обучающего датасета: {len(dataloader.dataset)} изображений")

    for epoch in range(1, config.epochs + 1):
        epoch_start_time = perf_counter()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        for batch_idx, (real_images, _) in enumerate(dataloader, start=1):
            real_images = real_images.to(device)
            current_batch_size = real_images.size(0)

            real_labels = torch.ones(current_batch_size, 1, device=device)
            fake_labels = torch.zeros(current_batch_size, 1, device=device)

            optimizer_d.zero_grad()

            real_output = discriminator(real_images)
            real_loss = criterion(real_output, real_labels)

            noise = torch.randn(current_batch_size, config.latent_dim, device=device)
            fake_images = generator(noise)
            fake_output = discriminator(fake_images.detach())
            fake_loss = criterion(fake_output, fake_labels)

            d_loss = 0.5 * (real_loss + fake_loss)
            d_loss.backward()
            optimizer_d.step()

            optimizer_g.zero_grad()

            noise = torch.randn(current_batch_size, config.latent_dim, device=device)
            generated_images = generator(noise)
            generator_output = discriminator(generated_images)
            g_loss = criterion(generator_output, real_labels)

            g_loss.backward()
            optimizer_g.step()

            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()

            should_log_batch = (
                batch_idx == 1
                or batch_idx % config.log_interval == 0
                or batch_idx == num_batches
            )
            if should_log_batch:
                print(
                    f"Эпоха [{epoch}/{config.epochs}] "
                    f"Батч [{batch_idx}/{num_batches}] "
                    f"Потери D: {d_loss.item():.4f} "
                    f"Потери G: {g_loss.item():.4f}"
                )

        avg_d_loss = epoch_d_loss / num_batches
        avg_g_loss = epoch_g_loss / num_batches
        epoch_duration = perf_counter() - epoch_start_time

        discriminator_losses.append(avg_d_loss)
        generator_losses.append(avg_g_loss)
        history_rows.append(
            {
                "epoch": epoch,
                "generator_loss": round(avg_g_loss, 6),
                "discriminator_loss": round(avg_d_loss, 6),
            }
        )

        print(
            f"Сводка по эпохе [{epoch}/{config.epochs}]: "
            f"средние потери D={avg_d_loss:.4f}, средние потери G={avg_g_loss:.4f}, "
            f"время={epoch_duration:.1f} с"
        )

        should_save_sample = (
            epoch == 1
            or epoch % config.sample_interval == 0
            or epoch == config.epochs
        )
        if should_save_sample:
            sample_path = config.samples_dir / f"sample_epoch_{epoch:03d}.png"
            save_sample_grid(generator, fixed_noise, sample_path, device=device)
            sample_records.append((epoch, sample_path.name))
            print(f"Сетка примеров сохранена в: {sample_path}")

    save_training_artifacts(
        generator=generator,
        discriminator=discriminator,
        generator_losses=generator_losses,
        discriminator_losses=discriminator_losses,
        history_rows=history_rows,
        sample_records=sample_records,
        fixed_noise=fixed_noise,
        device=device,
        total_training_time=perf_counter() - training_start_time,
    )


def main() -> None:
    train()


if __name__ == "__main__":
    main()
