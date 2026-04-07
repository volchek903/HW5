from __future__ import annotations

import argparse
from pathlib import Path

import torch

import config
from models import Generator
from utils import ensure_directories, save_sample_grid, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate new images using the trained GAN generator."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=config.generator_checkpoint,
        help="Path to the saved generator weights.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=config.num_sample_images,
        help="Number of images to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.random_seed,
        help="Random seed used to generate the latent noise.",
    )
    parser.add_argument(
        "--grid-cols",
        type=int,
        default=4,
        help="Number of columns in the saved image grid.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.generated_dir / "generated_samples.png",
        help="Path for the resulting image grid.",
    )
    return parser.parse_args()


def load_generator(checkpoint_path: Path, device: torch.device) -> Generator:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Run train_gan.py before generating samples."
        )

    generator = Generator(config.latent_dim).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(state_dict)
    generator.eval()
    return generator


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be a positive integer.")
    if args.grid_cols <= 0:
        raise ValueError("--grid-cols must be a positive integer.")

    set_seed(args.seed)

    ensure_directories([config.generated_dir, args.output.parent])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = load_generator(args.checkpoint, device)

    noise = torch.randn(args.num_samples, config.latent_dim, device=device)
    nrow = min(args.grid_cols, args.num_samples)
    save_sample_grid(generator, noise, args.output, device=device, nrow=nrow)

    print(f"Используемое устройство: {device}")
    print(f"Случайное зерно: {args.seed}")
    print(f"Сетка сгенерированных изображений сохранена в: {args.output}")


if __name__ == "__main__":
    main()
