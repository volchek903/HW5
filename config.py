from pathlib import Path

batch_size: int = 128
latent_dim: int = 100
learning_rate: float = 0.0002
epochs: int = 30
sample_interval: int = 5
log_interval: int = 50
random_seed: int = 42

image_size: int = 28
num_channels: int = 1
num_sample_images: int = 16
adam_betas: tuple[float, float] = (0.5, 0.999)

data_dir: Path = Path("data")
output_dir: Path = Path("outputs")
samples_dir: Path = output_dir / "samples"
checkpoints_dir: Path = output_dir / "checkpoints"
plots_dir: Path = output_dir / "plots"
generated_dir: Path = output_dir / "generated"

generator_checkpoint: Path = checkpoints_dir / "generator_final.pth"
discriminator_checkpoint: Path = checkpoints_dir / "discriminator_final.pth"
loss_plot_path: Path = plots_dir / "loss_curve.png"
progress_plot_path: Path = plots_dir / "training_progress.png"
history_path: Path = output_dir / "training_history.csv"
summary_path: Path = output_dir / "training_summary.json"
