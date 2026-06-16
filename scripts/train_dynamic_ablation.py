from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from dynamic_ablation_models import build_dynamic_ablation_model
from dynamic_data import (
    DynamicBatchCollator,
    DynamicIndexDataset,
    create_mock_dynamic_bundle,
    load_dynamic_feature_bundle,
    load_dynamic_feature_bundle_from_raw_states,
    split_dynamic_indices,
)
from losses import ModulationRegularizationLoss, TemporalContrastiveLoss
from utils import CSVMetricLogger, create_run_dir, resolve_device, set_seed


class DynamicAblationLoss(torch.nn.Module):
    def __init__(
        self,
        initial_temperature: float,
        regularization_weight: float,
        disable_temporal_loss: bool = False,
        disable_reg_loss: bool = False,
    ) -> None:
        super().__init__()
        self.disable_temporal_loss = disable_temporal_loss
        self.disable_reg_loss = disable_reg_loss
        self.regularization_weight = regularization_weight
        self.temporal = TemporalContrastiveLoss(initial_temperature=initial_temperature)
        self.regularization = ModulationRegularizationLoss()

    def forward(
        self,
        projected_embeddings: torch.Tensor,
        positive_mask: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        zero = projected_embeddings.sum() * 0.0
        temporal_loss = zero if self.disable_temporal_loss else self.temporal(projected_embeddings, positive_mask)
        reg_loss = zero if self.disable_reg_loss else self.regularization(gamma, beta)
        total = temporal_loss + self.regularization_weight * reg_loss
        return {
            "loss": total,
            "temporal_loss": temporal_loss,
            "regularization_loss": reg_loss,
        }

    @property
    def temperature(self) -> float:
        return self.temporal.temperature


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train dynamic ablation variants for StaDA.")
    parser.add_argument("--static-embeddings", type=str, default="")
    parser.add_argument("--static-cell-ids", type=str, default="")
    parser.add_argument("--dynamic-windows", type=str, default="")
    parser.add_argument("--time-context", type=str, default="")
    parser.add_argument("--sample-index", type=str, default="")
    parser.add_argument("--positive-mask", type=str, default="")
    parser.add_argument("--current-state", type=str, default="")
    parser.add_argument("--raw-dynamic-dir", type=str, default="")
    parser.add_argument("--granularity", choices=["hourly", "daily", "monthly"], default="hourly")
    parser.add_argument("--node-index", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--static-dim", type=int, default=128)
    parser.add_argument("--dynamic-dim", type=int, default=4)
    parser.add_argument("--time-dim", type=int, default=16)
    parser.add_argument("--window-length", type=int, default=12)
    parser.add_argument("--sequence-hidden-dim", type=int, default=128)
    parser.add_argument("--time-hidden-dim", type=int, default=64)
    parser.add_argument("--condition-dim", type=int, default=128)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--film-hidden-dim", type=int, default=128)
    parser.add_argument("--gru-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--regularization-weight", type=float, default=0.1)
    parser.add_argument("--nearby-radius", type=int, default=3)
    parser.add_argument("--similarity-threshold", type=float, default=0.8)
    parser.add_argument("--periodic-offsets", type=int, nargs="*", default=[-24, 24])
    parser.add_argument("--adaptation", choices=["film", "static_only", "concat", "add", "gated"], default="film")
    parser.add_argument("--disable-time-context", action="store_true")
    parser.add_argument("--disable-dynamic-history", action="store_true")
    parser.add_argument("--disable-temporal-loss", action="store_true")
    parser.add_argument("--disable-reg-loss", action="store_true")
    parser.add_argument("--use-mock-data", action="store_true")
    parser.add_argument("--mock-samples", type=int, default=256)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    return parser


def move_batch_to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    moved: Dict[str, object] = {
        "cell_ids": batch["cell_ids"],
        "global_indices": batch["global_indices"],
    }
    for key in ["static_embedding", "dynamic_window", "time_context", "positive_mask", "time_index"]:
        moved[key] = batch[key].to(device)
    return moved


def resolve_raw_dynamic_paths(raw_dir: Path, granularity: str) -> tuple[Path, Path, Path, Path]:
    if granularity == "hourly":
        return (
            raw_dir / "hourly_dynamic_state.npy",
            raw_dir / "cell_ids.npy",
            raw_dir / "time_context.npy",
            raw_dir / "time_index.csv",
        )
    if granularity == "daily":
        return (
            raw_dir / "daily_dynamic_state.npy",
            raw_dir / "cell_ids.npy",
            raw_dir / "day_context.npy",
            raw_dir / "date_index.csv",
        )
    return (
        raw_dir / "monthly_dynamic_state.npy",
        raw_dir / "cell_ids.npy",
        raw_dir / "month_context.npy",
        raw_dir / "month_index.csv",
    )


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: DynamicAblationLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train: bool,
) -> Dict[str, float]:
    model.train(train)
    totals = {"loss": 0.0, "temporal_loss": 0.0, "regularization_loss": 0.0}
    steps = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        outputs = model(
            static_embedding=batch["static_embedding"],
            dynamic_window=batch["dynamic_window"],
            time_context=batch["time_context"],
            return_aux=True,
        )
        loss_dict = criterion(
            projected_embeddings=outputs["projected_representation"],
            positive_mask=batch["positive_mask"],
            gamma=outputs["gamma"],
            beta=outputs["beta"],
        )
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss_dict["loss"].backward()
            optimizer.step()
        for key in totals:
            totals[key] += float(loss_dict[key].detach().cpu())
        steps += 1
    if steps == 0:
        return totals
    return {key: value / steps for key, value in totals.items()}


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    criterion: DynamicAblationLoss,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    epoch: int,
    best_val_loss: float,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "loss_state_dict": criterion.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
        },
        path,
    )


def main() -> None:
    args = build_arg_parser().parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    if args.use_mock_data:
        bundle = create_mock_dynamic_bundle(
            sample_count=args.mock_samples,
            static_dim=args.static_dim,
            window_length=args.window_length,
            dynamic_dim=args.dynamic_dim,
            time_dim=args.time_dim,
            seed=args.seed,
        )
    else:
        required_static_paths = [args.static_embeddings, args.static_cell_ids]
        if any(not value for value in required_static_paths):
            raise ValueError("Real-data training requires --static-embeddings and --static-cell-ids.")
        if args.raw_dynamic_dir:
            state_path, state_cell_ids_path, context_path, time_index_path = resolve_raw_dynamic_paths(
                raw_dir=Path(args.raw_dynamic_dir),
                granularity=args.granularity,
            )
            bundle = load_dynamic_feature_bundle_from_raw_states(
                static_embeddings_path=Path(args.static_embeddings),
                static_cell_ids_path=Path(args.static_cell_ids),
                state_path=state_path,
                state_cell_ids_path=state_cell_ids_path,
                context_path=context_path,
                time_index_path=time_index_path,
                window_length=args.window_length,
                node_index_path=Path(args.node_index) if args.node_index else None,
                current_state_path=Path(args.current_state) if args.current_state else None,
                periodic_offsets=tuple(args.periodic_offsets),
                nearby_radius=args.nearby_radius,
                similarity_threshold=args.similarity_threshold,
                max_samples=args.max_samples,
            )
        else:
            required_paths = [args.dynamic_windows, args.time_context, args.sample_index]
            if any(not value for value in required_paths):
                raise ValueError(
                    "Real-data training requires either --raw-dynamic-dir or "
                    "--dynamic-windows, --time-context, and --sample-index."
                )
            bundle = load_dynamic_feature_bundle(
                static_embeddings_path=Path(args.static_embeddings),
                static_cell_ids_path=Path(args.static_cell_ids),
                dynamic_windows_path=Path(args.dynamic_windows),
                time_context_path=Path(args.time_context),
                sample_index_path=Path(args.sample_index),
                positive_mask_path=Path(args.positive_mask) if args.positive_mask else None,
                current_state_path=Path(args.current_state) if args.current_state else None,
                periodic_offsets=tuple(args.periodic_offsets),
                nearby_radius=args.nearby_radius,
                similarity_threshold=args.similarity_threshold,
                max_samples=args.max_samples,
            )
        args.static_dim = int(bundle.static_embeddings.shape[1])
        args.dynamic_dim = int(bundle.dynamic_windows.shape[2])
        args.time_dim = int(bundle.time_context.shape[1])
        args.window_length = int(bundle.dynamic_windows.shape[1])

    train_indices, val_indices = split_dynamic_indices(bundle.num_samples, args.val_ratio, args.seed)
    collator = DynamicBatchCollator(bundle)
    train_loader = DataLoader(
        DynamicIndexDataset(train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        DynamicIndexDataset(val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
    )

    model = build_dynamic_ablation_model(vars(args)).to(device)
    criterion = DynamicAblationLoss(
        initial_temperature=args.temperature,
        regularization_weight=args.regularization_weight,
        disable_temporal_loss=args.disable_temporal_loss,
        disable_reg_loss=args.disable_reg_loss,
    ).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    output_dir = Path(args.output_dir) if args.output_dir else create_run_dir(Path(__file__).resolve().parent / "outputs", prefix="dynamic_ablation")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = CSVMetricLogger(
        path=output_dir / "train_log.csv",
        fieldnames=[
            "epoch",
            "train_loss",
            "train_temporal_loss",
            "train_regularization_loss",
            "val_loss",
            "val_temporal_loss",
            "val_regularization_loss",
            "temperature",
            "epoch_seconds",
            "adaptation",
            "disable_time_context",
            "disable_dynamic_history",
            "disable_temporal_loss",
            "disable_reg_loss",
        ],
    )

    best_val_loss = float("inf")
    best_model_state = None
    no_improve_epochs = 0

    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        epoch_seconds = time.perf_counter() - epoch_start

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_model_state = copy.deepcopy(model.state_dict())
            no_improve_epochs = 0
            save_checkpoint(output_dir / "best.ckpt", model, criterion, optimizer, args, epoch + 1, best_val_loss)
        else:
            no_improve_epochs += 1
        save_checkpoint(output_dir / "last.ckpt", model, criterion, optimizer, args, epoch + 1, best_val_loss)

        logger.log(
            {
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "train_temporal_loss": train_metrics["temporal_loss"],
                "train_regularization_loss": train_metrics["regularization_loss"],
                "val_loss": val_metrics["loss"],
                "val_temporal_loss": val_metrics["temporal_loss"],
                "val_regularization_loss": val_metrics["regularization_loss"],
                "temperature": criterion.temperature,
                "epoch_seconds": epoch_seconds,
                "adaptation": args.adaptation,
                "disable_time_context": args.disable_time_context,
                "disable_dynamic_history": args.disable_dynamic_history,
                "disable_temporal_loss": args.disable_temporal_loss,
                "disable_reg_loss": args.disable_reg_loss,
            }
        )
        print(
            f"Epoch {epoch + 1:03d} | {args.adaptation} | "
            f"train_loss={train_metrics['loss']:.6f} | val_loss={val_metrics['loss']:.6f} | "
            f"temperature={criterion.temperature:.6f} | time={epoch_seconds:.2f}s",
            flush=True,
        )
        if args.early_stopping_patience > 0 and no_improve_epochs >= args.early_stopping_patience:
            print(f"Early stopping triggered after {no_improve_epochs} non-improving epochs.", flush=True)
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    save_checkpoint(output_dir / "final.ckpt", model, criterion, optimizer, args, args.epochs, best_val_loss)
    print(f"Dynamic ablation training finished. Outputs saved to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
