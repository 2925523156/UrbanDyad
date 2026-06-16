from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
from datetime import datetime
from typing import Dict

import torch
from torch.utils.data import DataLoader

from data import (
    DynamicBatchCollator,
    DynamicIndexDataset,
    create_mock_dynamic_bundle,
    load_monthly_dynamic_feature_bundle,
    split_dynamic_indices,
)
from losses import DynamicTotalLoss
from models import DynamicRegionRepresentationModel
from utils import CSVMetricLogger, create_run_dir, resolve_device, set_seed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the monthly dynamic region/cell representation model.")
    parser.add_argument("--static-embeddings", type=str, default="")
    parser.add_argument("--static-district-ids", type=str, default="")
    parser.add_argument("--static-region-ids", type=str, default="")
    parser.add_argument("--static-cell-ids", type=str, default="")
    parser.add_argument("--monthly-state", type=str, default="")
    parser.add_argument("--district-ids", type=str, default="")
    parser.add_argument("--region-ids", type=str, default="")
    parser.add_argument("--cell-ids", type=str, default="")
    parser.add_argument("--month-context", type=str, default="")
    parser.add_argument("--month-index", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--resume-from", type=str, default="")
    parser.add_argument("--no-output-timestamp", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--static-dim", type=int, default=128)
    parser.add_argument("--dynamic-dim", type=int, default=4)
    parser.add_argument("--time-dim", type=int, default=4)
    parser.add_argument("--window-length", type=int, default=3)
    parser.add_argument("--sequence-hidden-dim", type=int, default=128)
    parser.add_argument("--time-hidden-dim", type=int, default=64)
    parser.add_argument("--condition-dim", type=int, default=128)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--film-hidden-dim", type=int, default=128)
    parser.add_argument("--gru-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--regularization-weight", type=float, default=0.05)
    parser.add_argument("--nearby-radius", type=int, default=2)
    parser.add_argument("--similarity-threshold", type=float, default=0.7)
    parser.add_argument("--periodic-offsets", type=int, nargs="*", default=[])
    parser.add_argument("--use-mock-data", action="store_true")
    parser.add_argument("--mock-samples", type=int, default=256)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    return parser


def build_model(args: argparse.Namespace) -> DynamicRegionRepresentationModel:
    return DynamicRegionRepresentationModel(
        static_dim=args.static_dim,
        dynamic_dim=args.dynamic_dim,
        time_dim=args.time_dim,
        sequence_hidden_dim=args.sequence_hidden_dim,
        time_hidden_dim=args.time_hidden_dim,
        condition_dim=args.condition_dim,
        projection_dim=args.projection_dim,
        film_hidden_dim=args.film_hidden_dim,
        gru_layers=args.gru_layers,
        dropout=args.dropout,
    )


def move_batch_to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    moved: Dict[str, object] = {
        "region_ids": batch.get("region_ids", batch.get("cell_ids", [])),
        "global_indices": batch["global_indices"],
    }
    for key in ["static_embedding", "dynamic_window", "time_context", "positive_mask", "time_index"]:
        moved[key] = batch[key].to(device)
    return moved


def run_epoch(
    model: DynamicRegionRepresentationModel,
    loader: DataLoader,
    criterion: DynamicTotalLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train: bool,
) -> Dict[str, float]:
    model.train(train)
    total_loss = 0.0
    temporal_total = 0.0
    reg_total = 0.0
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
        loss = loss_dict["loss"]

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.detach().cpu())
        temporal_total += float(loss_dict["temporal_loss"].detach().cpu())
        reg_total += float(loss_dict["regularization_loss"].detach().cpu())
        steps += 1

    if steps == 0:
        return {"loss": 0.0, "temporal_loss": 0.0, "regularization_loss": 0.0}
    return {
        "loss": total_loss / steps,
        "temporal_loss": temporal_total / steps,
        "regularization_loss": reg_total / steps,
    }


def save_checkpoint(
    path: Path,
    model: DynamicRegionRepresentationModel,
    criterion: DynamicTotalLoss,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    epoch: int,
    best_val_loss: float,
    no_improve_epochs: int,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "loss_state_dict": criterion.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "no_improve_epochs": no_improve_epochs,
    }
    torch.save(payload, path)


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.resume_from:
        return Path(args.resume_from).resolve().parent
    monthly_region_ids_path = args.region_ids or args.district_ids or args.cell_ids
    s2_like_run = bool(
        args.cell_ids
        or (monthly_region_ids_path and "cell_ids" in Path(monthly_region_ids_path).name.lower())
        or (args.monthly_state and "monthly_dynamic_inputs_s2" in str(args.monthly_state))
    )
    default_prefix = "dynamic_train_monthly_s2_od" if s2_like_run else "dynamic_train_monthly"
    if not args.output_dir:
        return create_run_dir(Path(__file__).resolve().parent / "outputs", prefix=default_prefix)
    output_dir = Path(args.output_dir)
    if args.no_output_timestamp:
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_dir = output_dir.with_name(f"{output_dir.name}_{timestamp}")
    timestamped_dir.mkdir(parents=True, exist_ok=True)
    return timestamped_dir


def load_checkpoint(
    checkpoint_path: Path,
    model: DynamicRegionRepresentationModel,
    criterion: DynamicTotalLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion.load_state_dict(checkpoint["loss_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def main() -> None:
    args = build_arg_parser().parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    static_region_ids_path = args.static_region_ids or args.static_district_ids or args.static_cell_ids
    monthly_region_ids_path = args.region_ids or args.district_ids or args.cell_ids

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
        required_paths = [
            args.static_embeddings,
            static_region_ids_path,
            args.monthly_state,
            monthly_region_ids_path,
            args.month_context,
            args.month_index,
        ]
        if any(not value for value in required_paths):
            raise ValueError(
                "Monthly-state training requires --static-embeddings, one of "
                "--static-region-ids/--static-district-ids/--static-cell-ids, "
                "--monthly-state, one of --region-ids/--district-ids/--cell-ids, "
                "--month-context, and --month-index."
            )
        bundle = load_monthly_dynamic_feature_bundle(
            static_embeddings_path=Path(args.static_embeddings),
            static_region_ids_path=Path(static_region_ids_path),
            monthly_state_path=Path(args.monthly_state),
            monthly_region_ids_path=Path(monthly_region_ids_path),
            month_context_path=Path(args.month_context),
            month_index_path=Path(args.month_index),
            window_length=args.window_length,
            periodic_offsets=tuple(args.periodic_offsets),
            nearby_radius=args.nearby_radius,
            similarity_threshold=args.similarity_threshold,
            max_samples=args.max_samples,
        )
        args.static_dim = int(bundle.static_embeddings.shape[1])
        args.dynamic_dim = int(bundle.monthly_state.shape[2])
        args.time_dim = int(bundle.month_context.shape[1])

    train_indices, val_indices = split_dynamic_indices(
        num_samples=bundle.num_samples,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    collator = DynamicBatchCollator(bundle, augment_positive_samples=True)
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

    model = build_model(args).to(device)
    criterion = DynamicTotalLoss(
        initial_temperature=args.temperature,
        regularization_weight=args.regularization_weight,
    ).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    output_dir = resolve_output_dir(args)
    args.output_dir = str(output_dir)
    start_epoch = 0
    best_val_loss = float("inf")
    best_model_state = None
    no_improve_epochs = 0
    final_epoch = 0

    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        checkpoint = load_checkpoint(
            checkpoint_path=resume_path,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        start_epoch = int(checkpoint.get("epoch", 0))
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        no_improve_epochs = int(checkpoint.get("no_improve_epochs", 0))
        best_ckpt_path = output_dir / "best.ckpt"
        if best_ckpt_path.exists():
            best_checkpoint = torch.load(best_ckpt_path, map_location=device)
            best_model_state = copy.deepcopy(best_checkpoint["model_state_dict"])
        elif best_val_loss < float("inf"):
            best_model_state = copy.deepcopy(model.state_dict())

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
        ],
        append=bool(args.resume_from),
    )

    print(
        f"Monthly dynamic training samples={bundle.num_samples} "
        f"window_length={args.window_length} static_dim={args.static_dim} "
        f"dynamic_dim={args.dynamic_dim} time_dim={args.time_dim} "
        f"periodic_offsets={args.periodic_offsets} "
        f"dynamic_window_mode=history_plus_current_state "
        f"context_mode=month_context_only",
        flush=True,
    )
    if args.resume_from:
        print(
            f"Resuming monthly training from: {args.resume_from} | "
            f"start_epoch={start_epoch + 1} | best_val_loss={best_val_loss:.6f}",
            flush=True,
        )

    for epoch in range(start_epoch, args.epochs):
        final_epoch = epoch + 1
        epoch_start = time.perf_counter()
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            train=True,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                train=False,
            )
        epoch_seconds = time.perf_counter() - epoch_start

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_model_state = copy.deepcopy(model.state_dict())
            no_improve_epochs = 0
            save_checkpoint(
                path=output_dir / "best.ckpt",
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                args=args,
                epoch=epoch + 1,
                best_val_loss=best_val_loss,
                no_improve_epochs=no_improve_epochs,
            )
        else:
            no_improve_epochs += 1

        save_checkpoint(
            path=output_dir / "last.ckpt",
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            args=args,
            epoch=epoch + 1,
            best_val_loss=best_val_loss,
            no_improve_epochs=no_improve_epochs,
        )

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
            }
        )
        print(
            f"Epoch {epoch + 1:03d} | "
            f"train_loss={train_metrics['loss']:.6f} | "
            f"val_loss={val_metrics['loss']:.6f} | "
            f"temperature={criterion.temperature:.6f} | "
            f"time={epoch_seconds:.2f}s",
            flush=True,
        )

        if args.early_stopping_patience > 0 and no_improve_epochs >= args.early_stopping_patience:
            print(
                f"Early stopping triggered after {no_improve_epochs} non-improving epochs.",
                flush=True,
            )
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    save_checkpoint(
        path=output_dir / "final.ckpt",
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        args=args,
        epoch=final_epoch,
        best_val_loss=best_val_loss,
        no_improve_epochs=no_improve_epochs,
    )
    print(f"Monthly dynamic training finished. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
