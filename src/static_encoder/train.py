from __future__ import annotations

import argparse
import contextlib
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from config import ExperimentConfig, config_to_dict, default_config, save_config
from data import (
    BatchCollator,
    IndexDataset,
    load_feature_bundle,
    sample_dual_view_keep_masks,
    split_indices,
)
from graph_utils import load_csr_graph
from losses import SymmetricInfoNCELoss
from models import TriModalFusionModel
from utils import CSVMetricLogger, create_run_dir, resolve_device, set_seed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the fusion encoder."
    )
    parser.add_argument(
        "--disable-poi",
        action="store_true",
        help="Disable the POI modality and train with RS+OSM only.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--early-stopping-min-delta", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--run-dir", type=str, default="")
    parser.add_argument("--graph-cache-dir", type=str, default="")
    parser.add_argument("--num-walks", type=int, default=None)
    parser.add_argument("--walk-length", type=int, default=None)
    parser.add_argument("--restart-prob", type=float, default=None)
    parser.add_argument("--poi-node-dim", type=int, default=None)
    parser.add_argument("--fusion-dim", type=int, default=None)
    parser.add_argument("--contrastive-dim", type=int, default=None)
    parser.add_argument("--transformer-depth", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--ff-dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    return parser


def apply_arg_overrides(
    config: ExperimentConfig,
    args: argparse.Namespace,
) -> ExperimentConfig:
    if args.disable_poi:
        config.model.use_poi = False
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.learning_rate is not None:
        config.training.learning_rate = args.learning_rate
    if args.weight_decay is not None:
        config.training.weight_decay = args.weight_decay
    if args.val_ratio is not None:
        config.training.val_ratio = args.val_ratio
    if args.early_stopping_patience is not None:
        config.training.early_stopping_patience = args.early_stopping_patience
    if args.early_stopping_min_delta is not None:
        config.training.early_stopping_min_delta = args.early_stopping_min_delta
    if args.num_workers is not None:
        config.training.num_workers = args.num_workers
    if args.seed is not None:
        config.training.seed = args.seed
    if args.device is not None:
        config.training.device = args.device
    if args.max_samples is not None:
        config.training.max_samples = args.max_samples
    if args.graph_cache_dir:
        config.paths.poi_graph_cache_dir = Path(args.graph_cache_dir)
    if args.num_walks is not None:
        config.random_walk.num_walks = args.num_walks
    if args.walk_length is not None:
        config.random_walk.walk_length = args.walk_length
    if args.restart_prob is not None:
        config.random_walk.restart_prob = args.restart_prob
    if args.poi_node_dim is not None:
        config.model.poi_node_dim = args.poi_node_dim
    if args.fusion_dim is not None:
        config.model.fusion_dim = args.fusion_dim
    if args.contrastive_dim is not None:
        config.model.contrastive_dim = args.contrastive_dim
    if args.transformer_depth is not None:
        config.model.transformer_depth = args.transformer_depth
    if args.num_heads is not None:
        config.model.num_heads = args.num_heads
    if args.ff_dim is not None:
        config.model.ff_dim = args.ff_dim
    if args.dropout is not None:
        config.model.dropout = args.dropout
    return config


def create_model(config: ExperimentConfig, num_poi_nodes: int) -> TriModalFusionModel:
    return TriModalFusionModel(
        num_poi_nodes=num_poi_nodes,
        use_poi=config.model.use_poi,
        poi_node_dim=config.model.poi_node_dim,
        fusion_dim=config.model.fusion_dim,
        contrastive_dim=config.model.contrastive_dim,
        transformer_depth=config.model.transformer_depth,
        num_heads=config.model.num_heads,
        ff_dim=config.model.ff_dim,
        dropout=config.model.dropout,
    )


def move_batch_to_device(
    batch: Dict[str, object],
    device: torch.device,
) -> Dict[str, object]:
    moved: Dict[str, object] = {"cell_ids": batch["cell_ids"]}
    for key in [
        "global_indices",
        "rs_features",
        "osm_features",
        "street_features",
        "poi_local_ids",
        "has_poi",
        "has_street",
    ]:
        moved[key] = batch[key].to(device)
    return moved


def run_epoch(
    model: TriModalFusionModel,
    loader: DataLoader,
    graph,
    loss_fn: SymmetricInfoNCELoss,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    config: ExperimentConfig,
    device: torch.device,
    train: bool,
    epoch_seed: int,
) -> float:
    model.train(train)
    epoch_losses = []
    rng = np.random.default_rng(epoch_seed)

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        if not config.model.use_poi:
            batch["has_poi"] = torch.zeros_like(batch["has_poi"], dtype=torch.bool)
        keep0, keep1 = sample_dual_view_keep_masks(
            has_poi=batch["has_poi"].detach().cpu().numpy(),
            has_street=batch["has_street"].detach().cpu().numpy(),
            rng=rng,
        )
        keep0 = keep0.to(device)
        keep1 = keep1.to(device)

        autocast_enabled = bool(config.training.amp and device.type == "cuda")
        context = (
            torch.cuda.amp.autocast(enabled=True)
            if autocast_enabled
            else contextlib.nullcontext()
        )

        with torch.set_grad_enabled(train):
            with context:
                tokens = model.encode_modalities(
                    rs_features=batch["rs_features"],
                    osm_features=batch["osm_features"],
                    street_features=batch["street_features"],
                    poi_local_ids=batch["poi_local_ids"],
                    has_poi=batch["has_poi"],
                    has_street=batch["has_street"],
                    graph=graph,
                    num_walks=config.random_walk.num_walks,
                    walk_length=config.random_walk.walk_length,
                    restart_prob=config.random_walk.restart_prob,
                    rng=rng,
                )
                z0 = model.fuse_tokens(tokens, keep0)
                z1 = model.fuse_tokens(tokens, keep1)
                loss = loss_fn(z0, z1)

            if train:
                optimizer.zero_grad(set_to_none=True)
                if autocast_enabled:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        epoch_losses.append(float(loss.detach().cpu()))

    return float(np.mean(epoch_losses)) if epoch_losses else 0.0


def format_seconds(seconds: float) -> str:
    total = int(max(seconds, 0))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def save_checkpoint(
    path: Path,
    model: TriModalFusionModel,
    loss_fn: SymmetricInfoNCELoss,
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    epoch: int,
    best_val_loss: float,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "loss_state_dict": loss_fn.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config_to_dict(config),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
    }
    torch.save(payload, path)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    config = apply_arg_overrides(default_config(), args)
    set_seed(config.training.seed)
    device = resolve_device(config.training.device)

    run_dir = (
        Path(args.run_dir)
        if args.run_dir
        else create_run_dir(config.paths.output_root, prefix="train")
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, run_dir / "config.json")

    if config.model.use_poi:
        graph = load_csr_graph(config.paths.poi_graph_cache_dir)
        num_poi_nodes = graph.num_nodes
    else:
        graph = None
        num_poi_nodes = 1
    bundle = load_feature_bundle(
        paths=config.paths,
        max_samples=config.training.max_samples,
    )

    train_indices, val_indices = split_indices(
        num_samples=bundle.num_samples,
        val_ratio=config.training.val_ratio,
        seed=config.training.seed,
    )
    collator = BatchCollator(bundle)
    train_loader = DataLoader(
        IndexDataset(train_indices),
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collator,
    )
    val_loader = DataLoader(
        IndexDataset(val_indices),
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collator,
    )

    model = create_model(config=config, num_poi_nodes=num_poi_nodes).to(device)
    loss_fn = SymmetricInfoNCELoss().to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scaler = GradScaler(enabled=bool(config.training.amp and device.type == "cuda"))

    logger = CSVMetricLogger(
        path=run_dir / "train_log.csv",
        fieldnames=[
            "epoch",
            "train_loss",
            "val_loss",
            "temperature",
            "best_val_loss",
            "no_improve_epochs",
            "train_seconds",
            "val_seconds",
            "epoch_seconds",
            "eta_seconds",
        ],
    )

    best_val_loss = float("inf")
    no_improve_epochs = 0
    for epoch in range(config.training.epochs):
        epoch_start = time.perf_counter()
        print(
            f"Starting epoch {epoch + 1:03d}/{config.training.epochs:03d} "
            f"(use_poi={config.model.use_poi})...",
            flush=True,
        )
        train_start = time.perf_counter()
        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            graph=graph,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scaler=scaler,
            config=config,
            device=device,
            train=True,
            epoch_seed=config.training.seed + epoch,
        )
        train_seconds = time.perf_counter() - train_start
        val_start = time.perf_counter()
        val_loss = run_epoch(
            model=model,
            loader=val_loader,
            graph=graph,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scaler=scaler,
            config=config,
            device=device,
            train=False,
            epoch_seed=config.training.seed + 10_000 + epoch,
        )
        val_seconds = time.perf_counter() - val_start
        epoch_seconds = time.perf_counter() - epoch_start
        remaining_epochs = config.training.epochs - (epoch + 1)
        eta_seconds = remaining_epochs * epoch_seconds

        improved = (
            best_val_loss == float("inf")
            or val_loss < (best_val_loss - config.training.early_stopping_min_delta)
        )
        if improved:
            best_val_loss = val_loss
            no_improve_epochs = 0
            save_checkpoint(
                path=run_dir / "best.ckpt",
                model=model,
                loss_fn=loss_fn,
                optimizer=optimizer,
                config=config,
                epoch=epoch + 1,
                best_val_loss=best_val_loss,
            )
        else:
            no_improve_epochs += 1

        logger.log(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "temperature": loss_fn.temperature,
                "best_val_loss": best_val_loss,
                "no_improve_epochs": no_improve_epochs,
                "train_seconds": train_seconds,
                "val_seconds": val_seconds,
                "epoch_seconds": epoch_seconds,
                "eta_seconds": eta_seconds,
            }
        )
        print(
            f"Epoch {epoch + 1:03d} | train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | temperature={loss_fn.temperature:.6f} | "
            f"best_val_loss={best_val_loss:.6f} | "
            f"no_improve={no_improve_epochs} | "
            f"train_time={format_seconds(train_seconds)} | "
            f"val_time={format_seconds(val_seconds)} | "
            f"epoch_time={format_seconds(epoch_seconds)} | "
            f"eta={format_seconds(eta_seconds)}",
            flush=True,
        )

        save_checkpoint(
            path=run_dir / "last.ckpt",
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            config=config,
            epoch=epoch + 1,
            best_val_loss=best_val_loss,
        )

        if (
            config.training.early_stopping_patience > 0
            and no_improve_epochs >= config.training.early_stopping_patience
        ):
            print(
                "Early stopping triggered | "
                f"patience={config.training.early_stopping_patience} | "
                f"min_delta={config.training.early_stopping_min_delta} | "
                f"best_val_loss={best_val_loss:.6f}",
                flush=True,
            )
            break

    print(f"Training finished. Outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()
