from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dynamic_ablation_models import build_dynamic_ablation_model
from dynamic_data import (
    DynamicBatchCollator,
    DynamicIndexDataset,
    create_mock_dynamic_bundle,
    load_dynamic_feature_bundle,
    load_dynamic_feature_bundle_from_raw_states,
)
from utils import resolve_device, set_seed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export dynamic embeddings from a dynamic ablation checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
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
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--export-contrastive", action="store_true")
    parser.add_argument("--use-mock-data", action="store_true")
    parser.add_argument("--mock-samples", type=int, default=128)
    return parser


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


def save_embedding_csv(
    output_path: Path,
    cell_ids: np.ndarray,
    time_index: np.ndarray,
    embeddings: np.ndarray,
) -> None:
    header = ["cell_id", "time_index"] + [f"feat_{idx}" for idx in range(embeddings.shape[1])]
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for cell_id, t_idx, vector in zip(cell_ids.tolist(), time_index.tolist(), embeddings):
            writer.writerow([cell_id, int(t_idx), *vector.tolist()])


def main() -> None:
    args = build_arg_parser().parse_args()
    set_seed(args.seed)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    train_args = checkpoint["args"]
    device = resolve_device(args.device if args.device != "auto" else str(train_args.get("device", "auto")))

    if args.use_mock_data:
        bundle = create_mock_dynamic_bundle(
            sample_count=args.mock_samples,
            static_dim=int(train_args["static_dim"]),
            window_length=int(train_args["window_length"]),
            dynamic_dim=int(train_args["dynamic_dim"]),
            time_dim=int(train_args["time_dim"]),
            seed=args.seed,
        )
    else:
        required_static_paths = [args.static_embeddings, args.static_cell_ids]
        if any(not value for value in required_static_paths):
            raise ValueError("Export requires --static-embeddings and --static-cell-ids.")
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
                window_length=int(train_args["window_length"]),
                node_index_path=Path(args.node_index) if args.node_index else None,
                current_state_path=Path(args.current_state) if args.current_state else None,
                periodic_offsets=tuple(train_args.get("periodic_offsets", [-24, 24])),
                nearby_radius=int(train_args.get("nearby_radius", 3)),
                similarity_threshold=float(train_args.get("similarity_threshold", 0.8)),
                max_samples=args.max_samples,
            )
        else:
            required_paths = [args.dynamic_windows, args.time_context, args.sample_index]
            if any(not value for value in required_paths):
                raise ValueError(
                    "Export requires either --raw-dynamic-dir or "
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
                periodic_offsets=tuple(train_args.get("periodic_offsets", [-24, 24])),
                nearby_radius=int(train_args.get("nearby_radius", 3)),
                similarity_threshold=float(train_args.get("similarity_threshold", 0.8)),
                max_samples=args.max_samples,
            )

    model = build_dynamic_ablation_model(train_args).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    collator = DynamicBatchCollator(bundle)
    loader = DataLoader(
        DynamicIndexDataset(np.arange(bundle.num_samples, dtype=np.int64)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )

    dynamic_embeddings = np.empty((bundle.num_samples, int(train_args["static_dim"])), dtype=np.float32)
    contrastive_embeddings = np.empty((bundle.num_samples, int(train_args["projection_dim"])), dtype=np.float32)

    with torch.no_grad():
        for batch in loader:
            outputs = model(
                static_embedding=batch["static_embedding"].to(device),
                dynamic_window=batch["dynamic_window"].to(device),
                time_context=batch["time_context"].to(device),
                return_aux=True,
            )
            global_indices = batch["global_indices"].numpy()
            dynamic_embeddings[global_indices] = outputs["dynamic_representation"].cpu().numpy()
            contrastive_embeddings[global_indices] = outputs["projected_representation"].cpu().numpy()

    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent / "dynamic_exports"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "dynamic_embeddings.npy", dynamic_embeddings)
    save_embedding_csv(output_dir / "dynamic_embeddings.csv", bundle.cell_ids, bundle.time_index, dynamic_embeddings)
    print(f"Saved dynamic embeddings to: {output_dir / 'dynamic_embeddings.npy'}", flush=True)
    print(f"Saved dynamic CSV to: {output_dir / 'dynamic_embeddings.csv'}", flush=True)

    if args.export_contrastive:
        np.save(output_dir / "contrastive_embeddings.npy", contrastive_embeddings)
        print(f"Saved contrastive embeddings to: {output_dir / 'contrastive_embeddings.npy'}", flush=True)


if __name__ == "__main__":
    main()
