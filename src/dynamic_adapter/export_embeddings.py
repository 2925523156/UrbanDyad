from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import (
    DailyDynamicFeatureBundle,
    DynamicBatchCollator,
    DynamicFeatureBundle,
    DynamicIndexDataset,
    HourlyDynamicFeatureBundle,
    MonthlyDynamicFeatureBundle,
    create_mock_dynamic_bundle,
    load_daily_dynamic_feature_bundle,
    load_dynamic_feature_bundle,
    load_hourly_dynamic_feature_bundle,
    load_monthly_dynamic_feature_bundle,
)
from models import DynamicRegionRepresentationModel
from utils import resolve_device, set_seed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export dynamic embeddings from a trained dynamic encoder.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--static-embeddings", type=str, default="")
    parser.add_argument("--static-district-ids", type=str, default="")
    parser.add_argument("--static-region-ids", type=str, default="")
    parser.add_argument("--static-cell-ids", type=str, default="")
    parser.add_argument("--dynamic-windows", type=str, default="")
    parser.add_argument("--hourly-state", type=str, default="")
    parser.add_argument("--daily-state", type=str, default="")
    parser.add_argument("--monthly-state", type=str, default="")
    parser.add_argument("--district-ids", type=str, default="")
    parser.add_argument("--region-ids", type=str, default="")
    parser.add_argument("--cell-ids", type=str, default="")
    parser.add_argument("--time-context", type=str, default="")
    parser.add_argument("--day-context", type=str, default="")
    parser.add_argument("--month-context", type=str, default="")
    parser.add_argument("--sample-index", type=str, default="")
    parser.add_argument("--time-index", type=str, default="")
    parser.add_argument("--date-index", type=str, default="")
    parser.add_argument("--month-index", type=str, default="")
    parser.add_argument("--positive-mask", type=str, default="")
    parser.add_argument("--current-state", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--export-contrastive", action="store_true")
    parser.add_argument("--use-mock-data", action="store_true")
    parser.add_argument("--mock-samples", type=int, default=128)
    return parser


def save_embedding_csv(
    output_path: Path,
    region_ids: np.ndarray,
    time_index: np.ndarray,
    embeddings: np.ndarray,
) -> None:
    header = ["region_id", "time_index"] + [f"feat_{idx}" for idx in range(embeddings.shape[1])]
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for region_id, t_idx, vector in zip(region_ids.tolist(), time_index.tolist(), embeddings):
            writer.writerow([region_id, int(t_idx), *vector.tolist()])


def build_export_index(
    bundle: DynamicFeatureBundle | HourlyDynamicFeatureBundle | DailyDynamicFeatureBundle | MonthlyDynamicFeatureBundle,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(bundle, (HourlyDynamicFeatureBundle, DailyDynamicFeatureBundle, MonthlyDynamicFeatureBundle)):
        global_indices = np.arange(bundle.num_samples, dtype=np.int64)
        region_indices = np.mod(global_indices, bundle.num_cells).astype(np.int64)
        target_times = (global_indices // bundle.num_cells + bundle.window_length).astype(np.int64)
        return bundle.cell_ids[region_indices], target_times
    return bundle.cell_ids, bundle.time_index


def build_model_from_args(args: dict[str, object]) -> DynamicRegionRepresentationModel:
    return DynamicRegionRepresentationModel(
        static_dim=int(args["static_dim"]),
        dynamic_dim=int(args["dynamic_dim"]),
        time_dim=int(args["time_dim"]),
        sequence_hidden_dim=int(args["sequence_hidden_dim"]),
        time_hidden_dim=int(args["time_hidden_dim"]),
        condition_dim=int(args["condition_dim"]),
        projection_dim=int(args["projection_dim"]),
        film_hidden_dim=int(args["film_hidden_dim"]),
        gru_layers=int(args["gru_layers"]),
        dropout=float(args["dropout"]),
    )


def main() -> None:
    args = build_arg_parser().parse_args()
    set_seed(args.seed)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    train_args = checkpoint["args"]
    device = resolve_device(args.device if args.device != "auto" else str(train_args.get("device", "auto")))
    static_region_ids_path = args.static_district_ids or args.static_region_ids or args.static_cell_ids
    hourly_region_ids_path = args.district_ids or args.region_ids or args.cell_ids

    if args.use_mock_data:
        bundle = create_mock_dynamic_bundle(
            sample_count=args.mock_samples,
            static_dim=int(train_args["static_dim"]),
            window_length=int(train_args["window_length"]),
            dynamic_dim=int(train_args["dynamic_dim"]),
            time_dim=int(train_args["time_dim"]),
            seed=args.seed,
        )
    elif args.hourly_state:
        required_paths = [
            args.static_embeddings,
            static_region_ids_path,
            args.hourly_state,
            hourly_region_ids_path,
            args.time_context,
            args.time_index,
        ]
        if any(not value for value in required_paths):
            raise ValueError(
                "Hourly-state export requires --static-embeddings, --static-district-ids, "
                "--hourly-state, --district-ids, --time-context, and --time-index."
            )
        bundle = load_hourly_dynamic_feature_bundle(
            static_embeddings_path=Path(args.static_embeddings),
            static_region_ids_path=Path(static_region_ids_path),
            hourly_state_path=Path(args.hourly_state),
            hourly_region_ids_path=Path(hourly_region_ids_path),
            time_context_path=Path(args.time_context),
            time_index_path=Path(args.time_index),
            window_length=int(train_args.get("window_length", 24)),
            periodic_offsets=tuple(train_args.get("periodic_offsets", [-24, 24])),
            nearby_radius=int(train_args.get("nearby_radius", 3)),
            similarity_threshold=float(train_args.get("similarity_threshold", 0.8)),
            max_samples=args.max_samples,
        )
    elif args.daily_state:
        required_paths = [
            args.static_embeddings,
            static_region_ids_path,
            args.daily_state,
            hourly_region_ids_path,
            args.day_context,
            args.date_index,
        ]
        if any(not value for value in required_paths):
            raise ValueError(
                "Daily-state export requires --static-embeddings, one of "
                "--static-region-ids/--static-district-ids/--static-cell-ids, "
                "--daily-state, one of --region-ids/--district-ids/--cell-ids, "
                "--day-context, and --date-index."
            )
        bundle = load_daily_dynamic_feature_bundle(
            static_embeddings_path=Path(args.static_embeddings),
            static_region_ids_path=Path(static_region_ids_path),
            daily_state_path=Path(args.daily_state),
            daily_region_ids_path=Path(hourly_region_ids_path),
            day_context_path=Path(args.day_context),
            date_index_path=Path(args.date_index),
            window_length=int(train_args.get("window_length", 7)),
            periodic_offsets=tuple(train_args.get("periodic_offsets", [-7, 7])),
            nearby_radius=int(train_args.get("nearby_radius", 3)),
            similarity_threshold=float(train_args.get("similarity_threshold", 0.8)),
            max_samples=args.max_samples,
            same_day_of_month_positives=bool(train_args.get("same_day_of_month_positives", False)),
            same_day_of_week_positives=bool(train_args.get("same_day_of_week_positives", False)),
        )
    elif args.monthly_state:
        required_paths = [
            args.static_embeddings,
            static_region_ids_path,
            args.monthly_state,
            hourly_region_ids_path,
            args.month_context,
            args.month_index,
        ]
        if any(not value for value in required_paths):
            raise ValueError(
                "Monthly-state export requires --static-embeddings, one of "
                "--static-region-ids/--static-district-ids/--static-cell-ids, "
                "--monthly-state, one of --region-ids/--district-ids/--cell-ids, "
                "--month-context, and --month-index."
            )
        bundle = load_monthly_dynamic_feature_bundle(
            static_embeddings_path=Path(args.static_embeddings),
            static_region_ids_path=Path(static_region_ids_path),
            monthly_state_path=Path(args.monthly_state),
            monthly_region_ids_path=Path(hourly_region_ids_path),
            month_context_path=Path(args.month_context),
            month_index_path=Path(args.month_index),
            window_length=int(train_args.get("window_length", 6)),
            periodic_offsets=tuple(train_args.get("periodic_offsets", [])),
            nearby_radius=int(train_args.get("nearby_radius", 2)),
            similarity_threshold=float(train_args.get("similarity_threshold", 0.7)),
            max_samples=args.max_samples,
        )
    else:
        required_paths = [
            args.static_embeddings,
            static_region_ids_path,
            args.dynamic_windows,
            args.time_context,
            args.sample_index,
        ]
        if any(not value for value in required_paths):
            raise ValueError(
                "Export requires --static-embeddings, --static-district-ids/--static-region-ids, "
                "--dynamic-windows, --time-context, and --sample-index."
            )
        bundle = load_dynamic_feature_bundle(
            static_embeddings_path=Path(args.static_embeddings),
            static_region_ids_path=Path(static_region_ids_path),
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

    model = build_model_from_args(train_args).to(device)
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
            static_embedding = batch["static_embedding"].to(device)
            dynamic_window = batch["dynamic_window"].to(device)
            time_context = batch["time_context"].to(device)
            outputs = model(
                static_embedding=static_embedding,
                dynamic_window=dynamic_window,
                time_context=time_context,
                return_aux=True,
            )
            global_indices = batch["global_indices"].numpy()
            dynamic_embeddings[global_indices] = outputs["dynamic_representation"].cpu().numpy()
            contrastive_embeddings[global_indices] = outputs["projected_representation"].cpu().numpy()

    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent / "dynamic_exports"
    output_dir.mkdir(parents=True, exist_ok=True)

    dynamic_npy = output_dir / "dynamic_embeddings.npy"
    dynamic_csv = output_dir / "dynamic_embeddings.csv"
    np.save(dynamic_npy, dynamic_embeddings)
    export_region_ids, export_time_index = build_export_index(bundle)
    save_embedding_csv(dynamic_csv, export_region_ids, export_time_index, dynamic_embeddings)

    print(f"Saved dynamic embeddings to: {dynamic_npy}")
    print(f"Saved dynamic CSV to: {dynamic_csv}")

    if args.export_contrastive:
        contrastive_npy = output_dir / "contrastive_embeddings.npy"
        np.save(contrastive_npy, contrastive_embeddings)
        print(f"Saved contrastive embeddings to: {contrastive_npy}")


if __name__ == "__main__":
    main()
