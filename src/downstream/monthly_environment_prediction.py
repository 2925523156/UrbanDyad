from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


class TabularRegressionDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.features = torch.from_numpy(features.astype(np.float32, copy=False))
        self.labels = torch.from_numpy(labels.astype(np.float32, copy=False)).view(-1, 1)

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class GRURegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        head_hidden_dims: list[int],
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        layers: list[nn.Module] = []
        prev_dim = hidden_dim
        for hidden in head_hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden
        layers.append(nn.Linear(prev_dim, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(x)
        return self.head(hidden[-1])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict monthly environmental indicators from monthly dynamic embeddings.")
    parser.add_argument(
        "--input-csv",
        type=str,
        default=str(Path("data") / "sample" / "dynamic_embeddings_with_monthly_environment.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path("outputs") / "monthly_environment"),
    )
    parser.add_argument("--target-col", type=str, default="ndvi")
    parser.add_argument("--date-col", type=str, default="month")
    parser.add_argument("--region-col", type=str, default="region_id")
    parser.add_argument("--forecast-horizon", type=int, default=1)
    parser.add_argument("--embedding-history-length", type=int, default=6)
    parser.add_argument("--no-target-history", action="store_true")
    parser.add_argument("--target-history-only", action="store_true")
    parser.add_argument("--autoregressive-target-history", action="store_true")
    parser.add_argument(
        "--split-mode",
        type=str,
        choices=["date", "region", "rolling", "spatiotemporal"],
        default="date",
    )
    parser.add_argument("--test-date-ratio", type=float, default=0.2)
    parser.add_argument("--val-date-ratio", type=float, default=0.4)
    parser.add_argument("--rolling-min-train-months", type=int, default=3)
    parser.add_argument("--test-region-ratio", type=float, default=0.2)
    parser.add_argument("--val-region-ratio", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--model-type", type=str, choices=["gru", "mlp"], default="mlp")
    parser.add_argument("--gru-hidden-dim", type=int, default=128)
    parser.add_argument("--gru-layers", type=int, default=1)
    parser.add_argument("--hidden-dims", type=int, nargs="*", default=[128])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no-output-timestamp", action="store_true")
    parser.add_argument("--log1p-target", action="store_true")
    return parser


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def resolve_output_dir(output_dir: Path, add_timestamp: bool) -> Path:
    if add_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = output_dir.with_name(f"{output_dir.name}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def normalize_month(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip()
    if text.str.contains("-").any():
        return text.str.replace("-", "", regex=False)
    return text


def filter_all_null_regions(
    df: pd.DataFrame,
    region_col: str,
    target_col: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    raw_row_count = int(len(df))
    raw_region_count = int(df[region_col].astype(str).nunique())
    non_null_counts = df.groupby(region_col, sort=False)[target_col].count()
    dropped_regions = non_null_counts[non_null_counts.eq(0)].index.astype(str)
    filtered = df.loc[~df[region_col].astype(str).isin(dropped_regions)].copy()
    stats = {
        "raw_row_count": raw_row_count,
        "raw_region_count": raw_region_count,
        "dropped_all_null_regions": int(len(dropped_regions)),
        "remaining_regions_after_filter": int(filtered[region_col].astype(str).nunique()),
        "row_count_after_region_filter": int(len(filtered)),
    }
    print(
        "Region filter | "
        f"raw_regions={stats['raw_region_count']} "
        f"dropped_all_null_regions={stats['dropped_all_null_regions']} "
        f"remaining_regions={stats['remaining_regions_after_filter']} "
        f"raw_rows={stats['raw_row_count']} "
        f"rows_after_region_filter={stats['row_count_after_region_filter']}",
        flush=True,
    )
    return filtered, stats


def save_data_filter_summary(output_dir: Path, summary: dict[str, object]) -> None:
    with (output_dir / "data_filter_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def load_table(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    df = pd.read_csv(args.input_csv, dtype={args.region_col: str, args.date_col: str})
    feature_cols = [col for col in df.columns if col.startswith("feat_")]
    if not feature_cols:
        raise ValueError("No feature columns found. Expected columns named feat_0, feat_1, ...")

    required_cols = [args.region_col, "time_index", args.date_col, args.target_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df, filter_stats = filter_all_null_regions(df, args.region_col, args.target_col)
    df = df.dropna(subset=feature_cols + [args.target_col, args.date_col, "time_index"]).copy()
    filter_stats["row_count_after_dropna"] = int(len(df))
    filter_stats["region_count_after_dropna"] = int(df[args.region_col].astype(str).nunique())
    print(
        "Row filter | "
        f"rows_after_dropna={filter_stats['row_count_after_dropna']} "
        f"regions_after_dropna={filter_stats['region_count_after_dropna']}",
        flush=True,
    )
    df[args.date_col] = normalize_month(df[args.date_col])
    df["time_index"] = df["time_index"].astype(int)
    df = df.sort_values([args.region_col, "time_index"]).reset_index(drop=True)
    return df, feature_cols, filter_stats


def build_forecast_table(
    df: pd.DataFrame,
    feature_cols: list[str],
    region_col: str,
    date_col: str,
    target_col: str,
    forecast_horizon: int,
    embedding_history_length: int,
    include_target_history: bool,
    target_history_only: bool,
) -> tuple[pd.DataFrame, list[str]]:
    if forecast_horizon < 0:
        raise ValueError("--forecast-horizon must be >= 0.")
    if embedding_history_length < 1:
        raise ValueError("--embedding-history-length must be >= 1.")

    feature_names = []
    for lag in range(embedding_history_length - 1, -1, -1):
        suffix = "t" if lag == 0 else f"t_minus_{lag}"
        if not target_history_only:
            feature_names.extend([f"{col}_{suffix}" for col in feature_cols])
        if include_target_history:
            feature_names.append(f"{target_col}_{suffix}")

    rows: list[dict[str, object]] = []
    feature_blocks: list[np.ndarray] = []
    grouped = df.groupby(region_col, sort=False)
    for region_id, group in grouped:
        group = group.sort_values("time_index").reset_index(drop=True)
        feature_values = group[feature_cols].to_numpy(dtype=np.float32)
        target_values = group[target_col].to_numpy(dtype=np.float32)
        for input_pos in range(embedding_history_length - 1, len(group)):
            target_pos = input_pos + forecast_horizon
            if target_pos >= len(group):
                continue
            history = feature_values[input_pos - embedding_history_length + 1 : input_pos + 1]
            target_history = (
                target_values[input_pos - embedding_history_length + 1 : input_pos + 1]
                if include_target_history
                else None
            )
            history_parts = []
            for step_idx in range(embedding_history_length):
                step_parts = []
                if not target_history_only:
                    step_parts.append(history[step_idx])
                if target_history is not None:
                    step_parts.append(np.asarray([target_history[step_idx]], dtype=np.float32))
                history_parts.append(np.concatenate(step_parts, axis=0))

            input_row = group.iloc[input_pos]
            target_row = group.iloc[target_pos]
            rows.append(
                {
                    region_col: str(region_id),
                    "input_time_index": int(input_row["time_index"]),
                    "input_month": input_row[date_col],
                    "target_time_index": int(target_row["time_index"]),
                    "target_month": target_row[date_col],
                    target_col: float(target_row[target_col]),
                }
            )
            feature_blocks.append(np.concatenate(history_parts, axis=0))

    if not rows:
        raise ValueError(
            "No supervised samples were built. Try smaller --embedding-history-length or --forecast-horizon."
        )
    supervised = pd.DataFrame(rows)
    features = pd.DataFrame(np.vstack(feature_blocks), columns=feature_names)
    return pd.concat([supervised, features], axis=1), feature_names


def split_by_date(
    df: pd.DataFrame,
    date_col: str,
    test_date_ratio: float,
    val_date_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dates = np.array(sorted(df[date_col].unique()))
    if len(dates) < 3:
        raise ValueError("Need at least 3 unique target months for train/val/test splitting.")

    test_count = max(1, int(round(len(dates) * test_date_ratio)))
    train_val_dates = dates[:-test_count]
    test_dates = dates[-test_count:]

    val_count = max(1, int(round(len(train_val_dates) * val_date_ratio)))
    if len(train_val_dates) - val_count < 1:
        val_count = max(1, len(train_val_dates) - 1)
    train_dates = train_val_dates[:-val_count]
    val_dates = train_val_dates[-val_count:]

    return (
        df[date_col].isin(train_dates).to_numpy(),
        df[date_col].isin(val_dates).to_numpy(),
        df[date_col].isin(test_dates).to_numpy(),
    )


def split_by_region(
    df: pd.DataFrame,
    region_col: str,
    test_region_ratio: float,
    val_region_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    regions = np.array(sorted(df[region_col].astype(str).unique()))
    if len(regions) < 3:
        raise ValueError("Need at least 3 unique regions for train/val/test splitting.")

    rng = np.random.default_rng(seed)
    rng.shuffle(regions)
    test_count = max(1, int(round(len(regions) * test_region_ratio)))
    test_regions = regions[:test_count]
    train_val_regions = regions[test_count:]

    val_count = max(1, int(round(len(train_val_regions) * val_region_ratio)))
    if len(train_val_regions) - val_count < 1:
        val_count = max(1, len(train_val_regions) - 1)
    val_regions = train_val_regions[:val_count]
    train_regions = train_val_regions[val_count:]

    train_mask = df[region_col].astype(str).isin(train_regions).to_numpy()
    val_mask = df[region_col].astype(str).isin(val_regions).to_numpy()
    test_mask = df[region_col].astype(str).isin(test_regions).to_numpy()
    return train_mask, val_mask, test_mask


def split_by_spatiotemporal(
    df: pd.DataFrame,
    date_col: str,
    region_col: str,
    test_date_ratio: float,
    val_date_ratio: float,
    test_region_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dates = np.array(sorted(df[date_col].astype(str).unique()))
    if len(dates) < 3:
        raise ValueError("Need at least 3 unique target months for spatiotemporal splitting.")

    test_count = max(1, int(round(len(dates) * test_date_ratio)))
    train_val_dates = dates[:-test_count]
    test_dates = dates[-test_count:]
    if len(train_val_dates) < 2:
        raise ValueError("Not enough non-test months left for train/val in spatiotemporal splitting.")

    val_count = max(1, int(round(len(train_val_dates) * val_date_ratio)))
    if len(train_val_dates) - val_count < 1:
        val_count = max(1, len(train_val_dates) - 1)
    train_dates = train_val_dates[:-val_count]
    val_dates = train_val_dates[-val_count:]

    regions = np.array(sorted(df[region_col].astype(str).unique()))
    if len(regions) < 2:
        raise ValueError("Need at least 2 unique regions for spatiotemporal splitting.")

    rng = np.random.default_rng(seed)
    rng.shuffle(regions)
    test_region_count = max(1, int(round(len(regions) * test_region_ratio)))
    if test_region_count >= len(regions):
        test_region_count = len(regions) - 1
    test_regions = regions[:test_region_count]
    seen_regions = regions[test_region_count:]

    target_dates = df[date_col].astype(str)
    region_ids = df[region_col].astype(str)
    train_mask = target_dates.isin(train_dates).to_numpy()
    val_mask = (target_dates.isin(val_dates) & region_ids.isin(seen_regions)).to_numpy()
    test_mask = (target_dates.isin(test_dates) & region_ids.isin(test_regions)).to_numpy()

    if not train_mask.any():
        raise ValueError("Spatiotemporal split produced an empty training set.")
    if not val_mask.any():
        raise ValueError("Spatiotemporal split produced an empty validation set.")
    if not test_mask.any():
        raise ValueError("Spatiotemporal split produced an empty test set.")

    if np.any(train_mask & val_mask) or np.any(train_mask & test_mask):
        raise ValueError("Spatiotemporal split masks overlap with training samples.")
    if np.any(val_mask & test_mask):
        raise ValueError("Spatiotemporal split masks overlap across validation/test samples.")

    return train_mask, val_mask, test_mask


def transform_target(values: np.ndarray, log1p_target: bool) -> np.ndarray:
    if log1p_target:
        return np.log1p(np.clip(values, a_min=0.0, a_max=None))
    return values


def inverse_target(values: np.ndarray, log1p_target: bool) -> np.ndarray:
    if log1p_target:
        return np.expm1(values)
    return values


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total = 0.0
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(features), labels)
        loss.backward()
        optimizer.step()
        total += float(loss.detach().cpu()) * features.shape[0]
    return total / len(loader.dataset)


def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            loss = criterion(model(features), labels)
            total += float(loss.detach().cpu()) * features.shape[0]
    return total / len(loader.dataset)


def predict(model: nn.Module, features: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    loader = DataLoader(
        TabularRegressionDataset(features, np.zeros(features.shape[0], dtype=np.float32)),
        batch_size=batch_size,
        shuffle=False,
    )
    model.eval()
    outputs = []
    with torch.no_grad():
        for batch_features, _ in loader:
            outputs.append(model(batch_features.to(device)).cpu().numpy().reshape(-1))
    return np.concatenate(outputs, axis=0)


def transform_features_with_scaler(
    features: np.ndarray,
    scaler: StandardScaler,
    sequence_length: int,
    model_type: str,
) -> np.ndarray:
    if model_type == "mlp":
        return scaler.transform(features).astype(np.float32)

    if features.shape[1] % sequence_length != 0:
        raise ValueError("Feature width must be divisible by sequence length for GRU input.")
    step_dim = features.shape[1] // sequence_length
    features_3d = features.reshape(features.shape[0], sequence_length, step_dim)
    scaled = scaler.transform(features_3d.reshape(-1, step_dim)).reshape(features_3d.shape)
    return scaled.astype(np.float32)


def scale_features(
    features: np.ndarray,
    train_mask: np.ndarray,
    sequence_length: int,
    model_type: str,
) -> tuple[np.ndarray, StandardScaler, int]:
    if model_type == "mlp":
        scaler = StandardScaler()
        scaler.fit(features[train_mask])
        return scaler.transform(features).astype(np.float32), scaler, features.shape[1]

    if features.shape[1] % sequence_length != 0:
        raise ValueError("Feature width must be divisible by sequence length for GRU input.")
    step_dim = features.shape[1] // sequence_length
    features_3d = features.reshape(features.shape[0], sequence_length, step_dim)
    scaler = StandardScaler()
    scaler.fit(features_3d[train_mask].reshape(-1, step_dim))
    scaled = scaler.transform(features_3d.reshape(-1, step_dim)).reshape(features_3d.shape)
    return scaled.astype(np.float32), scaler, step_dim


def build_target_history_feature_names(target_col: str, embedding_history_length: int) -> list[str]:
    names = []
    for lag in range(embedding_history_length - 1, -1, -1):
        suffix = "t" if lag == 0 else f"t_minus_{lag}"
        names.append(f"{target_col}_{suffix}")
    return names


def predict_with_autoregressive_target_history(
    df: pd.DataFrame,
    feature_cols: list[str],
    region_col: str,
    target_col: str,
    predict_mask: np.ndarray,
    model: nn.Module,
    scaler: StandardScaler,
    args: argparse.Namespace,
    device: torch.device,
) -> pd.Series:
    if args.no_target_history:
        raise ValueError("--autoregressive-target-history requires target history to be enabled.")
    if args.forecast_horizon != 1:
        raise ValueError("--autoregressive-target-history currently requires --forecast-horizon 1.")

    target_history_cols = build_target_history_feature_names(target_col, args.embedding_history_length)
    missing = [col for col in target_history_cols if col not in feature_cols]
    if missing:
        raise ValueError(f"Missing target history columns required for autoregressive inference: {missing}")

    pred_df = df.loc[predict_mask, [region_col, "input_time_index", "target_time_index"] + feature_cols].copy()
    pred_df = pred_df.sort_values(["target_time_index", region_col]).reset_index().rename(columns={"index": "orig_index"})
    predicted_values = pd.Series(index=df.index[predict_mask], dtype=np.float32)
    predicted_lookup: dict[tuple[str, int], float] = {}

    for target_time, group in pred_df.groupby("target_time_index", sort=True):
        chunk_features = group[feature_cols].copy()
        input_times = group["input_time_index"].to_numpy(dtype=np.int32)
        region_ids = group[region_col].astype(str).to_numpy()

        for step_idx, col_name in enumerate(target_history_cols):
            history_times = input_times - args.embedding_history_length + 1 + step_idx
            col_values = chunk_features[col_name].to_numpy(dtype=np.float32, copy=True)
            replaced = False
            for row_idx, (region_id, history_time) in enumerate(zip(region_ids, history_times)):
                predicted = predicted_lookup.get((region_id, int(history_time)))
                if predicted is not None:
                    col_values[row_idx] = predicted
                    replaced = True
            if replaced:
                chunk_features[col_name] = col_values

        scaled_chunk = transform_features_with_scaler(
            chunk_features[feature_cols].to_numpy(dtype=np.float32),
            scaler=scaler,
            sequence_length=args.embedding_history_length,
            model_type=args.model_type,
        )
        pred_chunk = predict(model, scaled_chunk, args.batch_size, device)
        pred_chunk = inverse_target(pred_chunk, args.log1p_target)
        pred_chunk = np.clip(pred_chunk, a_min=0.0, a_max=None)

        for orig_index, region_id, pred_value in zip(
            group["orig_index"].to_numpy(dtype=np.int64),
            region_ids,
            pred_chunk,
        ):
            predicted_values.loc[int(orig_index)] = float(pred_value)
            predicted_lookup[(region_id, int(target_time))] = float(pred_value)

    return predicted_values


def build_model(args: argparse.Namespace, model_input_dim: int, device: torch.device) -> nn.Module:
    if args.model_type == "gru":
        return GRURegressor(
            input_dim=model_input_dim,
            hidden_dim=args.gru_hidden_dim,
            num_layers=args.gru_layers,
            dropout=args.dropout,
            head_hidden_dims=args.hidden_dims,
        ).to(device)
    return MLPRegressor(
        input_dim=model_input_dim,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
    ).to(device)


def fit_and_predict_split(
    args: argparse.Namespace,
    supervised_df: pd.DataFrame,
    feature_cols: list[str],
    features: np.ndarray,
    labels_train_space: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    predict_mask: np.ndarray,
    recursive_mask: np.ndarray | None,
    device: torch.device,
    log_writer: csv.DictWriter | None = None,
    fold_name: str = "single",
) -> tuple[np.ndarray, np.ndarray, float, int]:
    if not train_mask.any():
        raise ValueError(f"{fold_name}: empty training split.")
    if not val_mask.any():
        raise ValueError(f"{fold_name}: empty validation split.")
    if not predict_mask.any():
        raise ValueError(f"{fold_name}: empty prediction split.")

    all_features, scaler, model_input_dim = scale_features(
        features=features,
        train_mask=train_mask,
        sequence_length=args.embedding_history_length,
        model_type=args.model_type,
    )
    train_features = all_features[train_mask]
    val_features = all_features[val_mask]
    train_labels = labels_train_space[train_mask].astype(np.float32)
    val_labels = labels_train_space[val_mask].astype(np.float32)

    train_loader = DataLoader(
        TabularRegressionDataset(train_features, train_labels),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TabularRegressionDataset(val_features, val_labels),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = build_model(args, model_input_dim, device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0
    completed_epochs = 0
    for epoch in range(1, args.epochs + 1):
        completed_epochs = epoch
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        if log_writer is not None:
            log_writer.writerow(
                {
                    "fold": fold_name,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if epoch == 1 or epoch % 25 == 0:
            print(
                f"{fold_name} | Epoch {epoch:04d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}",
                flush=True,
            )
        if no_improve >= args.patience:
            print(f"{fold_name} | Early stopping at epoch {epoch}.", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    pred_train_space = predict(model, all_features[train_mask], args.batch_size, device)
    train_pred_values = inverse_target(pred_train_space, args.log1p_target)
    train_pred_values = np.clip(train_pred_values, a_min=0.0, a_max=None)
    if args.autoregressive_target_history:
        if recursive_mask is None:
            raise ValueError(f"{fold_name}: recursive inference requires a recursive mask.")
        recursive_predictions = predict_with_autoregressive_target_history(
            df=supervised_df,
            feature_cols=feature_cols,
            region_col=args.region_col,
            target_col=args.target_col,
            predict_mask=recursive_mask,
            model=model,
            scaler=scaler,
            args=args,
            device=device,
        )
        pred_values = recursive_predictions.loc[supervised_df.index[predict_mask]].to_numpy(dtype=np.float32)
    else:
        pred_target_space = predict(model, all_features[predict_mask], args.batch_size, device)
        pred_values = inverse_target(pred_target_space, args.log1p_target)
        pred_values = np.clip(pred_values, a_min=0.0, a_max=None)
    return pred_values, train_pred_values, best_val_loss, completed_epochs


def compute_metrics(true_values: np.ndarray, predicted_values: np.ndarray) -> Dict[str, float]:
    rmse = math.sqrt(mean_squared_error(true_values, predicted_values))
    nonzero_mask = np.abs(true_values) > 1e-8
    if nonzero_mask.any():
        mape = np.mean(
            np.abs((true_values[nonzero_mask] - predicted_values[nonzero_mask]) / true_values[nonzero_mask])
        ) * 100.0
    else:
        mape = float("nan")
    return {
        "mae": float(mean_absolute_error(true_values, predicted_values)),
        "rmse": float(rmse),
        "mape": float(mape),
        "r2": float(r2_score(true_values, predicted_values)),
    }


def save_prediction_plots(
    df_out: pd.DataFrame,
    target_col: str,
    output_dir: Path,
) -> None:
    plot_df = df_out.sort_values(["target_month", "region_id"]).reset_index(drop=True)
    x = np.arange(len(plot_df))

    plt.figure(figsize=(16, 6))
    plt.plot(x, plot_df[target_col].to_numpy(dtype=np.float32), label="True", linewidth=1.4)
    plt.plot(x, plot_df["prediction"].to_numpy(dtype=np.float32), label="Predicted", linewidth=1.2, alpha=0.85)

    month_positions = plot_df.groupby("target_month", sort=True).head(1).index.to_numpy()
    month_labels = plot_df.loc[month_positions, "target_month"].astype(str).to_numpy()
    plt.xticks(month_positions, month_labels, rotation=45, ha="right")
    plt.xlabel("Target month, then district order")
    plt.ylabel(target_col)
    plt.title("Monthly environmental prediction curve: all region-month samples")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "prediction_curve_all_samples.png", dpi=200)
    plt.close()

    monthly_mean = (
        plot_df.groupby("target_month", as_index=False)[[target_col, "prediction"]]
        .mean()
        .sort_values("target_month")
    )
    plt.figure(figsize=(12, 5))
    plt.plot(monthly_mean["target_month"], monthly_mean[target_col], marker="o", label="True monthly mean")
    plt.plot(monthly_mean["target_month"], monthly_mean["prediction"], marker="o", label="Predicted monthly mean")
    plt.xlabel("Target month")
    plt.ylabel(target_col)
    plt.title("Monthly mean environmental prediction curve")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "prediction_curve_monthly_mean.png", dpi=200)
    plt.close()


def save_standard_loss_plot(log_path: Path, output_dir: Path) -> None:
    if not log_path.exists():
        return
    log_df = pd.read_csv(log_path)
    if not {"epoch", "train_loss", "val_loss"}.issubset(log_df.columns):
        return
    plt.figure(figsize=(10, 5))
    plt.plot(log_df["epoch"], log_df["train_loss"], label="Train loss")
    plt.plot(log_df["epoch"], log_df["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve_train_val.png", dpi=200)
    plt.close()
    plt.figure(figsize=(10, 5))
    plt.plot(log_df["epoch"], log_df["train_loss"], label="Train loss")
    plt.plot(log_df["epoch"], log_df["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Training and validation loss by epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve_train_val_by_epoch.png", dpi=200)
    plt.close()
    plt.figure(figsize=(10, 5))
    plt.semilogy(log_df["epoch"], np.clip(log_df["train_loss"], 1e-12, None), label="Train loss")
    plt.semilogy(log_df["epoch"], np.clip(log_df["val_loss"], 1e-12, None), label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss (log scale)")
    plt.title("Training and validation loss by epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve_train_val_by_epoch_log.png", dpi=200)
    plt.close()


def save_rolling_epoch_loss_plot(log_path: Path, output_dir: Path) -> None:
    if not log_path.exists():
        return
    log_df = pd.read_csv(log_path)
    required_cols = {"fold", "epoch", "train_loss", "val_loss"}
    if not required_cols.issubset(log_df.columns):
        return

    epoch_summary = (
        log_df.groupby("epoch", as_index=False)
        .agg(
            train_loss_mean=("train_loss", "mean"),
            train_loss_std=("train_loss", "std"),
            val_loss_mean=("val_loss", "mean"),
            val_loss_std=("val_loss", "std"),
        )
        .sort_values("epoch")
    )
    epoch_summary = epoch_summary.fillna(0.0)
    epochs = epoch_summary["epoch"].to_numpy(dtype=np.float64)
    train_mean = epoch_summary["train_loss_mean"].to_numpy(dtype=np.float64)
    train_std = epoch_summary["train_loss_std"].to_numpy(dtype=np.float64)
    val_mean = epoch_summary["val_loss_mean"].to_numpy(dtype=np.float64)
    val_std = epoch_summary["val_loss_std"].to_numpy(dtype=np.float64)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_mean, label="Train loss mean")
    plt.fill_between(epochs, train_mean - train_std, train_mean + train_std, alpha=0.15)
    plt.plot(epochs, val_mean, label="Validation loss mean")
    plt.fill_between(epochs, val_mean - val_std, val_mean + val_std, alpha=0.15)
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Rolling folds: training and validation loss by epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve_train_val_by_epoch.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.semilogy(epochs, np.clip(train_mean, 1e-12, None), label="Train loss mean")
    plt.semilogy(epochs, np.clip(val_mean, 1e-12, None), label="Validation loss mean")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss (log scale)")
    plt.title("Rolling folds: training and validation loss by epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve_train_val_by_epoch_log.png", dpi=200)
    plt.close()


def save_rolling_loss_plots(fold_metrics: pd.DataFrame, output_dir: Path) -> None:
    if fold_metrics.empty:
        return
    x_labels = fold_metrics["target_month"].astype(str).to_numpy()
    x = np.arange(len(fold_metrics))

    train_mse = np.square(fold_metrics["train_rmse"].to_numpy(dtype=np.float64))
    test_mse = np.square(fold_metrics["rmse"].to_numpy(dtype=np.float64))
    plt.figure(figsize=(13, 5))
    plt.plot(x, train_mse, marker="o", label="Train MSE")
    plt.plot(x, test_mse, marker="o", label="Test MSE")
    plt.xticks(x, x_labels, rotation=45, ha="right")
    plt.xlabel("Rolling test month")
    plt.ylabel("MSE")
    plt.title("Rolling train/test MSE by target month")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve_train_test_mse.png", dpi=200)
    plt.close()

    plt.figure(figsize=(13, 5))
    plt.plot(x, fold_metrics["train_rmse"], marker="o", label="Train RMSE")
    plt.plot(x, fold_metrics["rmse"], marker="o", label="Test RMSE")
    plt.xticks(x, x_labels, rotation=45, ha="right")
    plt.xlabel("Rolling test month")
    plt.ylabel("RMSE")
    plt.title("Rolling train/test RMSE by target month")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve_train_test_rmse.png", dpi=200)
    plt.close()

    plt.figure(figsize=(13, 5))
    plt.plot(x, fold_metrics["train_r2"], marker="o", label="Train R2")
    plt.plot(x, fold_metrics["r2"], marker="o", label="Test R2")
    plt.axhline(0.0, color="gray", linewidth=1, linestyle="--")
    plt.xticks(x, x_labels, rotation=45, ha="right")
    plt.xlabel("Rolling test month")
    plt.ylabel("R2")
    plt.title("Rolling train/test R2 by target month")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "r2_curve_train_test.png", dpi=200)
    plt.close()


def run_rolling_validation(
    args: argparse.Namespace,
    df: pd.DataFrame,
    feature_cols: list[str],
    output_dir: Path,
    device: torch.device,
    data_filter_summary: Optional[dict[str, object]] = None,
) -> None:
    dates = np.array(sorted(df["target_month"].astype(str).unique()))
    min_train_months = max(1, args.rolling_min_train_months)
    first_test_idx = min_train_months + 1
    if len(dates) <= first_test_idx:
        raise ValueError(
            "Not enough target months for rolling validation. "
            "Try smaller --rolling-min-train-months."
        )

    features = df[feature_cols].to_numpy(dtype=np.float32)
    labels_raw = df[args.target_col].to_numpy(dtype=np.float32)
    labels_train_space = transform_target(labels_raw, args.log1p_target)

    prediction_frames = []
    fold_rows = []
    log_path = output_dir / "train_log.csv"
    with log_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["fold", "epoch", "train_loss", "val_loss"])
        writer.writeheader()
        for fold_idx, test_idx in enumerate(range(first_test_idx, len(dates)), start=1):
            test_month = dates[test_idx]
            val_month = dates[test_idx - 1]
            train_months = dates[: test_idx - 1]
            train_mask = df["target_month"].astype(str).isin(train_months).to_numpy()
            val_mask = df["target_month"].astype(str).eq(val_month).to_numpy()
            test_mask = df["target_month"].astype(str).eq(test_month).to_numpy()
            fold_name = f"fold_{fold_idx:02d}_{test_month}"

            pred_values, train_pred_values, best_val_loss, completed_epochs = fit_and_predict_split(
                args=args,
                supervised_df=df,
                feature_cols=feature_cols,
                features=features,
                labels_train_space=labels_train_space,
                train_mask=train_mask,
                val_mask=val_mask,
                predict_mask=test_mask,
                recursive_mask=(val_mask | test_mask) if args.autoregressive_target_history else None,
                device=device,
                log_writer=writer,
                fold_name=fold_name,
            )
            handle.flush()

            train_true = df.loc[train_mask, args.target_col].to_numpy(dtype=np.float32)
            train_metrics = compute_metrics(
                true_values=train_true,
                predicted_values=train_pred_values.astype(np.float32),
            )
            fold_out = df.loc[
                test_mask,
                [args.region_col, "input_time_index", "input_month", "target_time_index", "target_month", args.target_col],
            ].copy()
            fold_out["split"] = "test"
            fold_out["fold"] = fold_idx
            fold_out["train_start_month"] = train_months[0]
            fold_out["train_end_month"] = train_months[-1]
            fold_out["val_month"] = val_month
            fold_out["prediction"] = pred_values
            fold_out["error"] = fold_out["prediction"] - fold_out[args.target_col]
            prediction_frames.append(fold_out)

            metrics = compute_metrics(
                true_values=fold_out[args.target_col].to_numpy(dtype=np.float32),
                predicted_values=fold_out["prediction"].to_numpy(dtype=np.float32),
            )
            fold_rows.append(
                {
                    "fold": fold_idx,
                    "split": "test",
                    "train_start_month": train_months[0],
                    "train_end_month": train_months[-1],
                    "val_month": val_month,
                    "target_month": test_month,
                    "samples": int(test_mask.sum()),
                    "target_months": 1,
                    "regions": int(df.loc[test_mask, args.region_col].nunique()),
                    "train_samples": int(train_mask.sum()),
                    "train_target_months": int(df.loc[train_mask, "target_month"].nunique()),
                    "train_regions": int(df.loc[train_mask, args.region_col].nunique()),
                    "train_mae": train_metrics["mae"],
                    "train_rmse": train_metrics["rmse"],
                    "train_mape": train_metrics["mape"],
                    "train_r2": train_metrics["r2"],
                    "best_val_loss": best_val_loss,
                    "epochs": completed_epochs,
                    **metrics,
                }
            )

    df_out = pd.concat(prediction_frames, axis=0).reset_index(drop=True)
    overall_metrics = compute_metrics(
        true_values=df_out[args.target_col].to_numpy(dtype=np.float32),
        predicted_values=df_out["prediction"].to_numpy(dtype=np.float32),
    )
    fold_metrics_df = pd.DataFrame(fold_rows)
    metrics_rows = [
        {
            "split": "rolling_test",
            "samples": int(len(df_out)),
            "target_months": int(df_out["target_month"].nunique()),
            "regions": int(df_out[args.region_col].nunique()),
            **overall_metrics,
        },
        {
            "split": "rolling_train_mean",
            "samples": int(fold_metrics_df["train_samples"].mean()),
            "target_months": int(round(fold_metrics_df["train_target_months"].mean())),
            "regions": int(round(fold_metrics_df["train_regions"].mean())),
            "mae": float(fold_metrics_df["train_mae"].mean()),
            "rmse": float(fold_metrics_df["train_rmse"].mean()),
            "mape": float(fold_metrics_df["train_mape"].mean()),
            "r2": float(fold_metrics_df["train_r2"].mean()),
        }
    ]

    pd.DataFrame(metrics_rows).to_csv(output_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    fold_metrics_df.to_csv(output_dir / "rolling_fold_metrics.csv", index=False, encoding="utf-8-sig")
    df_out.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    if data_filter_summary is not None:
        summary_payload = dict(data_filter_summary)
        summary_payload.update(
            {
                "supervised_sample_count": int(len(df)),
                "supervised_region_count": int(df[args.region_col].astype(str).nunique()),
                "supervised_target_month_count": int(df["target_month"].astype(str).nunique()),
            }
        )
        save_data_filter_summary(output_dir, summary_payload)
    save_prediction_plots(df_out=df_out, target_col=args.target_col, output_dir=output_dir)
    save_rolling_loss_plots(fold_metrics=fold_metrics_df, output_dir=output_dir)
    save_rolling_epoch_loss_plot(log_path=log_path, output_dir=output_dir)

    print(
        f"Rolling validation folds={len(fold_rows)} "
        f"test_months={df_out['target_month'].nunique()} "
        f"test_samples={len(df_out)}",
        flush=True,
    )
    print(f"Saved outputs to: {output_dir}")
    print(pd.DataFrame(metrics_rows).to_string(index=False))


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.target_history_only and args.no_target_history:
        raise ValueError("--target-history-only cannot be used with --no-target-history.")
    if args.autoregressive_target_history and args.no_target_history:
        raise ValueError("--autoregressive-target-history requires target history to be enabled.")
    if args.autoregressive_target_history and args.forecast_horizon != 1:
        raise ValueError("--autoregressive-target-history currently requires --forecast-horizon 1.")
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = resolve_output_dir(Path(args.output_dir), add_timestamp=not args.no_output_timestamp)

    raw_df, raw_feature_cols, data_filter_stats = load_table(args)
    df, feature_cols = build_forecast_table(
        df=raw_df,
        feature_cols=raw_feature_cols,
        region_col=args.region_col,
        date_col=args.date_col,
        target_col=args.target_col,
        forecast_horizon=args.forecast_horizon,
        embedding_history_length=args.embedding_history_length,
        include_target_history=not args.no_target_history,
        target_history_only=args.target_history_only,
    )
    if args.split_mode == "rolling":
        run_rolling_validation(
            args=args,
            df=df,
            feature_cols=feature_cols,
            output_dir=output_dir,
            device=device,
            data_filter_summary={
                **data_filter_stats,
                "input_csv": str(Path(args.input_csv)),
                "target_col": args.target_col,
            },
        )
        return

    if args.split_mode == "date":
        train_mask, val_mask, test_mask = split_by_date(
            df=df,
            date_col="target_month",
            test_date_ratio=args.test_date_ratio,
            val_date_ratio=args.val_date_ratio,
        )
    elif args.split_mode == "spatiotemporal":
        train_mask, val_mask, test_mask = split_by_spatiotemporal(
            df=df,
            date_col="target_month",
            region_col=args.region_col,
            test_date_ratio=args.test_date_ratio,
            val_date_ratio=args.val_date_ratio,
            test_region_ratio=args.test_region_ratio,
            seed=args.seed,
        )
    else:
        train_mask, val_mask, test_mask = split_by_region(
            df=df,
            region_col=args.region_col,
            test_region_ratio=args.test_region_ratio,
            val_region_ratio=args.val_region_ratio,
            seed=args.seed,
        )

    features = df[feature_cols].to_numpy(dtype=np.float32)
    labels_raw = df[args.target_col].to_numpy(dtype=np.float32)
    labels_train_space = transform_target(labels_raw, args.log1p_target)

    all_features, scaler, model_input_dim = scale_features(
        features=features,
        train_mask=train_mask,
        sequence_length=args.embedding_history_length,
        model_type=args.model_type,
    )
    train_features = all_features[train_mask]
    val_features = all_features[val_mask]
    train_labels = labels_train_space[train_mask].astype(np.float32)
    val_labels = labels_train_space[val_mask].astype(np.float32)

    train_loader = DataLoader(
        TabularRegressionDataset(train_features, train_labels),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TabularRegressionDataset(val_features, val_labels),
        batch_size=args.batch_size,
        shuffle=False,
    )

    if args.model_type == "gru":
        model = GRURegressor(
            input_dim=model_input_dim,
            hidden_dim=args.gru_hidden_dim,
            num_layers=args.gru_layers,
            dropout=args.dropout,
            head_hidden_dims=args.hidden_dims,
        ).to(device)
    else:
        model = MLPRegressor(
            input_dim=model_input_dim,
            hidden_dims=args.hidden_dims,
            dropout=args.dropout,
        ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0
    with (output_dir / "train_log.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss = evaluate_loss(model, val_loader, criterion, device)
            writer.writerow({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            handle.flush()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if epoch == 1 or epoch % 25 == 0:
                print(f"Epoch {epoch:04d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}", flush=True)
            if no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch}.", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_pred_space = predict(model, all_features[train_mask], args.batch_size, device)
    train_pred_values = inverse_target(train_pred_space, args.log1p_target)
    train_pred_values = np.clip(train_pred_values, a_min=0.0, a_max=None)
    pred_all = np.full(len(df), np.nan, dtype=np.float32)
    pred_all[train_mask] = train_pred_values
    if args.autoregressive_target_history:
        eval_predictions = predict_with_autoregressive_target_history(
            df=df,
            feature_cols=feature_cols,
            region_col=args.region_col,
            target_col=args.target_col,
            predict_mask=(val_mask | test_mask),
            model=model,
            scaler=scaler,
            args=args,
            device=device,
        )
        pred_all[eval_predictions.index.to_numpy(dtype=np.int64)] = eval_predictions.to_numpy(dtype=np.float32)
    else:
        pred_all_train_space = predict(model, all_features, args.batch_size, device)
        pred_all = inverse_target(pred_all_train_space, args.log1p_target)
        pred_all = np.clip(pred_all, a_min=0.0, a_max=None)
    df_out = df[
        [args.region_col, "input_time_index", "input_month", "target_time_index", "target_month", args.target_col]
    ].copy()
    df_out["split"] = "unused"
    df_out.loc[train_mask, "split"] = "train"
    df_out.loc[val_mask, "split"] = "val"
    df_out.loc[test_mask, "split"] = "test"
    df_out["prediction"] = pred_all
    df_out["error"] = df_out["prediction"] - df_out[args.target_col]

    metrics_rows = []
    split_masks: list[tuple[str, np.ndarray]] = [("train", train_mask), ("val", val_mask)]
    split_masks.append(("test", test_mask))

    for split_name, mask in split_masks:
        if mask is None or not mask.any():
            continue
        metrics = compute_metrics(
            true_values=df_out.loc[mask, args.target_col].to_numpy(dtype=np.float32),
            predicted_values=df_out.loc[mask, "prediction"].to_numpy(dtype=np.float32),
        )
        metrics_rows.append(
            {
                "split": split_name,
                "samples": int(mask.sum()),
                "target_months": int(df.loc[mask, "target_month"].nunique()),
                "regions": int(df.loc[mask, args.region_col].nunique()),
                **metrics,
            }
        )

    used_mask = df_out["split"].ne("unused")
    all_metrics = compute_metrics(
        true_values=df_out.loc[used_mask, args.target_col].to_numpy(dtype=np.float32),
        predicted_values=df_out.loc[used_mask, "prediction"].to_numpy(dtype=np.float32),
    )
    metrics_rows.append(
        {
            "split": "all",
            "samples": int(used_mask.sum()),
            "target_months": int(df_out.loc[used_mask, "target_month"].nunique()),
            "regions": int(df_out.loc[used_mask, args.region_col].nunique()),
            **all_metrics,
        }
    )

    pd.DataFrame(metrics_rows).to_csv(output_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    df_out.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    save_data_filter_summary(
        output_dir,
        {
            **data_filter_stats,
            "input_csv": str(Path(args.input_csv)),
            "target_col": args.target_col,
            "supervised_sample_count": int(len(df)),
            "supervised_region_count": int(df[args.region_col].astype(str).nunique()),
            "supervised_target_month_count": int(df["target_month"].astype(str).nunique()),
        },
    )
    save_prediction_plots(df_out=df_out, target_col=args.target_col, output_dir=output_dir)
    save_standard_loss_plot(log_path=output_dir / "train_log.csv", output_dir=output_dir)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_cols": feature_cols,
            "args": vars(args),
            "best_val_loss": best_val_loss,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
        },
        output_dir / "best_model.pt",
    )

    print(
        f"Built supervised samples={len(df)} "
        f"history={args.embedding_history_length} forecast_horizon={args.forecast_horizon} "
        f"target_history={not args.no_target_history}",
        flush=True,
    )
    print(f"Saved outputs to: {output_dir}")
    print(pd.DataFrame(metrics_rows).to_string(index=False))


if __name__ == "__main__":
    main()
