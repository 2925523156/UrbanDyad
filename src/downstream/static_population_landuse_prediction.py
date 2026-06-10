from __future__ import annotations

import argparse
import copy
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class ArrayDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, task: str) -> None:
        self.features = torch.from_numpy(features.astype(np.float32, copy=False))
        if task == "population":
            self.labels = torch.from_numpy(labels.astype(np.float32, copy=False)).view(-1, 1)
        else:
            self.labels = torch.from_numpy(labels.astype(np.int64, copy=False))

    def __getitem__(self, index: int):
        return self.features[index], self.labels[index]

    def __len__(self) -> int:
        return int(self.features.shape[0])


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run static downstream tasks on S2 static embeddings.")
    parser.add_argument("--task", choices=["population", "landuse"], required=True)
    parser.add_argument("--features", required=True, help="Path to static embedding .npy file.")
    parser.add_argument("--labels", required=True, help="Path to downstream label .npy file.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--sample-mode", choices=["all", "nonzero"], default="nonzero")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--no-timestamp", action="store_true", help="Write directly into output-root/run-name.")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_arrays(
    features_path: Path,
    labels_path: Path,
    task: str,
    sample_mode: str,
    max_samples: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    features = np.load(features_path)
    labels = np.load(labels_path, allow_pickle=True)

    if labels.ndim > 1:
        if labels.shape[1] == 1:
            labels = labels[:, 0]
        else:
            raise ValueError(f"Expected 1D labels or shape [N, 1], got {labels.shape}")

    max_available = min(features.shape[0], labels.shape[0])
    features = np.asarray(features[:max_available], dtype=np.float32)
    labels = np.asarray(labels[:max_available])

    if task == "population":
        labels = labels.astype(np.float32, copy=False)
        keep_mask = labels > 0 if sample_mode == "nonzero" else np.ones(max_available, dtype=bool)
    else:
        labels = labels.astype(np.int64, copy=False)
        keep_mask = np.isfinite(labels)

    features = features[keep_mask]
    labels = labels[keep_mask]

    if max_samples > 0:
        features = features[:max_samples]
        labels = labels[:max_samples]

    if features.shape[0] <= 1:
        raise ValueError("Not enough samples after filtering.")

    stats: dict[str, object] = {
        "features_shape": list(features.shape),
        "raw_aligned_samples": int(max_available),
        "selected_samples": int(features.shape[0]),
        "filtered_samples": int(max_available - features.shape[0]),
        "sample_mode": sample_mode if task == "population" else "all_labeled",
    }

    if task == "landuse":
        unique_classes = np.sort(np.unique(labels))
        class_mapping = {int(old): int(new) for new, old in enumerate(unique_classes)}
        labels = np.asarray([class_mapping[int(value)] for value in labels], dtype=np.int64)
        mapped_classes, class_counts = np.unique(labels, return_counts=True)
        stats["class_mapping"] = class_mapping
        stats["class_counts"] = {int(cls): int(count) for cls, count in zip(mapped_classes, class_counts)}

    if task == "population":
        stats["population_nonzero"] = int((labels > 0).sum())
        stats["population_max"] = float(np.max(labels))
        stats["population_mean"] = float(np.mean(labels))

    return features, labels, stats


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * inputs.size(0)
    return total_loss / len(loader.dataset)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    task: str,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    pred_parts = []
    label_parts = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += float(loss.item()) * inputs.size(0)
            if task == "population":
                pred_parts.append(outputs.cpu().numpy().reshape(-1))
            else:
                pred_parts.append(torch.argmax(outputs, dim=1).cpu().numpy())
            label_parts.append(labels.cpu().numpy().reshape(-1))
    return total_loss / len(loader.dataset), np.concatenate(pred_parts), np.concatenate(label_parts)


def pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2 or float(np.std(y_true)) == 0.0 or float(np.std(y_pred)) == 0.0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def save_population_outputs(output_dir: Path, y_true: np.ndarray, y_pred: np.ndarray, test_loss: float) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    metrics = {
        "test_loss": float(test_loss),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "mse": float(mse),
        "r2": float(r2_score(y_true, y_pred)),
        "pcc": pearson_corr(y_true, y_pred),
    }
    with (output_dir / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_population", "predicted_population", "error", "absolute_error"])
        for true_value, pred_value in zip(y_true.tolist(), y_pred.tolist()):
            error = pred_value - true_value
            writer.writerow([true_value, pred_value, error, abs(error)])
    return metrics


def save_landuse_outputs(output_dir: Path, y_true: np.ndarray, y_pred: np.ndarray, test_loss: float) -> dict[str, float]:
    metrics = {
        "test_loss": float(test_loss),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    with (output_dir / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_class", "predicted_class", "correct"])
        for true_value, pred_value in zip(y_true.tolist(), y_pred.tolist()):
            writer.writerow([true_value, pred_value, int(true_value == pred_value)])

    cm = confusion_matrix(y_true, y_pred)
    np.savetxt(output_dir / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")
    report = classification_report(y_true, y_pred, digits=4, zero_division=0)
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    return metrics


def save_metrics(output_dir: Path, metrics: dict[str, float]) -> None:
    with (output_dir / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, value])


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / (args.run_name if args.no_timestamp else f"{args.run_name}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    features, labels, stats = load_arrays(
        features_path=Path(args.features),
        labels_path=Path(args.labels),
        task=args.task,
        sample_mode=args.sample_mode,
        max_samples=args.max_samples,
    )

    stratify_labels = labels if args.task == "landuse" else None
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=stratify_labels,
    )
    stratify_train = y_train if args.task == "landuse" else None
    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=stratify_train,
    )

    device = resolve_device(args.device)
    output_dim = 1 if args.task == "population" else int(np.max(labels)) + 1
    model = MLP(input_dim=x_train.shape[1], hidden_dims=[512, 256, 128, 64], output_dim=output_dim).to(device)
    criterion: nn.Module = nn.MSELoss() if args.task == "population" else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    train_loader = DataLoader(ArrayDataset(x_train, y_train, args.task), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(ArrayDataset(x_val, y_val, args.task), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(ArrayDataset(x_test, y_test, args.task), batch_size=args.batch_size, shuffle=False)

    config = vars(args).copy()
    config.update({"output_dir": str(output_dir), "device_resolved": str(device), "data_stats": stats})
    (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    best_val_loss = float("inf")
    best_state = None
    stale_epochs = 0
    log_rows = []

    print(f"Task: {args.task}")
    print(f"Features: {features.shape}")
    print(f"Output: {output_dir}")
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_pred, y_val_true = evaluate(model, val_loader, criterion, device, args.task)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1

        if args.task == "population":
            val_score = r2_score(y_val_true, val_pred)
            val_score_name = "val_r2"
        else:
            val_score = accuracy_score(y_val_true, val_pred)
            val_score_name = "val_accuracy"

        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                val_score_name: val_score,
                "best_val_loss": best_val_loss,
            }
        )

        if epoch % 50 == 0:
            print(
                f"Epoch {epoch} - train_loss={train_loss:.6f}, "
                f"val_loss={val_loss:.6f}, {val_score_name}={val_score:.6f}"
            )

        if stale_epochs > args.patience:
            print(f"Early stop at epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    torch.save(best_state, output_dir / "best_model.pt")
    test_loss, test_pred, test_labels = evaluate(model, test_loader, criterion, device, args.task)

    if args.task == "population":
        metrics = save_population_outputs(output_dir, test_labels, test_pred, test_loss)
    else:
        metrics = save_landuse_outputs(output_dir, test_labels, test_pred, test_loss)
    metrics["best_val_loss"] = float(best_val_loss)
    metrics["train_samples"] = int(x_train.shape[0])
    metrics["val_samples"] = int(x_val.shape[0])
    metrics["test_samples"] = int(x_test.shape[0])
    save_metrics(output_dir, metrics)

    with (output_dir / "train_log.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = sorted({key for row in log_rows for key in row.keys()})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log_rows)

    print("Final metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
