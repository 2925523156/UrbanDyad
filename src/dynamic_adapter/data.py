from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class DynamicFeatureBundle:
    static_embeddings: np.ndarray
    dynamic_windows: np.ndarray
    time_context: np.ndarray
    positive_pairs: list[np.ndarray]
    cell_ids: np.ndarray
    time_index: np.ndarray
    timestamps: Optional[np.ndarray] = None

    @property
    def num_samples(self) -> int:
        return int(self.dynamic_windows.shape[0])


@dataclass
class HourlyDynamicFeatureBundle:
    static_embeddings: np.ndarray
    hourly_state: np.ndarray
    time_context: np.ndarray
    cell_ids: np.ndarray
    time_index: np.ndarray
    window_length: int
    nearby_radius: int = 3
    similarity_threshold: float = 0.8
    periodic_offsets: tuple[int, ...] = (-24, 24)
    max_samples: int = 0
    timestamps: Optional[np.ndarray] = None

    @property
    def num_cells(self) -> int:
        return int(self.cell_ids.shape[0])

    @property
    def num_valid_times(self) -> int:
        return max(0, int(self.hourly_state.shape[0]) - int(self.window_length))

    @property
    def num_samples(self) -> int:
        total = self.num_valid_times * self.num_cells
        return min(total, self.max_samples) if self.max_samples > 0 else total


@dataclass
class DailyDynamicFeatureBundle:
    static_embeddings: np.ndarray
    daily_state: np.ndarray
    day_context: np.ndarray
    cell_ids: np.ndarray
    time_index: np.ndarray
    window_length: int
    nearby_radius: int = 3
    similarity_threshold: float = 0.8
    periodic_offsets: tuple[int, ...] = (-7, 7)
    max_samples: int = 0
    timestamps: Optional[np.ndarray] = None
    day_of_month: Optional[np.ndarray] = None
    day_of_week: Optional[np.ndarray] = None
    same_day_of_month_positives: bool = False
    same_day_of_week_positives: bool = False

    @property
    def num_cells(self) -> int:
        return int(self.cell_ids.shape[0])

    @property
    def num_valid_times(self) -> int:
        return max(0, int(self.daily_state.shape[0]) - int(self.window_length))

    @property
    def num_samples(self) -> int:
        total = self.num_valid_times * self.num_cells
        return min(total, self.max_samples) if self.max_samples > 0 else total


@dataclass
class MonthlyDynamicFeatureBundle:
    static_embeddings: np.ndarray
    monthly_state: np.ndarray
    month_context: np.ndarray
    cell_ids: np.ndarray
    time_index: np.ndarray
    window_length: int
    nearby_radius: int = 2
    similarity_threshold: float = 0.7
    periodic_offsets: tuple[int, ...] = ()
    max_samples: int = 0
    timestamps: Optional[np.ndarray] = None

    @property
    def num_cells(self) -> int:
        return int(self.cell_ids.shape[0])

    @property
    def num_valid_times(self) -> int:
        return max(0, int(self.monthly_state.shape[0]) - int(self.window_length))

    @property
    def num_samples(self) -> int:
        total = self.num_valid_times * self.num_cells
        return min(total, self.max_samples) if self.max_samples > 0 else total


class DynamicIndexDataset(Dataset):
    def __init__(self, indices: np.ndarray) -> None:
        self.indices = indices.astype(np.int64, copy=False)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int) -> int:
        return int(self.indices[idx])


class DynamicBatchCollator:
    def __init__(
        self,
        bundle: DynamicFeatureBundle | HourlyDynamicFeatureBundle | DailyDynamicFeatureBundle | MonthlyDynamicFeatureBundle,
        augment_positive_samples: bool = False,
        max_positive_samples_per_anchor: int = 2,
    ) -> None:
        self.bundle = bundle
        self.augment_positive_samples = augment_positive_samples
        self.max_positive_samples_per_anchor = max_positive_samples_per_anchor

    def __call__(self, batch_indices: Iterable[int]) -> Dict[str, object]:
        if isinstance(self.bundle, HourlyDynamicFeatureBundle):
            return self._collate_temporal_state(
                batch_indices,
                state_attr="hourly_state",
                context_attr="time_context",
                append_current_state_to_window=True,
            )
        if isinstance(self.bundle, DailyDynamicFeatureBundle):
            return self._collate_temporal_state(
                batch_indices,
                state_attr="daily_state",
                context_attr="day_context",
                append_current_state_to_window=True,
            )
        if isinstance(self.bundle, MonthlyDynamicFeatureBundle):
            return self._collate_temporal_state(
                batch_indices,
                state_attr="monthly_state",
                context_attr="month_context",
                append_current_state_to_window=True,
            )
        return self._collate_windowed(batch_indices)

    def _collate_windowed(self, batch_indices: Iterable[int]) -> Dict[str, object]:
        indices = np.asarray(list(batch_indices), dtype=np.int64)
        if self.augment_positive_samples:
            indices = self._augment_windowed_indices(indices)
        local_lookup = {int(global_idx): local_idx for local_idx, global_idx in enumerate(indices.tolist())}
        positive_mask = np.zeros((indices.shape[0], indices.shape[0]), dtype=np.bool_)
        for row_idx, global_idx in enumerate(indices.tolist()):
            positive_globals = self.bundle.positive_pairs[global_idx]
            for target_idx in positive_globals.tolist():
                local_col = local_lookup.get(int(target_idx))
                if local_col is not None:
                    positive_mask[row_idx, local_col] = True
        return {
            "global_indices": torch.from_numpy(indices.copy()),
            "static_embedding": torch.from_numpy(
                np.asarray(self.bundle.static_embeddings[indices], dtype=np.float32)
            ),
            "dynamic_window": torch.from_numpy(
                np.asarray(self.bundle.dynamic_windows[indices], dtype=np.float32)
            ),
            "time_context": torch.from_numpy(
                np.asarray(self.bundle.time_context[indices], dtype=np.float32)
            ),
            "positive_mask": torch.from_numpy(positive_mask),
            "cell_ids": self.bundle.cell_ids[indices].tolist(),
            "region_ids": self.bundle.cell_ids[indices].tolist(),
            "time_index": torch.from_numpy(self.bundle.time_index[indices].astype(np.int64, copy=False)),
        }

    def _collate_temporal_state(
        self,
        batch_indices: Iterable[int],
        state_attr: str,
        context_attr: str,
        append_current_state_to_context: bool = False,
        append_current_state_to_window: bool = False,
    ) -> Dict[str, object]:
        bundle = self.bundle
        temporal_state = getattr(bundle, state_attr)
        temporal_context = getattr(bundle, context_attr)
        indices = np.asarray(list(batch_indices), dtype=np.int64)
        if self.augment_positive_samples:
            indices = self._augment_temporal_state_indices(indices, state_attr=state_attr)
        cell_indices = np.mod(indices, bundle.num_cells).astype(np.int64)
        target_times = (indices // bundle.num_cells + bundle.window_length).astype(np.int64)

        window_steps = bundle.window_length + (1 if append_current_state_to_window else 0)
        dynamic_windows = np.empty((indices.shape[0], window_steps, temporal_state.shape[2]), dtype=np.float32)
        for row_idx, (target_time, cell_idx) in enumerate(zip(target_times.tolist(), cell_indices.tolist())):
            end_time = target_time + 1 if append_current_state_to_window else target_time
            dynamic_windows[row_idx] = np.asarray(
                temporal_state[target_time - bundle.window_length : end_time, cell_idx, :],
                dtype=np.float32,
            )

        positive_mask = self._build_temporal_state_positive_mask(
            cell_indices=cell_indices,
            target_times=target_times,
            state_attr=state_attr,
        )
        time_context = np.asarray(temporal_context[target_times], dtype=np.float32)
        if append_current_state_to_context:
            current_state = np.asarray(temporal_state[target_times, cell_indices, :], dtype=np.float32)
            time_context = np.concatenate([time_context, current_state], axis=1)
        return {
            "global_indices": torch.from_numpy(indices.copy()),
            "static_embedding": torch.from_numpy(
                np.asarray(bundle.static_embeddings[cell_indices], dtype=np.float32)
            ),
            "dynamic_window": torch.from_numpy(dynamic_windows),
            "time_context": torch.from_numpy(time_context),
            "positive_mask": torch.from_numpy(positive_mask),
            "cell_ids": bundle.cell_ids[cell_indices].tolist(),
            "region_ids": bundle.cell_ids[cell_indices].tolist(),
            "time_index": torch.from_numpy(target_times.astype(np.int64, copy=False)),
        }

    def _augment_windowed_indices(self, indices: np.ndarray) -> np.ndarray:
        bundle = self.bundle
        expanded = indices.tolist()
        seen = set(int(value) for value in expanded)
        for global_idx in indices.tolist():
            positives = bundle.positive_pairs[int(global_idx)]
            added = 0
            for pos_idx in positives.tolist():
                if int(pos_idx) not in seen:
                    expanded.append(int(pos_idx))
                    seen.add(int(pos_idx))
                    added += 1
                if added >= self.max_positive_samples_per_anchor:
                    break
        return np.asarray(expanded, dtype=np.int64)

    def _augment_temporal_state_indices(self, indices: np.ndarray, state_attr: str) -> np.ndarray:
        expanded = indices.tolist()
        seen = set(int(value) for value in expanded)
        for global_idx in indices.tolist():
            added = 0
            for pos_idx in self._temporal_state_positive_candidates(int(global_idx), state_attr=state_attr):
                if pos_idx not in seen:
                    expanded.append(pos_idx)
                    seen.add(pos_idx)
                    added += 1
                if added >= self.max_positive_samples_per_anchor:
                    break
        return np.asarray(expanded, dtype=np.int64)

    def _temporal_state_positive_candidates(self, global_idx: int, state_attr: str) -> list[int]:
        bundle = self.bundle
        temporal_state = getattr(bundle, state_attr)
        cell_idx = global_idx % bundle.num_cells
        target_time = global_idx // bundle.num_cells + bundle.window_length
        candidates: list[int] = []
        for offset in bundle.periodic_offsets:
            candidate_time = int(target_time + offset)
            if bundle.window_length <= candidate_time < temporal_state.shape[0]:
                candidates.append((candidate_time - bundle.window_length) * bundle.num_cells + cell_idx)

        if (
            state_attr == "daily_state"
            and getattr(bundle, "same_day_of_month_positives", False)
            and getattr(bundle, "day_of_month", None) is not None
        ):
            day_of_month = getattr(bundle, "day_of_month")
            anchor_day = int(day_of_month[target_time])
            for candidate_time in range(bundle.window_length, temporal_state.shape[0]):
                if candidate_time == target_time:
                    continue
                if int(day_of_month[candidate_time]) == anchor_day:
                    candidates.append((candidate_time - bundle.window_length) * bundle.num_cells + cell_idx)

        if (
            state_attr == "daily_state"
            and getattr(bundle, "same_day_of_week_positives", False)
            and getattr(bundle, "day_of_week", None) is not None
        ):
            day_of_week = getattr(bundle, "day_of_week")
            anchor_weekday = int(day_of_week[target_time])
            for candidate_time in range(bundle.window_length, temporal_state.shape[0]):
                if candidate_time == target_time:
                    continue
                if int(day_of_week[candidate_time]) == anchor_weekday:
                    candidates.append((candidate_time - bundle.window_length) * bundle.num_cells + cell_idx)

        anchor_state = np.asarray(temporal_state[target_time, cell_idx, :], dtype=np.float32)
        anchor_norm = float(np.linalg.norm(anchor_state))
        for gap in range(1, bundle.nearby_radius + 1):
            for candidate_time in (target_time - gap, target_time + gap):
                if not (bundle.window_length <= candidate_time < temporal_state.shape[0]):
                    continue
                candidate_state = np.asarray(temporal_state[candidate_time, cell_idx, :], dtype=np.float32)
                denom = anchor_norm * float(np.linalg.norm(candidate_state))
                similarity = 0.0 if denom <= 0.0 else float(np.dot(anchor_state, candidate_state) / denom)
                if similarity >= bundle.similarity_threshold:
                    candidates.append((candidate_time - bundle.window_length) * bundle.num_cells + cell_idx)
        return candidates

    def _build_temporal_state_positive_mask(
        self,
        cell_indices: np.ndarray,
        target_times: np.ndarray,
        state_attr: str,
    ) -> np.ndarray:
        bundle = self.bundle
        temporal_state = getattr(bundle, state_attr)
        batch_size = int(cell_indices.shape[0])
        positive_mask = np.zeros((batch_size, batch_size), dtype=np.bool_)
        hours = np.mod(target_times, 24) if state_attr == "hourly_state" else None
        current_state = np.asarray(
            temporal_state[target_times, cell_indices, :],
            dtype=np.float32,
        )
        norms = np.linalg.norm(current_state, axis=1)

        periodic_offsets = set(int(offset) for offset in bundle.periodic_offsets)
        for row_idx in range(batch_size):
            for col_idx in range(batch_size):
                if row_idx == col_idx or cell_indices[row_idx] != cell_indices[col_idx]:
                    continue
                time_diff = int(target_times[col_idx] - target_times[row_idx])
                if time_diff in periodic_offsets:
                    positive_mask[row_idx, col_idx] = True
                    continue
                if (
                    state_attr == "daily_state"
                    and getattr(bundle, "same_day_of_month_positives", False)
                    and getattr(bundle, "day_of_month", None) is not None
                ):
                    row_time = int(target_times[row_idx])
                    col_time = int(target_times[col_idx])
                    if row_time != col_time and int(bundle.day_of_month[row_time]) == int(bundle.day_of_month[col_time]):
                        positive_mask[row_idx, col_idx] = True
                        continue
                if (
                    state_attr == "daily_state"
                    and getattr(bundle, "same_day_of_week_positives", False)
                    and getattr(bundle, "day_of_week", None) is not None
                ):
                    row_time = int(target_times[row_idx])
                    col_time = int(target_times[col_idx])
                    if row_time != col_time and int(bundle.day_of_week[row_time]) == int(bundle.day_of_week[col_time]):
                        positive_mask[row_idx, col_idx] = True
                        continue
                if hours is not None and hours[row_idx] == hours[col_idx] and time_diff != 0:
                    positive_mask[row_idx, col_idx] = True
                    continue

                gap = abs(time_diff)
                if gap == 0 or gap > bundle.nearby_radius:
                    continue
                denom = float(norms[row_idx] * norms[col_idx])
                similarity = 0.0 if denom <= 0.0 else float(
                    np.dot(current_state[row_idx], current_state[col_idx]) / denom
                )
                if similarity >= bundle.similarity_threshold:
                    positive_mask[row_idx, col_idx] = True
        return positive_mask


def load_static_embedding_mapping(
    embeddings_path: Path,
    region_ids_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    embeddings = np.load(embeddings_path)
    if embeddings.ndim != 2:
        raise ValueError("Static embeddings must have shape [N, D].")

    if region_ids_path.suffix.lower() == ".npy":
        region_ids = np.load(region_ids_path, allow_pickle=True)
    else:
        df = pd.read_csv(region_ids_path)
        if "region_id" in df.columns:
            id_column = "region_id"
        elif "district_id" in df.columns:
            id_column = "district_id"
        elif "QXBH" in df.columns:
            id_column = "QXBH"
        elif "cell_id" in df.columns:
            id_column = "cell_id"
        else:
            raise ValueError("Static id file must contain a 'region_id', 'district_id', 'QXBH', or 'cell_id' column.")
        region_ids = df[id_column].astype(str).to_numpy()

    region_ids = np.asarray(region_ids).astype(str)
    if embeddings.shape[0] != region_ids.shape[0]:
        raise ValueError("Static embeddings row count must match static region ids.")
    return embeddings, region_ids


def _resolve_time_index(index_df: pd.DataFrame) -> tuple[np.ndarray, Optional[np.ndarray]]:
    timestamps: Optional[np.ndarray] = None
    if "time_index" in index_df.columns:
        time_index = index_df["time_index"].to_numpy(dtype=np.int64)
        if "timestamp" in index_df.columns:
            timestamps = index_df["timestamp"].astype(str).to_numpy()
        return time_index, timestamps

    if "timestamp" in index_df.columns:
        timestamp_series = pd.to_datetime(index_df["timestamp"], errors="raise")
        base_timestamp = timestamp_series.min()
        time_delta = timestamp_series - base_timestamp
        time_index = (time_delta.dt.total_seconds() // 3600).astype(np.int64).to_numpy()
        timestamps = index_df["timestamp"].astype(str).to_numpy()
        return time_index, timestamps

    if {"day_index", "hour"} <= set(index_df.columns):
        time_index = (
            index_df["day_index"].to_numpy(dtype=np.int64) * 24
            + index_df["hour"].to_numpy(dtype=np.int64)
        )
        return time_index, None

    raise ValueError(
        "Sample index must contain 'time_index', 'timestamp', or both 'day_index' and 'hour'."
    )


def _resolve_hour(index_df: pd.DataFrame, time_index: np.ndarray) -> np.ndarray:
    if "hour" in index_df.columns:
        return index_df["hour"].to_numpy(dtype=np.int64)
    return np.mod(time_index, 24).astype(np.int64)


def _resolve_current_dynamic_state(dynamic_windows: np.ndarray, current_state_path: Optional[Path]) -> np.ndarray:
    if current_state_path is not None:
        current_state = np.load(current_state_path)
        if current_state.ndim != 2:
            raise ValueError("Current dynamic state must have shape [N, D].")
        return np.asarray(current_state, dtype=np.float32)
    return np.asarray(dynamic_windows[:, -1, :], dtype=np.float32)


def _resolve_region_id_column(df: pd.DataFrame, context: str) -> str:
    for column in ("region_id", "district_id", "QXBH", "cell_id"):
        if column in df.columns:
            return column
    raise ValueError(f"{context} must contain one of: region_id, district_id, QXBH, cell_id.")


def _build_positive_pairs(
    cell_ids: np.ndarray,
    time_index: np.ndarray,
    hour: np.ndarray,
    current_state: np.ndarray,
    periodic_offsets: tuple[int, ...],
    nearby_radius: int,
    similarity_threshold: float,
) -> list[np.ndarray]:
    cell_ids = cell_ids.astype(str)
    current_norm = np.linalg.norm(current_state, axis=1)
    sample_count = cell_ids.shape[0]
    positives: list[set[int]] = [set() for _ in range(sample_count)]

    cell_time_lookup: dict[tuple[str, int], int] = {}
    same_cell_hour_lookup: dict[tuple[str, int], list[int]] = {}
    same_cell_indices: dict[str, list[int]] = {}

    for idx, (cell_id, t_idx, hour_value) in enumerate(zip(cell_ids.tolist(), time_index.tolist(), hour.tolist())):
        cell_time_lookup[(cell_id, int(t_idx))] = idx
        same_cell_hour_lookup.setdefault((cell_id, int(hour_value)), []).append(idx)
        same_cell_indices.setdefault(cell_id, []).append(idx)

    for idx, (cell_id, t_idx, hour_value) in enumerate(zip(cell_ids.tolist(), time_index.tolist(), hour.tolist())):
        for offset in periodic_offsets:
            matched_idx = cell_time_lookup.get((cell_id, int(t_idx + offset)))
            if matched_idx is not None:
                positives[idx].add(int(matched_idx))

        for candidate_idx in same_cell_hour_lookup.get((cell_id, int(hour_value)), []):
            if candidate_idx != idx:
                positives[idx].add(int(candidate_idx))

        for candidate_idx in same_cell_indices.get(cell_id, []):
            if candidate_idx == idx:
                continue
            gap = abs(int(time_index[candidate_idx]) - int(t_idx))
            if gap == 0 or gap > nearby_radius:
                continue
            denom = float(current_norm[idx] * current_norm[candidate_idx])
            similarity = 0.0 if denom <= 0.0 else float(
                np.dot(current_state[idx], current_state[candidate_idx]) / denom
            )
            if similarity >= similarity_threshold:
                positives[idx].add(int(candidate_idx))

    return [
        np.asarray(sorted(candidate_set), dtype=np.int64)
        for candidate_set in positives
    ]


def load_dynamic_feature_bundle(
    static_embeddings_path: Path,
    static_region_ids_path: Path,
    dynamic_windows_path: Path,
    time_context_path: Path,
    sample_index_path: Path,
    positive_mask_path: Optional[Path] = None,
    current_state_path: Optional[Path] = None,
    periodic_offsets: tuple[int, ...] = (-24, 24),
    nearby_radius: int = 3,
    similarity_threshold: float = 0.8,
    max_samples: int = 0,
) -> DynamicFeatureBundle:
    static_embeddings, static_cell_ids = load_static_embedding_mapping(
        embeddings_path=static_embeddings_path,
        region_ids_path=static_region_ids_path,
    )
    static_lookup = {cell_id: idx for idx, cell_id in enumerate(static_cell_ids.tolist())}

    dynamic_windows = np.load(dynamic_windows_path)
    time_context = np.load(time_context_path)
    if dynamic_windows.ndim != 3:
        raise ValueError("Dynamic windows must have shape [N, W, D_dyn].")
    if time_context.ndim != 2:
        raise ValueError("Time context must have shape [N, D_time].")
    if dynamic_windows.shape[0] != time_context.shape[0]:
        raise ValueError("Dynamic windows and time context must have the same sample count.")

    index_df = pd.read_csv(sample_index_path)
    region_id_column = _resolve_region_id_column(index_df, "Sample index file")
    if index_df.shape[0] != dynamic_windows.shape[0]:
        raise ValueError("Sample index rows must match dynamic window sample count.")

    cell_ids = index_df[region_id_column].astype(str).to_numpy()
    time_index, timestamps = _resolve_time_index(index_df)
    hour = _resolve_hour(index_df, time_index)
    current_state = _resolve_current_dynamic_state(dynamic_windows, current_state_path)
    if current_state.shape[0] != dynamic_windows.shape[0]:
        raise ValueError("Current state row count must match dynamic windows.")

    aligned_static_embeddings = np.empty((cell_ids.shape[0], static_embeddings.shape[1]), dtype=np.float32)
    for row_idx, cell_id in enumerate(cell_ids.tolist()):
        static_idx = static_lookup.get(cell_id)
        if static_idx is None:
            raise ValueError(f"Sample region id not found in static embeddings mapping: {cell_id}")
        aligned_static_embeddings[row_idx] = np.asarray(static_embeddings[static_idx], dtype=np.float32)

    if positive_mask_path is not None:
        dense_mask = np.load(positive_mask_path)
        if dense_mask.shape != (cell_ids.shape[0], cell_ids.shape[0]):
            raise ValueError("Positive mask file must have shape [N, N].")
        positive_pairs = [
            np.flatnonzero(np.asarray(dense_mask[row_idx], dtype=np.bool_)).astype(np.int64)
            for row_idx in range(dense_mask.shape[0])
        ]
    else:
        positive_pairs = _build_positive_pairs(
            cell_ids=cell_ids,
            time_index=time_index,
            hour=hour,
            current_state=current_state,
            periodic_offsets=periodic_offsets,
            nearby_radius=nearby_radius,
            similarity_threshold=similarity_threshold,
        )

    if max_samples > 0:
        selected = np.arange(min(max_samples, dynamic_windows.shape[0]), dtype=np.int64)
        global_to_local = {int(global_idx): local_idx for local_idx, global_idx in enumerate(selected.tolist())}
        positive_pairs = [
            np.asarray(
                [
                    global_to_local[int(pos_idx)]
                    for pos_idx in positive_pairs[int(global_idx)].tolist()
                    if int(pos_idx) in global_to_local
                ],
                dtype=np.int64,
            )
            for global_idx in selected.tolist()
        ]
        aligned_static_embeddings = aligned_static_embeddings[selected]
        dynamic_windows = dynamic_windows[selected]
        time_context = time_context[selected]
        cell_ids = cell_ids[selected]
        time_index = time_index[selected]
        if timestamps is not None:
            timestamps = timestamps[selected]

    return DynamicFeatureBundle(
        static_embeddings=np.asarray(aligned_static_embeddings, dtype=np.float32),
        dynamic_windows=np.asarray(dynamic_windows, dtype=np.float32),
        time_context=np.asarray(time_context, dtype=np.float32),
        positive_pairs=positive_pairs,
        cell_ids=cell_ids,
        time_index=np.asarray(time_index, dtype=np.int64),
        timestamps=timestamps,
    )


def load_hourly_dynamic_feature_bundle(
    static_embeddings_path: Path,
    static_region_ids_path: Path,
    hourly_state_path: Path,
    hourly_region_ids_path: Path,
    time_context_path: Path,
    time_index_path: Path,
    window_length: int = 24,
    periodic_offsets: tuple[int, ...] = (-24, 24),
    nearby_radius: int = 3,
    similarity_threshold: float = 0.8,
    max_samples: int = 0,
) -> HourlyDynamicFeatureBundle:
    static_embeddings, static_cell_ids = load_static_embedding_mapping(
        embeddings_path=static_embeddings_path,
        region_ids_path=static_region_ids_path,
    )
    static_lookup = {cell_id: idx for idx, cell_id in enumerate(static_cell_ids.tolist())}

    hourly_state = np.load(hourly_state_path, mmap_mode="r")
    if hourly_state.ndim != 3:
        raise ValueError("Hourly state must have shape [T, N_region, D_dyn].")
    time_context = np.load(time_context_path)
    if time_context.ndim != 2:
        raise ValueError("Time context must have shape [T, D_time].")
    if time_context.shape[0] != hourly_state.shape[0]:
        raise ValueError("Time context rows must match hourly state time steps.")

    hourly_cell_ids = np.load(hourly_region_ids_path, allow_pickle=True).astype(str)
    if hourly_cell_ids.shape[0] != hourly_state.shape[1]:
        raise ValueError("Hourly region id count must match hourly state region axis.")

    time_index_df = pd.read_csv(time_index_path)
    if "time_index" not in time_index_df.columns:
        raise ValueError("time_index.csv must contain a 'time_index' column.")
    if time_index_df.shape[0] != hourly_state.shape[0]:
        raise ValueError("time_index.csv rows must match hourly state time steps.")
    time_index = time_index_df["time_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(time_index, np.arange(hourly_state.shape[0], dtype=np.int64)):
        raise ValueError("time_index.csv must be continuous from 0 to T-1.")

    if window_length <= 0:
        raise ValueError("window_length must be positive.")
    if hourly_state.shape[0] <= window_length:
        raise ValueError("Hourly state does not contain enough time steps for the requested window_length.")

    aligned_static_embeddings = np.empty(
        (hourly_cell_ids.shape[0], static_embeddings.shape[1]),
        dtype=np.float32,
    )
    for row_idx, cell_id in enumerate(hourly_cell_ids.tolist()):
        static_idx = static_lookup.get(cell_id)
        if static_idx is None:
            raise ValueError(f"Hourly region id not found in static embeddings mapping: {cell_id}")
        aligned_static_embeddings[row_idx] = np.asarray(static_embeddings[static_idx], dtype=np.float32)

    timestamps = None
    if "date" in time_index_df.columns and "hour" in time_index_df.columns:
        timestamps = (
            time_index_df["date"].astype(str)
            + " "
            + time_index_df["hour"].astype(int).astype(str).str.zfill(2)
            + ":00"
        ).to_numpy()

    return HourlyDynamicFeatureBundle(
        static_embeddings=aligned_static_embeddings,
        hourly_state=hourly_state,
        time_context=np.asarray(time_context, dtype=np.float32),
        cell_ids=hourly_cell_ids,
        time_index=time_index,
        window_length=window_length,
        nearby_radius=nearby_radius,
        similarity_threshold=similarity_threshold,
        periodic_offsets=periodic_offsets,
        max_samples=max_samples,
        timestamps=timestamps,
    )


def load_daily_dynamic_feature_bundle(
    static_embeddings_path: Path,
    static_region_ids_path: Path,
    daily_state_path: Path,
    daily_region_ids_path: Path,
    day_context_path: Path,
    date_index_path: Path,
    window_length: int = 7,
    periodic_offsets: tuple[int, ...] = (-7, 7),
    nearby_radius: int = 3,
    similarity_threshold: float = 0.8,
    max_samples: int = 0,
    same_day_of_month_positives: bool = False,
    same_day_of_week_positives: bool = False,
) -> DailyDynamicFeatureBundle:
    static_embeddings, static_cell_ids = load_static_embedding_mapping(
        embeddings_path=static_embeddings_path,
        region_ids_path=static_region_ids_path,
    )
    static_lookup = {cell_id: idx for idx, cell_id in enumerate(static_cell_ids.tolist())}

    daily_state = np.load(daily_state_path, mmap_mode="r")
    if daily_state.ndim != 3:
        raise ValueError("Daily state must have shape [T_day, N_region, D_dyn].")
    day_context = np.load(day_context_path)
    if day_context.ndim != 2:
        raise ValueError("Day context must have shape [T_day, D_time].")
    if day_context.shape[0] != daily_state.shape[0]:
        raise ValueError("Day context rows must match daily state time steps.")

    daily_region_ids = np.load(daily_region_ids_path, allow_pickle=True).astype(str)
    if daily_region_ids.shape[0] != daily_state.shape[1]:
        raise ValueError("Daily region id count must match daily state region axis.")

    date_index_df = pd.read_csv(date_index_path)
    if "time_index" not in date_index_df.columns:
        raise ValueError("date_index.csv must contain a 'time_index' column.")
    if date_index_df.shape[0] != daily_state.shape[0]:
        raise ValueError("date_index.csv rows must match daily state time steps.")
    time_index = date_index_df["time_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(time_index, np.arange(daily_state.shape[0], dtype=np.int64)):
        raise ValueError("date_index.csv must be continuous from 0 to T-1.")

    if window_length <= 0:
        raise ValueError("window_length must be positive.")
    if daily_state.shape[0] <= window_length:
        raise ValueError("Daily state does not contain enough time steps for the requested window_length.")

    aligned_static_embeddings = np.empty(
        (daily_region_ids.shape[0], static_embeddings.shape[1]),
        dtype=np.float32,
    )
    for row_idx, cell_id in enumerate(daily_region_ids.tolist()):
        static_idx = static_lookup.get(cell_id)
        if static_idx is None:
            raise ValueError(f"Daily region id not found in static embeddings mapping: {cell_id}")
        aligned_static_embeddings[row_idx] = np.asarray(static_embeddings[static_idx], dtype=np.float32)

    timestamps = None
    day_of_month = None
    day_of_week = None
    if "date" in date_index_df.columns:
        date_text = date_index_df["date"].astype(str).str.zfill(8)
        timestamps = date_text.to_numpy()
        day_of_month = date_text.str.slice(6, 8).astype(int).to_numpy(dtype=np.int64)
    if "day_of_week" in date_index_df.columns:
        day_of_week = date_index_df["day_of_week"].to_numpy(dtype=np.int64)

    return DailyDynamicFeatureBundle(
        static_embeddings=aligned_static_embeddings,
        daily_state=daily_state,
        day_context=np.asarray(day_context, dtype=np.float32),
        cell_ids=daily_region_ids,
        time_index=time_index,
        window_length=window_length,
        nearby_radius=nearby_radius,
        similarity_threshold=similarity_threshold,
        periodic_offsets=periodic_offsets,
        max_samples=max_samples,
        timestamps=timestamps,
        day_of_month=day_of_month,
        day_of_week=day_of_week,
        same_day_of_month_positives=same_day_of_month_positives,
        same_day_of_week_positives=same_day_of_week_positives,
    )


def load_monthly_dynamic_feature_bundle(
    static_embeddings_path: Path,
    static_region_ids_path: Path,
    monthly_state_path: Path,
    monthly_region_ids_path: Path,
    month_context_path: Path,
    month_index_path: Path,
    window_length: int = 6,
    periodic_offsets: tuple[int, ...] = (),
    nearby_radius: int = 2,
    similarity_threshold: float = 0.7,
    max_samples: int = 0,
) -> MonthlyDynamicFeatureBundle:
    static_embeddings, static_cell_ids = load_static_embedding_mapping(
        embeddings_path=static_embeddings_path,
        region_ids_path=static_region_ids_path,
    )
    static_lookup = {cell_id: idx for idx, cell_id in enumerate(static_cell_ids.tolist())}

    monthly_state = np.load(monthly_state_path, mmap_mode="r")
    if monthly_state.ndim != 3:
        raise ValueError("Monthly state must have shape [T_month, N_region, D_dyn].")
    month_context = np.load(month_context_path)
    if month_context.ndim != 2:
        raise ValueError("Month context must have shape [T_month, D_time].")
    if month_context.shape[0] != monthly_state.shape[0]:
        raise ValueError("Month context rows must match monthly state time steps.")

    monthly_region_ids = np.load(monthly_region_ids_path, allow_pickle=True).astype(str)
    if monthly_region_ids.shape[0] != monthly_state.shape[1]:
        raise ValueError("Monthly region id count must match monthly state region axis.")

    month_index_df = pd.read_csv(month_index_path)
    if "time_index" not in month_index_df.columns:
        raise ValueError("month_index.csv must contain a 'time_index' column.")
    if month_index_df.shape[0] != monthly_state.shape[0]:
        raise ValueError("month_index.csv rows must match monthly state time steps.")
    time_index = month_index_df["time_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(time_index, np.arange(monthly_state.shape[0], dtype=np.int64)):
        raise ValueError("month_index.csv must be continuous from 0 to T-1.")

    if window_length <= 0:
        raise ValueError("window_length must be positive.")
    if monthly_state.shape[0] <= window_length:
        raise ValueError("Monthly state does not contain enough time steps for the requested window_length.")

    aligned_static_embeddings = np.empty(
        (monthly_region_ids.shape[0], static_embeddings.shape[1]),
        dtype=np.float32,
    )
    for row_idx, cell_id in enumerate(monthly_region_ids.tolist()):
        static_idx = static_lookup.get(cell_id)
        if static_idx is None:
            raise ValueError(f"Monthly region id not found in static embeddings mapping: {cell_id}")
        aligned_static_embeddings[row_idx] = np.asarray(static_embeddings[static_idx], dtype=np.float32)

    timestamps = None
    if "month" in month_index_df.columns:
        timestamps = month_index_df["month"].astype(str).to_numpy()

    return MonthlyDynamicFeatureBundle(
        static_embeddings=aligned_static_embeddings,
        monthly_state=monthly_state,
        month_context=np.asarray(month_context, dtype=np.float32),
        cell_ids=monthly_region_ids,
        time_index=time_index,
        window_length=window_length,
        nearby_radius=nearby_radius,
        similarity_threshold=similarity_threshold,
        periodic_offsets=periodic_offsets,
        max_samples=max_samples,
        timestamps=timestamps,
    )


def split_dynamic_indices(
    num_samples: int,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(num_samples, dtype=np.int64)
    rng.shuffle(indices)
    val_size = int(round(num_samples * val_ratio))
    val_size = min(max(1, val_size), max(1, num_samples - 1))
    val_indices = np.sort(indices[:val_size])
    train_indices = np.sort(indices[val_size:])
    return train_indices, val_indices


def create_mock_dynamic_bundle(
    sample_count: int,
    static_dim: int,
    window_length: int,
    dynamic_dim: int,
    time_dim: int,
    seed: int = 42,
) -> DynamicFeatureBundle:
    rng = np.random.default_rng(seed)
    cell_count = max(4, sample_count // 6)
    cell_ids = np.asarray([f"cell_{idx % cell_count:04d}" for idx in range(sample_count)], dtype=object)
    time_index = np.asarray([idx // cell_count for idx in range(sample_count)], dtype=np.int64)
    static_bank = rng.normal(size=(cell_count, static_dim)).astype(np.float32)
    static_lookup = {f"cell_{idx:04d}": static_bank[idx] for idx in range(cell_count)}

    dynamic_windows = rng.normal(size=(sample_count, window_length, dynamic_dim)).astype(np.float32)
    time_context = rng.normal(size=(sample_count, time_dim)).astype(np.float32)
    static_embeddings = np.vstack([static_lookup[str(cell_id)] for cell_id in cell_ids.tolist()]).astype(np.float32)

    current_state = dynamic_windows[:, -1, :]
    hour = np.mod(time_index, 24).astype(np.int64)
    positive_pairs = _build_positive_pairs(
        cell_ids=cell_ids.astype(str),
        time_index=time_index,
        hour=hour,
        current_state=current_state,
        periodic_offsets=(-24, 24),
        nearby_radius=2,
        similarity_threshold=-1.0,
    )

    return DynamicFeatureBundle(
        static_embeddings=static_embeddings,
        dynamic_windows=dynamic_windows,
        time_context=time_context,
        positive_pairs=positive_pairs,
        cell_ids=cell_ids.astype(str),
        time_index=time_index,
        timestamps=None,
    )
