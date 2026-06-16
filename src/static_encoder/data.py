from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import PathConfig


@dataclass
class FeatureBundle:
    cell_ids: np.ndarray
    rs_features: np.ndarray
    osm_features: np.ndarray
    street_features: np.ndarray
    poi_local_ids: np.ndarray
    has_poi: np.ndarray
    has_street: np.ndarray

    @property
    def num_samples(self) -> int:
        return int(self.cell_ids.shape[0])


class IndexDataset(Dataset):
    def __init__(self, indices: np.ndarray) -> None:
        self.indices = indices.astype(np.int64, copy=False)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int) -> int:
        return int(self.indices[idx])


class BatchCollator:
    def __init__(self, bundle: FeatureBundle) -> None:
        self.bundle = bundle

    def __call__(self, batch_indices: Iterable[int]) -> Dict[str, object]:
        indices = np.asarray(list(batch_indices), dtype=np.int64)
        return {
            "global_indices": torch.from_numpy(indices.copy()),
            "cell_ids": self.bundle.cell_ids[indices].tolist(),
            "rs_features": torch.from_numpy(
                np.asarray(self.bundle.rs_features[indices], dtype=np.float32)
            ),
            "osm_features": torch.from_numpy(
                np.asarray(self.bundle.osm_features[indices], dtype=np.float32)
            ),
            "street_features": torch.from_numpy(
                np.asarray(self.bundle.street_features[indices], dtype=np.float32)
            ),
            "poi_local_ids": torch.from_numpy(
                self.bundle.poi_local_ids[indices].astype(np.int64, copy=False)
            ),
            "has_poi": torch.from_numpy(
                self.bundle.has_poi[indices].astype(np.bool_, copy=False)
            ),
            "has_street": torch.from_numpy(
                self.bundle.has_street[indices].astype(np.bool_, copy=False)
            ),
        }


def _load_city_cell_ids(city_assignment_csv: str) -> np.ndarray:
    df = pd.read_csv(city_assignment_csv, usecols=["cell_id"])
    return df["cell_id"].astype(str).to_numpy()


def _validate_poi_alignment(
    city_cell_ids: np.ndarray,
    poi_distribution_csv: str,
) -> None:
    poi_df = pd.read_csv(poi_distribution_csv, usecols=["cell_id"])
    poi_cell_ids = poi_df["cell_id"].astype(str).to_numpy()
    if poi_cell_ids.shape[0] != city_cell_ids.shape[0]:
        raise ValueError(
            "POI distribution row count does not match city_assignment.csv"
        )
    if not np.array_equal(poi_cell_ids, city_cell_ids):
        raise ValueError(
            "POI distribution cell_id order does not match city_assignment.csv"
        )


def _build_poi_local_mapping(
    city_cell_ids: np.ndarray,
    poi_graph_cell_ids_csv: str,
) -> Tuple[np.ndarray, np.ndarray]:
    poi_nodes_df = pd.read_csv(poi_graph_cell_ids_csv)
    poi_graph_cell_ids = poi_nodes_df["cell_id"].astype(str).to_numpy()
    mapping = {cell_id: idx for idx, cell_id in enumerate(poi_graph_cell_ids)}

    poi_local_ids = np.full(city_cell_ids.shape[0], -1, dtype=np.int64)
    has_poi = np.zeros(city_cell_ids.shape[0], dtype=np.bool_)
    for idx, cell_id in enumerate(city_cell_ids.tolist()):
        local_id = mapping.get(cell_id)
        if local_id is not None:
            poi_local_ids[idx] = int(local_id)
            has_poi[idx] = True
    return poi_local_ids, has_poi


def _load_street_features(
    city_cell_ids: np.ndarray,
    street_features_csv: str,
    street_features_npy: str,
) -> Tuple[np.ndarray, np.ndarray]:
    street_df = pd.read_csv(street_features_csv, usecols=["cell_id"])
    street_cell_ids = street_df["cell_id"].astype(str).to_numpy()
    if pd.Index(street_cell_ids).has_duplicates:
        raise ValueError("Street feature CSV contains duplicate cell_id values")

    street_features = np.load(street_features_npy, mmap_mode="r")
    if street_features.shape[0] != street_cell_ids.shape[0]:
        raise ValueError(
            "Street feature row count does not match street feature CSV"
        )
    if street_features.ndim != 2 or street_features.shape[1] != 768:
        raise ValueError("Street feature matrix must have shape [N, 768]")

    city_index = {cell_id: idx for idx, cell_id in enumerate(city_cell_ids.tolist())}
    aligned_features = np.zeros(
        (city_cell_ids.shape[0], street_features.shape[1]),
        dtype=np.float32,
    )
    has_street = np.zeros(city_cell_ids.shape[0], dtype=np.bool_)

    for local_idx, cell_id in enumerate(street_cell_ids.tolist()):
        global_idx = city_index.get(cell_id)
        if global_idx is None:
            raise ValueError(
                f"Street feature cell_id not found in city_assignment.csv: {cell_id}"
            )
        aligned_features[global_idx] = np.asarray(
            street_features[local_idx],
            dtype=np.float32,
        )
        has_street[global_idx] = True

    return aligned_features, has_street


def load_feature_bundle(
    paths: PathConfig,
    max_samples: int = 0,
) -> FeatureBundle:
    cell_ids = _load_city_cell_ids(str(paths.city_assignment_csv))
    _validate_poi_alignment(cell_ids, str(paths.poi_distribution_csv))

    rs_features = np.load(paths.rs_features_npy, mmap_mode="r")
    osm_features = np.load(paths.osm_features_npy, mmap_mode="r")
    if rs_features.shape[0] != cell_ids.shape[0]:
        raise ValueError("Remote sensing feature rows do not match cell_ids")
    if osm_features.shape[0] != cell_ids.shape[0]:
        raise ValueError("OSM feature rows do not match cell_ids")
    street_features, has_street = _load_street_features(
        city_cell_ids=cell_ids,
        street_features_csv=str(paths.street_features_csv),
        street_features_npy=str(paths.street_features_npy),
    )

    poi_local_ids, has_poi = _build_poi_local_mapping(
        city_cell_ids=cell_ids,
        poi_graph_cell_ids_csv=str(paths.poi_graph_cell_ids_csv),
    )

    if max_samples > 0:
        cell_ids = cell_ids[:max_samples]
        rs_features = rs_features[:max_samples]
        osm_features = osm_features[:max_samples]
        street_features = street_features[:max_samples]
        poi_local_ids = poi_local_ids[:max_samples]
        has_poi = has_poi[:max_samples]
        has_street = has_street[:max_samples]

    return FeatureBundle(
        cell_ids=cell_ids,
        rs_features=rs_features,
        osm_features=osm_features,
        street_features=street_features,
        poi_local_ids=poi_local_ids,
        has_poi=has_poi,
        has_street=has_street,
    )


def split_indices(
    num_samples: int,
    val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(num_samples, dtype=np.int64)
    rng.shuffle(indices)
    val_size = int(round(num_samples * val_ratio))
    val_size = max(1, val_size)
    val_indices = np.sort(indices[:val_size])
    train_indices = np.sort(indices[val_size:])
    return train_indices, val_indices


def _available_modalities(
    has_poi: bool,
    has_street: bool,
) -> List[int]:
    available: List[int] = []
    if has_poi:
        available.append(0)
    available.extend([1, 2])
    if has_street:
        available.append(3)
    return available


def available_modality_mask(
    has_poi: np.ndarray,
    has_street: np.ndarray,
) -> np.ndarray:
    mask = np.zeros((has_poi.shape[0], 4), dtype=np.float32)
    mask[:, 1] = 1.0
    mask[:, 2] = 1.0
    mask[has_poi, 0] = 1.0
    mask[has_street, 3] = 1.0
    return mask


def sample_dual_view_keep_masks(
    has_poi: np.ndarray,
    has_street: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    keep0 = np.zeros((has_poi.shape[0], 4), dtype=np.float32)
    keep1 = np.zeros((has_poi.shape[0], 4), dtype=np.float32)

    for row_idx, (poi_flag, street_flag) in enumerate(
        zip(has_poi.tolist(), has_street.tolist())
    ):
        available = _available_modalities(
            has_poi=bool(poi_flag),
            has_street=bool(street_flag),
        )
        subset_size = int(rng.integers(1, len(available)))
        selected = rng.choice(
            np.asarray(available, dtype=np.int64),
            size=subset_size,
            replace=False,
        )
        keep0[row_idx, selected] = 1.0
        for mod_idx in available:
            if keep0[row_idx, mod_idx] == 0.0:
                keep1[row_idx, mod_idx] = 1.0

    return torch.from_numpy(keep0), torch.from_numpy(keep1)
