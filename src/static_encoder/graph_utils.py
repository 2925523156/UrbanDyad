from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


BLOCK_RE = re.compile(
    r"block_r(?P<row_start>\d{6})_(?P<row_end>\d{6})_c(?P<col_start>\d{6})_(?P<col_end>\d{6})\.npy"
)


@dataclass
class CSRGraph:
    indptr: np.ndarray
    indices: np.ndarray
    weights: np.ndarray
    num_nodes: int
    threshold: float

    def row(self, node_id: int) -> Tuple[np.ndarray, np.ndarray]:
        start = int(self.indptr[node_id])
        end = int(self.indptr[node_id + 1])
        return self.indices[start:end], self.weights[start:end]


@dataclass
class SampledSubgraph:
    node_ids: np.ndarray
    edge_index: np.ndarray
    edge_weight: np.ndarray
    seed_positions: np.ndarray


def parse_block_filename(path: Path) -> Tuple[int, int, int, int]:
    match = BLOCK_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Unexpected block filename: {path.name}")
    row_start = int(match.group("row_start"))
    row_end = int(match.group("row_end")) + 1
    col_start = int(match.group("col_start"))
    col_end = int(match.group("col_end")) + 1
    return row_start, row_end, col_start, col_end


def iter_block_files(block_dir: Path, max_block_files: int = 0) -> List[Path]:
    files = sorted(block_dir.glob("block_r*.npy"))
    if max_block_files > 0:
        files = files[:max_block_files]
    if not files:
        raise FileNotFoundError(f"No block files found in: {block_dir}")
    return files


def _save_graph_meta(
    cache_dir: Path,
    num_nodes: int,
    edge_count: int,
    threshold: float,
    source_dir: Path,
) -> None:
    meta = {
        "num_nodes": int(num_nodes),
        "edge_count": int(edge_count),
        "threshold": float(threshold),
        "source_dir": str(source_dir),
        "format": "csr",
        "indptr_file": "indptr.npy",
        "indices_file": "indices.npy",
        "weights_file": "weights.npy",
    }
    with (cache_dir / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)


def build_threshold_csr_graph(
    block_dir: Path,
    num_nodes: int,
    threshold: float,
    cache_dir: Path,
    max_block_files: int = 0,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    block_files = iter_block_files(block_dir, max_block_files=max_block_files)

    row_counts = np.zeros(num_nodes, dtype=np.int64)
    for block_path in block_files:
        row_start, row_end, _, _ = parse_block_filename(block_path)
        block = np.load(block_path, mmap_mode="r")
        mask = block >= threshold
        row_counts[row_start:row_end] += mask.sum(axis=1, dtype=np.int64)

    indptr = np.empty(num_nodes + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(row_counts, out=indptr[1:])
    edge_count = int(indptr[-1])

    indices = np.lib.format.open_memmap(
        cache_dir / "indices.npy",
        mode="w+",
        dtype=np.int32,
        shape=(edge_count,),
    )
    weights = np.lib.format.open_memmap(
        cache_dir / "weights.npy",
        mode="w+",
        dtype=np.float32,
        shape=(edge_count,),
    )
    np.save(cache_dir / "indptr.npy", indptr)

    write_ptr = indptr[:-1].copy()
    for block_path in block_files:
        row_start, row_end, col_start, _ = parse_block_filename(block_path)
        block = np.load(block_path, mmap_mode="r")
        for local_row in range(block.shape[0]):
            values = block[local_row]
            keep = np.flatnonzero(values >= threshold)
            if keep.size == 0:
                continue
            row_id = row_start + local_row
            start = int(write_ptr[row_id])
            end = start + int(keep.size)
            indices[start:end] = (keep + col_start).astype(np.int32, copy=False)
            weights[start:end] = values[keep].astype(np.float32, copy=False)
            write_ptr[row_id] = end

    indices.flush()
    weights.flush()
    _save_graph_meta(
        cache_dir=cache_dir,
        num_nodes=num_nodes,
        edge_count=edge_count,
        threshold=threshold,
        source_dir=block_dir,
    )
    return cache_dir


def load_csr_graph(cache_dir: Path) -> CSRGraph:
    with (cache_dir / "meta.json").open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    indptr = np.load(cache_dir / meta["indptr_file"], mmap_mode="r")
    indices = np.load(cache_dir / meta["indices_file"], mmap_mode="r")
    weights = np.load(cache_dir / meta["weights_file"], mmap_mode="r")
    return CSRGraph(
        indptr=indptr,
        indices=indices,
        weights=weights,
        num_nodes=int(meta["num_nodes"]),
        threshold=float(meta["threshold"]),
    )


def _random_walk_from_seed(
    graph: CSRGraph,
    seed: int,
    num_walks: int,
    walk_length: int,
    restart_prob: float,
    rng: np.random.Generator,
) -> List[int]:
    visited: List[int] = [seed]
    for _ in range(num_walks):
        current = seed
        for _ in range(walk_length):
            if rng.random() < restart_prob:
                current = seed
            else:
                neighbors, weights = graph.row(current)
                if neighbors.size == 0:
                    current = seed
                else:
                    probs = weights.astype(np.float64, copy=False)
                    probs = probs / probs.sum()
                    current = int(rng.choice(neighbors, p=probs))
            visited.append(current)
    return visited


def sample_random_walk_subgraph(
    graph: CSRGraph,
    seed_nodes: np.ndarray,
    num_walks: int,
    walk_length: int,
    restart_prob: float,
    rng: np.random.Generator,
) -> SampledSubgraph:
    unique_seeds = np.unique(seed_nodes.astype(np.int32, copy=False))
    visited = set(unique_seeds.tolist())
    for seed in unique_seeds.tolist():
        visited.update(
            _random_walk_from_seed(
                graph=graph,
                seed=seed,
                num_walks=num_walks,
                walk_length=walk_length,
                restart_prob=restart_prob,
                rng=rng,
            )
        )

    node_ids = np.array(sorted(visited), dtype=np.int32)
    node_lookup = np.full(graph.num_nodes, -1, dtype=np.int64)
    node_lookup[node_ids] = np.arange(node_ids.shape[0], dtype=np.int64)

    row_parts: List[np.ndarray] = []
    col_parts: List[np.ndarray] = []
    weight_parts: List[np.ndarray] = []
    for local_row, node_id in enumerate(node_ids.tolist()):
        neighbors, weights = graph.row(node_id)
        mapped = node_lookup[neighbors]
        keep = mapped >= 0
        if not np.any(keep):
            continue
        row_parts.append(
            np.full(int(keep.sum()), local_row, dtype=np.int64)
        )
        col_parts.append(mapped[keep].astype(np.int64, copy=False))
        weight_parts.append(weights[keep].astype(np.float32, copy=False))

    if row_parts:
        edge_index = np.vstack(
            [np.concatenate(row_parts), np.concatenate(col_parts)]
        )
        edge_weight = np.concatenate(weight_parts).astype(np.float32, copy=False)
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_weight = np.empty((0,), dtype=np.float32)

    seed_positions = node_lookup[seed_nodes.astype(np.int32, copy=False)]
    return SampledSubgraph(
        node_ids=node_ids,
        edge_index=edge_index,
        edge_weight=edge_weight,
        seed_positions=seed_positions.astype(np.int64, copy=False),
    )

