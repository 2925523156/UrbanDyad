from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import default_config
from graph_utils import build_threshold_csr_graph


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a thresholded CSR graph cache for POI similarity blocks."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Similarity threshold for keeping POI edges.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="",
        help="Output cache directory. Defaults to config path.",
    )
    parser.add_argument(
        "--max-block-files",
        type=int,
        default=0,
        help="Optional debug limit for the number of block files to scan.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    config = default_config()
    threshold = float(args.threshold)
    max_block_files = int(args.max_block_files)
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
    else:
        cache_dir = config.paths.poi_graph_cache_dir

    poi_nodes_df = pd.read_csv(config.paths.poi_graph_cell_ids_csv)
    num_nodes = int(len(poi_nodes_df))

    print(f"POI nodes: {num_nodes}")
    print(f"Threshold: {threshold}")
    print(f"Blocks: {config.paths.poi_similarity_blocks_dir}")
    print(f"Cache dir: {cache_dir}")
    if max_block_files > 0:
        print(f"Debug mode: using first {max_block_files} block files only")

    build_threshold_csr_graph(
        block_dir=config.paths.poi_similarity_blocks_dir,
        num_nodes=num_nodes,
        threshold=threshold,
        cache_dir=cache_dir,
        max_block_files=max_block_files,
    )
    print("Graph cache build finished.")


if __name__ == "__main__":
    main()
