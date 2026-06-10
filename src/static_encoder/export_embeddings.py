from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import config_from_dict
from data import BatchCollator, IndexDataset, available_modality_mask, load_feature_bundle
from graph_utils import load_csr_graph
from models import TriModalFusionModel
from utils import ensure_dir, resolve_device, set_seed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export fusion embeddings in city_assignment.csv order."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--export-repeat",
        type=int,
        default=8,
        help="Repeat random-walk inference and average the resulting POI tokens.",
    )
    return parser


def save_embedding_csv(
    output_path: Path,
    cell_ids: np.ndarray,
    embeddings: np.ndarray,
) -> None:
    header = ["cell_id"] + [f"feat_{idx}" for idx in range(embeddings.shape[1])]
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for cell_id, vector in zip(cell_ids.tolist(), embeddings):
            writer.writerow([cell_id, *vector.tolist()])


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = config_from_dict(checkpoint["config"])
    set_seed(args.seed)
    device = resolve_device(config.training.device)

    if args.max_samples > 0:
        config.training.max_samples = args.max_samples

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

    model = TriModalFusionModel(
        num_poi_nodes=num_poi_nodes,
        use_poi=config.model.use_poi,
        poi_node_dim=config.model.poi_node_dim,
        fusion_dim=config.model.fusion_dim,
        contrastive_dim=config.model.contrastive_dim,
        transformer_depth=config.model.transformer_depth,
        num_heads=config.model.num_heads,
        ff_dim=config.model.ff_dim,
        dropout=config.model.dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    collator = BatchCollator(bundle)
    loader = DataLoader(
        IndexDataset(np.arange(bundle.num_samples, dtype=np.int64)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collator,
    )

    embeddings = np.empty(
        (bundle.num_samples, config.model.contrastive_dim),
        dtype=np.float32,
    )
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            rs_features = batch["rs_features"].to(device)
            osm_features = batch["osm_features"].to(device)
            street_features = batch["street_features"].to(device)
            poi_local_ids = batch["poi_local_ids"].to(device)
            has_poi = batch["has_poi"].to(device)
            has_street = batch["has_street"].to(device)
            if not config.model.use_poi:
                has_poi = torch.zeros_like(has_poi, dtype=torch.bool)
            token_sum = torch.zeros(
                (rs_features.shape[0], 4, config.model.fusion_dim),
                device=device,
                dtype=rs_features.dtype,
            )
            for repeat_idx in range(args.export_repeat):
                rng = np.random.default_rng(
                    args.seed + batch_idx * 10_000 + repeat_idx
                )
                token_sum += model.encode_modalities(
                    rs_features=rs_features,
                    osm_features=osm_features,
                    street_features=street_features,
                    poi_local_ids=poi_local_ids,
                    has_poi=has_poi,
                    has_street=has_street,
                    graph=graph,
                    num_walks=config.random_walk.num_walks,
                    walk_length=config.random_walk.walk_length,
                    restart_prob=config.random_walk.restart_prob,
                    rng=rng,
                )
            tokens = token_sum / float(args.export_repeat)
            keep_mask = torch.from_numpy(
                available_modality_mask(
                    has_poi=has_poi.detach().cpu().numpy(),
                    has_street=has_street.detach().cpu().numpy(),
                )
            ).to(device)
            batch_embeddings = model.fuse_tokens(tokens, keep_mask).cpu().numpy()
            global_indices = batch["global_indices"].numpy()
            embeddings[global_indices] = batch_embeddings

    if args.output_dir:
        output_dir = ensure_dir(Path(args.output_dir))
    else:
        output_dir = ensure_dir(checkpoint_path.parent / "exported_embeddings")

    npy_path = output_dir / "fusion_embeddings.npy"
    csv_path = output_dir / "fusion_embeddings.csv"
    np.save(npy_path, embeddings)
    save_embedding_csv(csv_path, bundle.cell_ids, embeddings)

    print(f"Saved embeddings to: {npy_path}")
    print(f"Saved CSV to: {csv_path}")


if __name__ == "__main__":
    main()
