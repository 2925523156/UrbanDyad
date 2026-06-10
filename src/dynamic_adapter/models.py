from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

from graph_utils import CSRGraph, sample_random_walk_subgraph


class FeatureProjector(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dim = max(output_dim, min(input_dim, output_dim * 2))
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class POIRandomWalkGATEncoder(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        node_dim: int,
        fusion_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, node_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        self.gat = GATv2Conv(
            in_channels=node_dim,
            out_channels=node_dim,
            heads=4,
            concat=False,
            dropout=dropout,
            edge_dim=1,
            add_self_loops=False,
        )
        self.norm = nn.LayerNorm(node_dim)
        self.output_proj = FeatureProjector(
            input_dim=node_dim,
            output_dim=fusion_dim,
            dropout=dropout,
        )

    def forward(
        self,
        seed_nodes: torch.Tensor,
        graph: CSRGraph,
        num_walks: int,
        walk_length: int,
        restart_prob: float,
        rng: np.random.Generator,
        device: torch.device,
    ) -> torch.Tensor:
        if seed_nodes.numel() == 0:
            return torch.empty(
                (0, self.output_proj.net[-1].out_features),
                device=device,
                dtype=self.embedding.weight.dtype,
            )

        sampled = sample_random_walk_subgraph(
            graph=graph,
            seed_nodes=seed_nodes.detach().cpu().numpy().astype(np.int32),
            num_walks=num_walks,
            walk_length=walk_length,
            restart_prob=restart_prob,
            rng=rng,
        )
        node_ids = torch.from_numpy(sampled.node_ids.astype(np.int64)).to(device)
        edge_index = torch.from_numpy(
            sampled.edge_index.astype(np.int64, copy=False)
        ).to(device)
        edge_weight = torch.from_numpy(
            sampled.edge_weight.astype(np.float32, copy=False)
        ).to(device)
        seed_positions = torch.from_numpy(
            sampled.seed_positions.astype(np.int64, copy=False)
        ).to(device)

        node_features = self.embedding(node_ids)
        node_features = self.gat(
            node_features,
            edge_index=edge_index,
            edge_attr=edge_weight.unsqueeze(-1),
        )
        node_features = self.norm(node_features)
        node_features = F.gelu(node_features)
        node_features = self.output_proj(node_features)
        return node_features[seed_positions]


class TriModalFusionModel(nn.Module):
    def __init__(
        self,
        num_poi_nodes: int,
        use_poi: bool = True,
        poi_node_dim: int = 128,
        fusion_dim: int = 256,
        contrastive_dim: int = 128,
        transformer_depth: int = 1,
        num_heads: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.use_poi = use_poi
        self.fusion_dim = fusion_dim
        self.contrastive_dim = contrastive_dim
        self.poi_encoder = POIRandomWalkGATEncoder(
            num_nodes=num_poi_nodes,
            node_dim=poi_node_dim,
            fusion_dim=fusion_dim,
            dropout=dropout,
        )
        self.rs_projector = FeatureProjector(768, fusion_dim, dropout=dropout)
        self.osm_projector = FeatureProjector(2304, fusion_dim, dropout=dropout)
        self.street_projector = FeatureProjector(768, fusion_dim, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_depth,
        )
        self.modality_embedding = nn.Parameter(
            torch.zeros(1, 4, fusion_dim)
        )
        nn.init.normal_(self.modality_embedding, mean=0.0, std=0.02)
        self.output_norm = nn.LayerNorm(fusion_dim)
        self.contrastive_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, contrastive_dim),
        )

    def encode_modalities(
        self,
        rs_features: torch.Tensor,
        osm_features: torch.Tensor,
        street_features: torch.Tensor,
        poi_local_ids: torch.Tensor,
        has_poi: torch.Tensor,
        has_street: torch.Tensor,
        graph: CSRGraph,
        num_walks: int,
        walk_length: int,
        restart_prob: float,
        rng: np.random.Generator,
    ) -> torch.Tensor:
        batch_size = rs_features.shape[0]
        device = rs_features.device
        tokens = torch.zeros(
            (batch_size, 4, self.fusion_dim),
            device=device,
            dtype=rs_features.dtype,
        )
        tokens[:, 1, :] = self.rs_projector(rs_features)
        tokens[:, 2, :] = self.osm_projector(osm_features)
        if bool(has_street.any()):
            street_token_batch = self.street_projector(street_features[has_street])
            tokens[has_street, 3, :] = street_token_batch.to(dtype=tokens.dtype)

        if self.use_poi and bool(has_poi.any()):
            if graph is None:
                raise ValueError("POI graph is required when POI is enabled.")
            poi_token_batch = self.poi_encoder(
                seed_nodes=poi_local_ids[has_poi],
                graph=graph,
                num_walks=num_walks,
                walk_length=walk_length,
                restart_prob=restart_prob,
                rng=rng,
                device=device,
            )
            poi_token_batch = poi_token_batch.to(dtype=tokens.dtype)
            tokens[has_poi, 0, :] = poi_token_batch
        return tokens

    def fuse_tokens(
        self,
        tokens: torch.Tensor,
        keep_mask: torch.Tensor,
        return_raw: bool = False,
    ) -> torch.Tensor:
        keep_mask = keep_mask.to(device=tokens.device, dtype=tokens.dtype)
        x = (tokens + self.modality_embedding) * keep_mask.unsqueeze(-1)
        x = self.transformer(x)
        pooled = (x * keep_mask.unsqueeze(-1)).sum(dim=1)
        denom = keep_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = pooled / denom
        pooled = self.output_norm(pooled)
        if return_raw:
            return pooled
        embedding = self.contrastive_head(pooled)
        return F.normalize(embedding, dim=-1)


class DynamicSequenceEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(x)
        return self.output_norm(hidden[-1])


class TimeContextEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DynamicConditionEncoder(nn.Module):
    def __init__(
        self,
        sequence_dim: int,
        time_dim: int,
        condition_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        fused_dim = sequence_dim + time_dim
        self.net = nn.Sequential(
            nn.Linear(fused_dim, condition_dim),
            nn.LayerNorm(condition_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(condition_dim, condition_dim),
            nn.LayerNorm(condition_dim),
        )

    def forward(
        self,
        sequence_features: torch.Tensor,
        time_features: torch.Tensor,
    ) -> torch.Tensor:
        fused = torch.cat([sequence_features, time_features], dim=-1)
        return self.net(fused)


class FiLMAdapter(nn.Module):
    def __init__(
        self,
        condition_dim: int,
        target_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.gamma_head = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, target_dim),
        )
        self.beta_head = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, target_dim),
        )
        nn.init.zeros_(self.gamma_head[-1].weight)
        nn.init.zeros_(self.gamma_head[-1].bias)
        nn.init.zeros_(self.beta_head[-1].weight)
        nn.init.zeros_(self.beta_head[-1].bias)

    def forward(
        self,
        static_embedding: torch.Tensor,
        condition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gamma = 1.0 + self.gamma_head(condition)
        beta = self.beta_head(condition)
        adapted = gamma * static_embedding + beta
        return adapted, gamma, beta


class DynamicProjectionHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        projection_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.net(x)
        return F.normalize(projected, dim=-1)


class DynamicRegionRepresentationModel(nn.Module):
    def __init__(
        self,
        static_dim: int,
        dynamic_dim: int,
        time_dim: int,
        sequence_hidden_dim: int,
        time_hidden_dim: int,
        condition_dim: int,
        projection_dim: int,
        film_hidden_dim: int,
        gru_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.static_dim = static_dim
        self.sequence_encoder = DynamicSequenceEncoder(
            input_dim=dynamic_dim,
            hidden_dim=sequence_hidden_dim,
            num_layers=gru_layers,
            dropout=dropout,
        )
        self.time_encoder = TimeContextEncoder(
            input_dim=time_dim,
            hidden_dim=time_hidden_dim,
            dropout=dropout,
        )
        self.condition_encoder = DynamicConditionEncoder(
            sequence_dim=sequence_hidden_dim,
            time_dim=time_hidden_dim,
            condition_dim=condition_dim,
            dropout=dropout,
        )
        self.film = FiLMAdapter(
            condition_dim=condition_dim,
            target_dim=static_dim,
            hidden_dim=film_hidden_dim,
            dropout=dropout,
        )
        self.output_norm = nn.LayerNorm(static_dim)
        self.projection_head = DynamicProjectionHead(
            input_dim=static_dim,
            projection_dim=projection_dim,
            dropout=dropout,
        )

    def forward(
        self,
        static_embedding: torch.Tensor,
        dynamic_window: torch.Tensor,
        time_context: torch.Tensor,
        return_aux: bool = True,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        sequence_features = self.sequence_encoder(dynamic_window)
        time_features = self.time_encoder(time_context)
        condition = self.condition_encoder(sequence_features, time_features)
        adapted, gamma, beta = self.film(static_embedding, condition)
        dynamic_representation = self.output_norm(adapted)
        projected = self.projection_head(dynamic_representation)
        if not return_aux:
            return projected
        return {
            "condition": condition,
            "dynamic_representation": dynamic_representation,
            "projected_representation": projected,
            "gamma": gamma,
            "beta": beta,
        }
