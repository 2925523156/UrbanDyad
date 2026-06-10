from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Type, TypeVar


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def module_root() -> Path:
    return Path(__file__).resolve().parent


@dataclass
class PathConfig:
    project_root: Path = field(default_factory=project_root)
    module_root: Path = field(default_factory=module_root)
    city_assignment_csv: Path = field(
        default_factory=lambda: project_root()
        / "输入数据"
        / "大湾区遥感数据"
        / "gba_cell_tif_outputs_by_city"
        / "city_assignment.csv"
    )
    rs_features_npy: Path = field(
        default_factory=lambda: project_root()
        / "输入数据"
        / "大湾区遥感数据"
        / "gba_cell_tif_outputs_by_city"
        / "remoteclip_features_vit_l_14.npy"
    )
    osm_features_npy: Path = field(
        default_factory=lambda: project_root()
        / "输入数据"
        / "大湾区OSM底图数据"
        / "gba_cell_tif_outputs_by_city"
        / "osm_mae_features_2304.npy"
    )
    street_features_csv: Path = field(
        default_factory=lambda: project_root()
        / "输入数据"
        / "大湾区街景图像"
        / "street_clip_features_768.csv"
    )
    street_features_npy: Path = field(
        default_factory=lambda: project_root()
        / "输入数据"
        / "大湾区街景图像"
        / "street_clip_features_768.npy"
    )
    poi_distribution_csv: Path = field(
        default_factory=lambda: project_root()
        / "输入数据"
        / "大湾区POI数据"
        / "poi_top23_by_gba_s2_cell.csv"
    )
    poi_graph_cell_ids_csv: Path = field(
        default_factory=lambda: project_root()
        / "输入数据"
        / "大湾区POI数据"
        / "poi_top23_gba_cosine_cell_ids.csv"
    )
    poi_similarity_blocks_dir: Path = field(
        default_factory=lambda: project_root()
        / "输入数据"
        / "大湾区POI数据"
        / "poi_top23_gba_cosine_blocks"
    )
    poi_graph_cache_dir: Path = field(
        default_factory=lambda: project_root()
        / "输入数据"
        / "大湾区POI数据"
        / "poi_threshold_0_8_csr_random_walk"
    )
    output_root: Path = field(
        default_factory=lambda: module_root() / "outputs"
    )


@dataclass
class GraphConfig:
    poi_threshold: float = 0.8
    max_block_files: int = 0


@dataclass
class RandomWalkConfig:
    num_walks: int = 4
    walk_length: int = 8
    restart_prob: float = 0.2


@dataclass
class ModelConfig:
    use_poi: bool = True
    poi_node_dim: int = 128
    fusion_dim: int = 256
    contrastive_dim: int = 128
    transformer_depth: int = 1
    num_heads: int = 4
    ff_dim: int = 512
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    batch_size: int = 64
    epochs: int = 400
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    val_ratio: float = 0.1
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 0.0
    num_workers: int = 0
    seed: int = 42
    max_samples: int = 0
    device: str = "auto"
    amp: bool = True


@dataclass
class ExperimentConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    random_walk: RandomWalkConfig = field(default_factory=RandomWalkConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


T = TypeVar("T")


def _coerce_dataclass(data: Dict[str, Any], cls: Type[T]) -> T:
    kwargs: Dict[str, Any] = {}
    for item in fields(cls):
        value = data.get(item.name)
        if value is None:
            continue
        if cls is PathConfig:
            kwargs[item.name] = Path(value)
        else:
            kwargs[item.name] = value
    return cls(**kwargs)


def default_config() -> ExperimentConfig:
    return ExperimentConfig()


def config_to_dict(config: ExperimentConfig) -> Dict[str, Any]:
    raw = asdict(config)

    def convert(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        if isinstance(obj, Path):
            return str(obj)
        return obj

    return convert(raw)


def config_from_dict(data: Dict[str, Any]) -> ExperimentConfig:
    return ExperimentConfig(
        paths=_coerce_dataclass(data["paths"], PathConfig),
        graph=_coerce_dataclass(data["graph"], GraphConfig),
        random_walk=_coerce_dataclass(data["random_walk"], RandomWalkConfig),
        model=_coerce_dataclass(data["model"], ModelConfig),
        training=_coerce_dataclass(data["training"], TrainingConfig),
    )


def save_config(config: ExperimentConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config_to_dict(config), handle, ensure_ascii=False, indent=2)


def load_config(path: Path) -> ExperimentConfig:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return config_from_dict(data)
