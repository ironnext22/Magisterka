from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

@dataclass(frozen=True)
class PathsCfg:
    data_raw: Path
    data_interim: Path
    data_processed: Path
    outputs: Path

@dataclass(frozen=True)
class NamingCfg:
    height_suffix: str
    seg_suffix: str
    tex_suffix: str
    extension: str


@dataclass(frozen=True)
class DatasetCfg:
    image_size: int


@dataclass(frozen=True)
class ProcessingCfg:
    normalize_heightmap: bool
    convert_segmentation_to_labels: bool


@dataclass(frozen=True)
class Config:
    project_name: str
    paths: PathsCfg
    naming: NamingCfg
    dataset: DatasetCfg
    processing: ProcessingCfg

def _require(d: Dict[str, Any], key: str) -> Any:
    if key not in d:
        raise KeyError(f"Missing key in config: {key}")
    return d[key]

def load_config(path: str | Path = "/configs/config.yaml") -> Config:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    project = _require(cfg, "project")
    paths = _require(cfg, "paths")
    naming = _require(cfg, "naming")
    dataset = _require(cfg, "dataset")
    processing = _require(cfg, "processing")

    return Config(
        project_name=_require(project, "name"),
        paths=PathsCfg(
            data_raw=Path(_require(paths, "data_raw")),
            data_interim=Path(_require(paths, "data_interim")),
            data_processed=Path(_require(paths, "data_processed")),
            outputs=Path(_require(paths, "outputs")),
        ),
        naming=NamingCfg(
            height_suffix=_require(naming, "height_suffix"),
            seg_suffix=_require(naming, "seg_suffix"),
            tex_suffix=_require(naming, "tex_suffix"),
            extension=_require(naming, "extension"),
        ),
        dataset=DatasetCfg(
            image_size=int(_require(dataset, "image_size")),
        ),
        processing=ProcessingCfg(
            normalize_heightmap=bool(_require(processing, "normalize_heightmap")),
            convert_segmentation_to_labels=bool(
                _require(processing, "convert_segmentation_to_labels")
            ),
        ),
    )