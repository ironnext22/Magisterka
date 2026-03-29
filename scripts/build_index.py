from projekt.settings import load_config
from projekt.data.pair_index import build_triplets_from_single_folder
from projekt.data.index_io import save_index_csv
from pathlib import Path

def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(project_root / "configs" / "config.yaml")

    raw_root = cfg.paths.data_raw
    interim_root = cfg.paths.data_interim

    out_csv = interim_root / "index.csv"

    triplets = build_triplets_from_single_folder(
        folder=raw_root,
        height_suffix=cfg.naming.height_suffix,
        seg_suffix=cfg.naming.seg_suffix,
        tex_suffix=cfg.naming.tex_suffix,
        extension=cfg.naming.extension,
        strict=True,
    )

    save_index_csv(triplets, raw_root, out_csv)

    print(f"Index created: {out_csv}")
    print(f"Samples found: {len(triplets)}")


if __name__ == "__main__":
    main()