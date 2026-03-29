from __future__ import annotations

from pathlib import Path

from projekt.data.preprocess import PreprocessConfig, run_preprocessing


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    index_path = project_root /"data" / "interim" / "index.csv"
    raw_root = project_root / "data"/ "raw"
    processed_root = project_root / "data" / "processed"

    config = PreprocessConfig(
        target_size=(256, 256),
        normalize_heightmap=True,
        overwrite=True,
    )

    processed_df = run_preprocessing(
        index_path=index_path,
        raw_root=raw_root,
        processed_root=processed_root,
        config=config,
    )

    print("Preprocessing completed.")
    print(f"Processed samples: {len(processed_df)}")
    print(f"Processed index saved to: {processed_root / 'index.csv'}")


if __name__ == "__main__":
    main()