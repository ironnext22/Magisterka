from projekt.settings import load_config
from projekt.data.index_io import load_index_csv
from projekt.data.dataset import TerrainTripletDataset
from pathlib import Path

def main():
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(project_root / "configs" / "config.yaml")

    index_path = cfg.paths.data_interim / "index.csv"

    triplets = load_index_csv(cfg.paths.data_raw, index_path)

    dataset = TerrainTripletDataset(triplets)

    print("Dataset size:", len(dataset))

    sample = dataset[0]

    print("Sample ID:", sample["id"])
    print("Heightmap shape:", sample["heightmap"].shape)
    print("Segmentation shape:", sample["segmentation"].shape)
    print("Texture shape:", sample["texture"].shape)

    import torch

    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("Torch CUDA:", torch.version.cuda)

if __name__ == "__main__":
    main()