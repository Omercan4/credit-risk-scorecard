"""Download the Lending Club data from Kaggle.

Needs a Kaggle API token (Kaggle -> Settings -> API -> Create New Token), stored in ~/.kaggle/.
The token stays on your machine: it is never part of this repository.
Only the two compressed files are fetched (about 650 MB), not the 4 GB of unpacked copies.
"""

from crs import config

FILES = ["accepted_2007_to_2018Q4.csv.gz", "rejected_2007_to_2018Q4.csv.gz"]


def download(raw_dir=config.RAW_DIR):
    from kaggle.api.kaggle_api_extended import KaggleApi  # optional dependency: pip install -e ".[download]"

    api = KaggleApi()
    api.authenticate()
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        if not (raw_dir / name).exists():
            api.dataset_download_file(config.KAGGLE_DATASET, name, path=str(raw_dir), quiet=False)
    return raw_dir
