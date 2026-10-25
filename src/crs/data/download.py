"""Download the Lending Club data from Kaggle.

Needs a Kaggle API token in ~/.kaggle/kaggle.json (Kaggle -> Settings -> API -> Create New Token).
The token stays on your machine: it is never part of this repository.
"""

from crs import config


def download(raw_dir=config.RAW_DIR):
    from kaggle.api.kaggle_api_extended import KaggleApi  # optional dependency: pip install -e ".[download]"

    api = KaggleApi()
    api.authenticate()
    raw_dir.mkdir(parents=True, exist_ok=True)
    api.dataset_download_files(config.KAGGLE_DATASET, path=str(raw_dir), unzip=True, quiet=False)
    return raw_dir
