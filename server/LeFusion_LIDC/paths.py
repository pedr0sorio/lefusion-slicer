import os
from pathlib import Path

# Project directories
HOME_FOLDER = Path().home()
PROJECT_ROOT_PATH = Path(__file__).parent.parent.parent # lefusion-slicer

PROJECT_LEFUSION_PATH = Path(__file__).parent # lefusion-slicer/server/LeFusion_LIDC
# ... Server Paths
PROJECT_SERVER_PATH = PROJECT_ROOT_PATH / "server" # lefusion-slicer/server
SERVER_DATA_DIR = PROJECT_SERVER_PATH / "data"
IN_SERVER_DATA_DIR = SERVER_DATA_DIR / 'in'
SERVER_ASSET_DATA_DIR = SERVER_DATA_DIR / 'in' / "mask-asset"
OUT_SERVER_DATA_DIR = SERVER_DATA_DIR / 'out' # TODO ensure the out files are saved here
# ... Slicer Extension Paths
PROJECT_SLICER_PATH = PROJECT_ROOT_PATH / "slicer" # lefusion-slicer/slicer

# Project SEED
SEED = 18021999
