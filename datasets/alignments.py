import os.path
from pathlib import Path
from zipfile import ZipFile

from configs import train_config


def unpack_alignments():
    if not os.path.exists(train_config.alignment_path):
        os.makedirs(train_config.alignment_path, exist_ok=True)
        ZipFile("alignments.zip", 'r').extractall(Path(train_config.alignment_path).parent)
