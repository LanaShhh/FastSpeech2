import logging
import os
import shutil
from pathlib import Path

from speechbrain.utils.data_utils import download_file

from configs import train_config
from datasets.build import build_from_path


def download_archive():
    if not Path(train_config.data_path).exists():
        os.makedirs(train_config.data_path, exist_ok=True)
    if not Path(train_config.dataset_archive_save_path).exists():
        logging.info(f"Downloading and unpacking dataset archive: {train_config.dataset_archive_path}")
        download_file(train_config.dataset_archive_path, dest=train_config.dataset_archive_save_path,
                      unpack=True, dest_unpack=train_config.data_path)


def preprocess_ljspeech():
    logging.info("Preprocessing metadata")
    in_dir = train_config.dataset_folder
    out_dir = train_config.mel_ground_truth
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    logging.info("Processing spectrograms")
    metadata = build_from_path(in_dir, out_dir)
    write_metadata(metadata, out_dir)

    shutil.move(os.path.join(train_config.mel_ground_truth, "train.txt"),
                os.path.join("data", "train.txt"))


def write_metadata(metadata, out_dir):
    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for m in metadata:
            f.write(m + '\n')
