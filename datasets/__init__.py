from configs import model_config
from datasets.BufferDataset import BufferDataset
from datasets.alignments import unpack_alignments
from datasets.build import get_data_to_buffer
from datasets.download import download_archive, preprocess_ljspeech


def prepare_data():
    download_archive()
    preprocess_ljspeech()
    unpack_alignments()
    buffer, (f0_min, f0_max, energy_min, energy_max) = get_data_to_buffer()
    model_config.f0_min = f0_min
    model_config.f0_max = f0_max
    model_config.energy_min = energy_min
    model_config.energy_max = energy_max
    print(f"Extracted f0_min={f0_min}, f0_max={f0_max}, energy_min={energy_min}, energy_max={energy_max}")
    return BufferDataset(buffer)
