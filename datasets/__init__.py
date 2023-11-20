from download import download_archive, preprocess_ljspeech
from build import get_data_to_buffer
from BufferDataset import BufferDataset
from alignments import unpack_alignments

from configs import model_config


def main():
    download_archive()
    preprocess_ljspeech()
    unpack_alignments()
    buffer, (f0_min, f0_max, energy_min, energy_max) = get_data_to_buffer()
    model_config.f0_min = f0_min
    model_config.f0_max = f0_max
    model_config.energy_min = energy_min
    model_config.energy_max = energy_max
    print(f"Extracted f0_min={f0_min}, f0_max={f0_max}, energy_min={energy_min}, energy_max={energy_max}")
    return BufferDataset(buffer, f0_min, f0_max,energy_min, energy_max)


if __name__ == "__main__":
    main()
