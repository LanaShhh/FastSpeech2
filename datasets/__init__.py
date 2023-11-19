from download import download_archive, preprocess_ljspeech
from build import get_data_to_buffer
from BufferDataset import BufferDataset
from alignments import unpack_alignments


def main():
    download_archive()
    preprocess_ljspeech()
    unpack_alignments()
    buffer, (f0_min, f0_max, energy_min, energy_max) = get_data_to_buffer()
    return BufferDataset(buffer, f0_min, f0_max,energy_min, energy_max)


if __name__ == "__main__":
    main()