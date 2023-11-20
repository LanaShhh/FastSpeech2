from dataclasses import dataclass

import torch


@dataclass
class MelSpectrogramConfig:
    num_mels = 80


@dataclass
class FastSpeechConfig:
    vocab_size = 300
    max_seq_len = 3000

    encoder_dim = 256
    encoder_n_layer = 4
    encoder_head = 2
    encoder_conv1d_filter_size = 1024

    decoder_dim = 256
    decoder_n_layer = 4
    decoder_head = 2
    decoder_conv1d_filter_size = 1024

    fft_conv1d_kernel = (9, 1)
    fft_conv1d_padding = (4, 0)

    duration_predictor_filter_size = 256
    duration_predictor_kernel_size = 3
    dropout = 0.1

    PAD = 0
    UNK = 1
    BOS = 2
    EOS = 3

    PAD_WORD = '<blank>'
    UNK_WORD = '<unk>'
    BOS_WORD = '<s>'
    EOS_WORD = '</s>'

    # parameters extracted from ljspeech dataset
    # after getting dataset again in training these parameters will be updated
    f0_min = -1
    f0_max = -1
    energy_min = -1
    energy_max = -1

    quantize_bins_cnt = 256  # added


@dataclass
class TrainConfig:
    checkpoint_path = "./checkpoints"
    logger_path = "./logger"
    data_path = './data'
    text_path = './data/train.txt'
    mel_ground_truth = "./data/mel"
    alignment_path = "./data/alignments"
    f0_path = './data/f0'  # added
    energy_path = './data/energy'  # added

    dataset_folder = "./data/LJSpeech-1.1"
    dataset_archive_path = 'https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2'  # added
    dataset_archive_save_path = './data/LJSpeech-1.1.tar.bz2'

    wandb_project = 'fastspeech_example'

    text_cleaners = ['english_cleaners']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 16
    epochs = 2000
    n_warm_up_step = 4000

    learning_rate = 1e-3
    weight_decay = 1e-6
    grad_clip_thresh = 1.0
    decay_step = [500000, 1000000, 2000000]

    save_step = 3000
    log_step = 5
    clear_Time = 20

    batch_expand_size = 32


mel_config = MelSpectrogramConfig()
model_config = FastSpeechConfig()
train_config = TrainConfig()
