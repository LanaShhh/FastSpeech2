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
    f0_min = 0.0
    f0_max = 795.7948608398438
    energy_min = 0.01786651276051998
    energy_max = 314.9619140625

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
    train_audio_path = "./train_audio"
    inf_audio_path = "./results"

    wandb_project = 'fastspeech2_sdzhumlyakova_implementation'

    text_cleaners = ['english_cleaners']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 16
    # epochs = 2000
    epochs = 500
    n_warm_up_step = 4000

    learning_rate = 1e-3
    weight_decay = 1e-6
    grad_clip_thresh = 1.0
    decay_step = [500000, 1000000, 2000000]

    # save_step = 3000
    save_step = 1000
    log_step = 5
    clear_Time = 20

    batch_expand_size = 32

    # for inference
    # generation configs
    configs = [
        {"speed": 1.0, "pitch": 1.0, "energy": 1.0},
        {"speed": 0.8, "pitch": 1.0, "energy": 1.0},
        {"speed": 1.2, "pitch": 1.0, "energy": 1.0},
        {"speed": 1.0, "pitch": 0.8, "energy": 1.0},
        {"speed": 1.0, "pitch": 1.2, "energy": 1.0},
        {"speed": 1.0, "pitch": 1.0, "energy": 0.8},
        {"speed": 1.0, "pitch": 1.0, "energy": 1.2},
        {"speed": 0.8, "pitch": 0.8, "energy": 0.8},
        {"speed": 1.2, "pitch": 1.2, "energy": 1.2}
    ]

    # for inference
    # texts to generate
    texts = [
        "A defibrillator is a device that gives a high energy electric shock to the heart of someone "
        "who is in cardiac arrest",
        "Massachusetts Institute of Technology may be best known for its math, science and engineering education",
        "Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined "
        "between probability distributions on a given metric space"
    ]

    # for training
    # text logged audios
    logging_text = ("The quality of a speech synthesizer is judged by its similarity "
                    "to the human voice and by its ability to be understood clearly")


mel_config = MelSpectrogramConfig()
model_config = FastSpeechConfig()
train_config = TrainConfig()
