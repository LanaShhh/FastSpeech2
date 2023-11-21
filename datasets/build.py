import logging
import os
import time

import librosa
import numpy as np
import pyworld as pw
import torch
from tqdm import tqdm

import audio
import audio.hparams_audio as hparams
from configs import train_config
from text import text_to_sequence
from utils import process_text


def build_from_path(in_dir, out_dir):
    index = 1
    texts = []
    if not os.path.exists(train_config.f0_path):
        os.makedirs(train_config.f0_path, exist_ok=True)
    if not os.path.exists(train_config.energy_path):
        os.makedirs(train_config.energy_path, exist_ok=True)
    with open(os.path.join(train_config.dataset_folder, 'metadata.csv'), encoding='utf-8') as f:
        for line in f.readlines():
            if index % 100 == 0:
                print("{:d} Done".format(index))
            parts = line.strip().split('|')
            wav_path = os.path.join(in_dir, 'wavs', '%s.wav' % parts[0])
            text = parts[2]
            texts.append(_process_utterance(out_dir, index, wav_path, text))

            index = index + 1

    return texts


def _process_utterance(out_dir, index, wav_path, text):
    # Compute a mel-scale spectrogram from the wav:
    mel_spectrogram, energy = audio.tools.get_mel(wav_path)
    mel_spectrogram = mel_spectrogram.numpy().astype(np.float32)
    energy = energy.numpy().astype(np.float32)

    # Write the spectrograms to disk:
    mel_filename = 'ljspeech-mel-%05d.npy' % index
    np.save(os.path.join(out_dir, mel_filename),
            mel_spectrogram.T, allow_pickle=False)

    # Write f0 and energy to disk
    wav, sr = librosa.load(wav_path, sr=None, dtype=np.float64)
    f0, t = pw.dio(wav, sr, frame_period=hparams.hop_length / hparams.sampling_rate * 1000)
    np.save(os.path.join(train_config.f0_path, f"{index}.npy"), f0)
    np.save(os.path.join(train_config.energy_path, f"{index}.npy"), energy)

    return text


def get_data_to_buffer():
    buffer = list()
    text = process_text(train_config.text_path)

    start = time.perf_counter()

    f0_min, f0_max = torch.inf, -torch.inf
    energy_min, energy_max = torch.inf, -torch.inf

    logging.info("Creating buffer for dataset")
    for i in tqdm(range(len(text))):

        mel_gt_name = os.path.join(
            train_config.mel_ground_truth, "ljspeech-mel-%05d.npy" % (i + 1))
        mel_gt_target = np.load(mel_gt_name)
        duration = np.load(os.path.join(
            train_config.alignment_path, str(i) + ".npy"))
        character = text[i][0:len(text[i])-1]
        character = np.array(
            text_to_sequence(character, train_config.text_cleaners))

        f0 = np.load(os.path.join(
            train_config.f0_path, str(i + 1) + ".npy")).astype(np.float32)
        energy = np.load(os.path.join(
            train_config.energy_path, str(i + 1) + ".npy"))[0]

        character = torch.from_numpy(character)
        duration = torch.from_numpy(duration)
        mel_gt_target = torch.from_numpy(mel_gt_target)
        f0 = torch.from_numpy(f0)
        energy = torch.from_numpy(energy)

        f0_min = min(f0_min, f0.min())
        f0_max = max(f0_max, f0.max())

        energy_min = min(energy_min, energy.min())
        energy_max = max(energy_max, energy.max())

        buffer.append({"text": character, "duration": duration,
                       "mel_target": mel_gt_target,
                       "f0": f0, "energy": energy})

    end = time.perf_counter()
    print("cost {:.2f}s to load all data into buffer.".format(end-start))

    return buffer, [f0_min, f0_max, energy_min, energy_max]
