import os

import numpy as np
import pandas as pd
import torch

import audio
import text as txt
import waveglow
from configs import train_config


def synthesis(model, text, speed_coef=1.0, pitch_coef=1.0, energy_coef=1.0):
    text = np.array(text)
    text = np.stack([text])
    src_pos = np.array([i+1 for i in range(text.shape[1])])
    src_pos = np.stack([src_pos])
    sequence = torch.from_numpy(text).long().to(train_config.device)
    src_pos = torch.from_numpy(src_pos).long().to(train_config.device)

    with torch.no_grad():
        model_output = model.forward(sequence, src_pos, None, None,
                                     None, None, None,
                                     speed_coef, pitch_coef, energy_coef)
    return model_output[0].cpu().transpose(0, 1), model_output.contiguous().transpose(1, 2)


def make_audio(model, waveglow_model, text=train_config.logging_text, path=train_config.inf_audio_path,
               res_prefix="res", speed_coef=1.0, pitch_coef=1.0, energy_coef=1.0):

    phn = txt.text_to_sequence(text, train_config.text_cleaners)

    mel, mel_cuda = synthesis(model, phn, speed_coef, pitch_coef, energy_coef)

    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    wav_path = f"{path}/{res_prefix}_s={speed_coef}_p={pitch_coef}_e={energy_coef}.wav"
    waveglow_wav_path = f"{path}/{res_prefix}_s={speed_coef}_p={pitch_coef}_e={energy_coef}_waveglow.wav"

    audio.tools.inv_mel_spec(
        mel, wav_path
    )

    waveglow.inference.inference(
        mel_cuda, waveglow_model,
        waveglow_wav_path
    )

    return wav_path, waveglow_wav_path


def log_to_wandb(logger, model, waveglow_model, subpath="latest", speed_coef=1.0, pitch_coef=1.0, energy_coef=1.0,
                 audios=None, epoch=train_config.save_step):
    if not os.path.exists(train_config.train_audio_path):
        os.makedirs(train_config.train_audio_path, exist_ok=True)

    if not os.path.exists(train_config.train_audio_path):
        os.makedirs(f"{train_config.train_audio_path}/{subpath}", exist_ok=True)

    wav_path, waveglow_wav_path = make_audio(model, waveglow_model,
                                             speed_coef=speed_coef, pitch_coef=pitch_coef, energy_coef=energy_coef)

    wav = logger.wandb.Audio(wav_path, sample_rate=audio.hparams_audio.sampling_rate)
    waveglow_wav = logger.wandb.Audio(waveglow_wav_path, sample_rate=audio.hparams_audio.sampling_rate)

    if audios is None:
        audios = {}

    audios_len = len(audios.keys())
    audios[audios_len] = {"epoch": epoch, "wav": wav, "waveglow_wav": waveglow_wav}

    logger.add_table("generated_audio", pd.DataFrame.from_dict(audios, orient="index"))

    return audios
