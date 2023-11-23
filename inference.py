import argparse

import torch
from tqdm import tqdm

import utils
from configs import mel_config, train_config, model_config
from model.FastSpeech2 import FastSpeech2
from synthesis import make_audio

parser = argparse.ArgumentParser(prog="FastSpeech2 inference")

parser.add_argument('model_state_dict_path', type=str,
                    help='relative path for model state dict')

args = parser.parse_args()

WaveGlow = utils.get_WaveGlow()
WaveGlow = WaveGlow.cuda()

model = FastSpeech2(mel_config, model_config)
model.load_state_dict(torch.load(args.model_state_dict_path, map_location='cuda:0')['model'])
model = model.to(train_config.device)
model = model.eval()

texts = train_config.texts

for config in train_config.configs:
    for i, text in tqdm(enumerate(texts)):
        make_audio(model, WaveGlow, text=text, path=train_config.inf_audio_path, res_prefix=f"res_{i}",
                   speed_coef=config["speed"], pitch_coef=config["pitch"], energy_coef=config["energy"])
