import argparse
import logging
import os

import torch
import torch.nn as nn
import wandb
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from collate import collate_fn_tensor
from configs import model_config, mel_config, train_config
from datasets import prepare_data
from loss import FastSpeech2Loss
from model.FastSpeech2 import FastSpeech2
from synthesis import log_to_wandb
from wandb_writer import WanDBWriter

parser = argparse.ArgumentParser(prog="FastSpeech2 training")

parser.add_argument('wandb_key', type=str, required=True,
                    help='Wandb key for logging')

args = parser.parse_args()

wandb.login("never", args.wandb_key)

dataset = prepare_data()

training_loader = DataLoader(
    dataset,
    batch_size=train_config.batch_expand_size * train_config.batch_size,
    shuffle=True,
    collate_fn=collate_fn_tensor,
    drop_last=True,
    num_workers=0
)

model = FastSpeech2(mel_config, model_config)
model = model.to(train_config.device)

fastspeech_loss = FastSpeech2Loss()
current_step = 0

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=train_config.learning_rate,
    betas=(0.9, 0.98),
    eps=1e-9)

scheduler = OneCycleLR(optimizer, **{
    "steps_per_epoch": len(training_loader) * train_config.batch_expand_size,
    "epochs": train_config.epochs,
    "anneal_strategy": "cos",
    "max_lr": train_config.learning_rate,
    "pct_start": 0.1
})

logger = WanDBWriter(train_config)

tqdm_bar = tqdm(total=train_config.epochs * len(training_loader) * train_config.batch_expand_size - current_step)

logging.info("Starting training")

audios = None

WaveGlow = utils.get_WaveGlow()
WaveGlow = WaveGlow.to(train_config.device).cuda()

for epoch in range(train_config.epochs):
    for i, batchs in enumerate(training_loader):
        # real batch start here
        for j, db in enumerate(batchs):
            current_step += 1
            tqdm_bar.update(1)

            logger.set_step(current_step)

            # Get Data
            character = db["text"].long().to(train_config.device)
            mel_target = db["mel_target"].float().to(train_config.device)
            duration = db["duration"].int().to(train_config.device)
            pitch = db["f0"].to(train_config.device)
            energy = db["energy"].to(train_config.device)
            mel_pos = db["mel_pos"].long().to(train_config.device)
            src_pos = db["src_pos"].long().to(train_config.device)
            max_mel_len = db["mel_max_len"]

            # Forward
            mel_output, duration_prediction, pitch_prediction, energy_prediction = \
                model(character, src_pos, mel_pos=mel_pos, mel_max_length=max_mel_len,
                      length_target=duration, pitch_target=pitch, energy_target=energy)

            # Calc Loss
            mel_loss, duration_loss, pitch_loss, energy_loss = fastspeech_loss(mel_output,
                                                      duration_prediction, pitch_prediction, energy_prediction,
                                                      mel_target, duration, pitch, energy)

            total_loss = mel_loss + duration_loss + pitch_loss + energy_loss

            # Logger
            t_l = total_loss.detach().cpu().numpy()
            m_l = mel_loss.detach().cpu().numpy()
            d_l = duration_loss.detach().cpu().numpy()
            p_l = pitch_loss.detach().cpu().numpy()
            e_l = energy_loss.detach().cpu().numpy()

            logger.add_scalar("duration_loss", d_l)
            logger.add_scalar("mel_loss", m_l)
            logger.add_scalar("total_loss", t_l)
            logger.add_scalar("pitch_loss", p_l)
            logger.add_scalar("energy_loss", e_l)

            # Backward
            total_loss.backward()

            # Clipping gradients to avoid gradient explosion
            nn.utils.clip_grad_norm_(
                model.parameters(), train_config.grad_clip_thresh)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            if not os.path.exists(train_config.data_path):
                os.makedirs(train_config.data_path, exist_ok=True)
            if not os.path.exists(train_config.checkpoint_path):
                os.makedirs(train_config.checkpoint_path, exist_ok=True)

            if current_step % train_config.save_step == 0:
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(
                )}, os.path.join(train_config.checkpoint_path, 'checkpoint_%d.pth.tar' % current_step))
                print("save model at step %d ..." % current_step)

            audios = log_to_wandb(logger, model, WaveGlow, subpath=f"{current_step}", audios=audios, epoch=current_step)
