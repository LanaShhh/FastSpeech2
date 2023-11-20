import torch.nn as nn


class FastSpeech2Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, mel, duration_predicted, pitch_predicted, energy_predicted,
                mel_target, duration_target, pitch_target, energy_target):
        mel_loss = self.mse_loss(mel, mel_target)

        duration_loss = self.l1_loss(duration_predicted, duration_target.float())
        pitch_loss = self.l1_loss(pitch_predicted, pitch_target.float())
        energy_loss = self.l1_loss(energy_predicted, energy_target.float())

        return mel_loss, duration_loss, pitch_loss, energy_loss
