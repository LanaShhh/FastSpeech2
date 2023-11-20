import torch
import torch.nn as nn

from model.LengthRegulator import LengthRegulator
from model.VariancePredictor import VariancePredictor
from configs import model_config


class VarianceAdaptor(nn.Module):
    """ Variance Adaptor """

    def __init__(self,model_config):
        super(VarianceAdaptor, self).__init__()

        self.lr = LengthRegulator(model_config)
        self.pitch_predictor = VariancePredictor(model_config)
        self.energy_predictor = VariancePredictor(model_config)

        # self.f0_min_log = torch.log(model_config.f0_min).item()
        # self.f0_max_log = torch.log(model_config.f0_max).item()
        # self.pitch_bins = nn.Parameter(torch.exp(
        #     torch.linspace(self.f0_min_log, self.f0_max_log, model_config.quantize_bins_cnt - 1)
        # ))
        self.f0_min = model_config.f0_min
        self.f0_max = model_config.f0_max
        self.energy_bins = nn.Parameter(
            torch.linspace(self.f0_min, self.f0_max, model_config.quantize_bins_cnt - 1)
        )

        self.energy_min = model_config.energy_min
        self.energy_max = model_config.energy_max
        self.pitch_bins = nn.Parameter(
            torch.linspace(self.energy_min, self.energy_max, model_config.quantize_bins_cnt - 1)
        )

        self.pitch_embed = nn.Embedding(model_config.quantize_bins_cnt, model_config.encoder_dim)
        self.energy_embed = nn.Embedding(model_config.quantize_bins_cnt, model_config.encoder_dim)

    def forward(self, x, alpha=1.0, duration_target=None, pitch_target=None, energy_target=None, mel_max_length=None):
        output, duration_predictor_output = self.lr(x, alpha, duration_target, mel_max_length)

        pitch_predictor_output = self.pitch_predictor(output)
        energy_predictor_output = self.energy_predictor(output)

        if pitch_target is not None:
            pitch_additional = torch.bucketize(pitch_target, self.pitch_bins)
        else:
            pitch_additional = pitch_predictor_output

        if energy_target is not None:
            energy_additional = torch.bucketize(energy_target, self.energy_bins)
        else:
            energy_additional = energy_predictor_output

        print(pitch_additional.size(), output.size())
        output = output + self.pitch_embed(pitch_additional)
        output = output + self.energy_embed(energy_additional)

        return output, duration_predictor_output, pitch_predictor_output, energy_predictor_output
