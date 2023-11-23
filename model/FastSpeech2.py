import torch
import torch.nn as nn

from model.Decoder import Decoder
from model.Encoder import Encoder
from model.VarianceAdaptor import VarianceAdaptor
from utils import get_mask_from_lengths
from model.LengthRegulator import LengthRegulator


class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, mel_config, model_config):
        super(FastSpeech2, self).__init__()

        self.encoder = Encoder(model_config)
        self.lr = LengthRegulator(model_config)
        self.variance_adaptor = VarianceAdaptor(model_config)
        self.decoder = Decoder(model_config)

        self.mel_linear = nn.Linear(model_config.decoder_dim, mel_config.num_mels)

    def mask_tensor(self, mel_output, position, mel_max_length):
        lengths = torch.max(position, -1)[0]
        mask = ~get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.)

    def forward(self, src_seq, src_pos, mel_pos=None, mel_max_length=None,
                length_target=None, pitch_target=None, energy_target=None, alpha=1.0, pitch_coef=1.0, energy_coef=1.0):
        output, _ = self.encoder(src_seq, src_pos)

        # Your code here
        if self.training:
            output, duration_predictor_output = self.lr(output, alpha, length_target, mel_max_length)
            output, pitch_predictor_output, energy_predictor_output = \
                self.variance_adaptor(output, pitch_target, energy_target)

            output = self.decoder(output, mel_pos)
            output = self.mask_tensor(output, mel_pos, mel_max_length)
            output = self.mel_linear(output)

            return output, duration_predictor_output, pitch_predictor_output, energy_predictor_output
        else:
            output, duration_predictor_output = self.lr(output, alpha)
            output, pitch_predictor_output, energy_predictor_output = \
                self.variance_adaptor(output, None, None, pitch_coef, energy_coef)

            output = self.decoder(output, duration_predictor_output)
            output = self.mel_linear(output)

            return output

