import torch
import torch.nn as nn

from sgan.models.transformer.custom_transformer import TransformerEncoder
from sgan.models.Utils import log


class SGANEncoder(nn.Module):
    """Encoder is part of both TrajectoryGenerator and
    TrajectoryDiscriminator"""
    def __init__(self, device, feature_count, layer_count=6, emb_size=512, ff_size=2048, heads=8, dropout=0.1):
        super(SGANEncoder, self).__init__()
        self.device = device
        self.tf_encoder = TransformerEncoder(
            enc_inp_size=feature_count,
            n=layer_count,
            d_model=emb_size,
            d_ff=ff_size,
            h=heads,
            dropout=dropout
        )

    def forward(self, obs_traj, src_att):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """
        final_h_tf = self.tf_encoder(obs_traj, src_att)

        return final_h_tf
