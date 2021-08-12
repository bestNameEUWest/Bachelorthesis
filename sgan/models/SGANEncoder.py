import torch
import torch.nn as nn

from sgan.models.transformer.batch import subsequent_mask
from sgan.models.Utils import log


class SGANEncoder(nn.Module):
    """Encoder is part of both TrajectoryGenerator and
    TrajectoryDiscriminator"""
    def __init__(self, device, tf_encoder):
        super(SGANEncoder, self).__init__()
        self.device = device
        self.tf_encoder = tf_encoder

    def forward(self, obs_traj):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """

        tf_permute = obs_traj.clone().permute(1, 0, 2)
        inp = tf_permute[:, 1:, :].to(self.device)
        src_att = torch.ones((inp.shape[0], 1, inp.shape[1])).to(self.device)
        final_h_tf = self.tf_encoder(inp, src_att)

        return final_h_tf, src_att
