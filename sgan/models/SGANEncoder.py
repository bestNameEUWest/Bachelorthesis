import torch
import torch.nn as nn

from sgan.models.transformer.batch import subsequent_mask
from sgan.models.Utils import log


class SGANEncoder(nn.Module):
    """Encoder is part of both TrajectoryGenerator and
    TrajectoryDiscriminator"""
    def __init__(self, injected_encoder=None, embedding_dim=64, h_dim=64, mlp_dim=1024, num_layers=1, dropout=0.0):
        super(SGANEncoder, self).__init__()

        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.LSTMencoder = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout)
        self.encoder = injected_encoder
        self.spatial_embedding = nn.Linear(2, h_dim)

    def forward(self, obs_traj):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """

        tf_permute = obs_traj.clone().permute(1, 0, 2)
        inp = tf_permute[:, 1:, :].cuda()
        src_att = torch.ones((inp.shape[0], 1, inp.shape[1])).cuda()
        final_h_tf = self.encoder(inp, src_att)

        return final_h_tf, src_att
