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

    def init_hidden(self, batch):
        return (
            torch.zeros(self.num_layers, batch, self.h_dim).cuda(),
            torch.zeros(self.num_layers, batch, self.h_dim).cuda()
        )

    def forward(self, obs_traj):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """

        # for TF encoder
        tf_permute = obs_traj.clone().permute(1, 0, 2)
        # log('tf_permute shape', tf_permute.shape)

        inp = tf_permute[:, 1:, :].cuda()
        src_att = torch.ones((inp.shape[0], 1, inp.shape[1])).cuda()
        final_h_tf = self.encoder(inp, src_att)

        # for LSTMencoder
        # we first have to embed them here (in TF we have this handled by our model)
        batch = obs_traj.size(1)
        # log('emb arg shape', obs_traj.contiguous().view(-1, 2).shape)
        obs_traj_embedding = self.spatial_embedding(obs_traj.contiguous().view(-1, 2))
        obs_traj_embedding = obs_traj_embedding.view(-1, batch, self.embedding_dim)
        # log('emb shape', obs_traj_embedding.shape)

        state_tuple = self.init_hidden(batch)
        output, state = self.LSTMencoder(obs_traj_embedding, state_tuple)
        # log('lstm output shape', output.shape)

        final_h_lstm = state[0]

        return final_h_tf, final_h_lstm
