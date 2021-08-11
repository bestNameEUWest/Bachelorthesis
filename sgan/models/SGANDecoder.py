import torch
import torch.nn as nn

from sgan.models.Pooling import PoolHiddenNet, SocialPooling
from sgan.models.Utils import make_mlp, log


class SGANDecoder(nn.Module):
    """Decoder is part of TrajectoryGenerator"""
    def __init__(
        self, seq_len, injected_decoder=None, embedding_dim=64, h_dim=128, mlp_dim=1024, num_layers=1,
        pool_every_timestep=True, dropout=0.0, bottleneck_dim=1024,
        activation='relu', batch_norm=True, pooling_type='pool_net',
        neighborhood_size=2.0, grid_size=8
    ):
        super(SGANDecoder, self).__init__()

        self.seq_len = seq_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.pool_every_timestep = pool_every_timestep

        self.LSTMdecoder = nn.LSTM(
            embedding_dim, h_dim, num_layers, dropout=dropout
        )

        self.decoder = injected_decoder

        if pool_every_timestep:
            if pooling_type == 'pool_net':
                self.pool_net = PoolHiddenNet(
                    embedding_dim=self.embedding_dim,
                    h_dim=self.h_dim,
                    mlp_dim=mlp_dim,
                    bottleneck_dim=bottleneck_dim,
                    activation=activation,
                    batch_norm=batch_norm,
                    dropout=dropout
                )
            elif pooling_type == 'spool':
                self.pool_net = SocialPooling(
                    h_dim=self.h_dim,
                    activation=activation,
                    batch_norm=batch_norm,
                    dropout=dropout,
                    neighborhood_size=neighborhood_size,
                    grid_size=grid_size
                )

            mlp_dims = [h_dim + bottleneck_dim, mlp_dim, h_dim]
            self.mlp = make_mlp(
                mlp_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout
            )

        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.hidden2pos = nn.Linear(h_dim, 2)

    def forward(self, last_pos, last_pos_rel, state_tuple, seq_start_end, src_att=None, dec_inp=None, trg_att=None):
        """
        Inputs:
        - last_pos: Tensor of shape (batch, 2)
        - last_pos_rel: Tensor of shape (batch, 2)
        - state_tuple: (hh, ch) each tensor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch
        Output:
        - pred_traj: tensor of shape (self.seq_len, batch, 2)
        """

        final_encoder_h = state_tuple[0].permute(1, 0, 2)

        # TODO: do this transformation somewhere else, here is not the right place
        # for TF should be torch.Size([166, 7, 128])
        test = nn.Linear(128, 64).cuda()
        final_encoder_h = test(final_encoder_h)

        pred_traj_fake_rel = self.decoder(final_encoder_h, src_att, dec_inp, trg_att)
        pred_traj_fake_rel = pred_traj_fake_rel.permute(1, 0, 2)[:, :, 0:2].contiguous()

        # this was part of the original code where they pooled the results between every timestep e.g. after each
        # prediction step they pooled them. I did not implement it here since we do not have a recurrent generation of
        # values in the Transformer and it wouldn't be trivial task to implement a pooling method in a attention tensor.

        # if self.pool_every_timestep:
        #     decoder_h = state_tuple[0]
        #     # log('decoder_h shape', decoder_h.shape)
        #     pool_h = self.pool_net(decoder_h, seq_start_end, curr_pos)
        #     decoder_h = torch.cat(
        #         [decoder_h.view(-1, self.h_dim), pool_h], dim=1)
        #     decoder_h = self.mlp(decoder_h)
        #     decoder_h = torch.unsqueeze(decoder_h, 0)
        #     state_tuple = (decoder_h, state_tuple[1])

        return pred_traj_fake_rel
