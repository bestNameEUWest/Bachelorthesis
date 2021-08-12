import torch
import torch.nn as nn

from sgan.models.Pooling import PoolHiddenNet, SocialPooling
from sgan.models.Utils import make_mlp, log


class SGANDecoder(nn.Module):
    """Decoder is part of TrajectoryGenerator"""
    def __init__(self, seq_len, device, tf_decoder):
        super(SGANDecoder, self).__init__()

        self.seq_len = seq_len
        self.device = device
        self.tf_decoder = tf_decoder

    def forward(self, state_tuple, src_att=None, dec_inp=None, trg_att=None):
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

        pred_traj_fake_rel = self.tf_decoder(final_encoder_h, src_att, dec_inp, trg_att)
        pred_traj_fake_rel = pred_traj_fake_rel.permute(1, 0, 2)[:, :, 0:2].contiguous()

        return pred_traj_fake_rel
