import torch.nn as nn

from sgan.models.transformer.custom_transformer import TransformerDecoder
from sgan.models.Utils import log


class SGANDecoder(nn.Module):
    """Decoder is part of TrajectoryGenerator"""
    def __init__(self, dec_inp_size=3, dec_out_size=3, layer_count=6, emb_size=512, ff_size=2048, heads=8,
                 dropout=0.1):
        super(SGANDecoder, self).__init__()

        self.tf_decoder = TransformerDecoder(
            dec_inp_size=dec_inp_size,
            dec_out_size=dec_out_size,
            n=layer_count,
            d_model=emb_size,
            d_ff=ff_size,
            h=heads,
            dropout=dropout,
        )

    def forward(self, state_tuple, src_att, dec_inp, trg_att):
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
