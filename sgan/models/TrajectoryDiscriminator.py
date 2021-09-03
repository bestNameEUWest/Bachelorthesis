import torch
import torch.nn as nn

from sgan.models.Pooling import PoolHiddenNet
from sgan.models.Utils import make_mlp, log
from sgan.models.transformer.custom_transformer import TransformerEncoder


class TrajectoryDiscriminator(nn.Module):
    def __init__(
        self, device, feature_count=2, layer_count=1, tf_emb_dim=64, tf_ff_size=2048, heads=8, bottleneck_dim=1024,
        pool_emb_dim=64, dropout=0.1, mlp_dim=1024, activation='relu', batch_norm=True, d_type='local',
    ):
        super(TrajectoryDiscriminator, self).__init__()

        self.d_type = d_type
        self.device = device

        self.encoder = TransformerEncoder(
            enc_inp_size=feature_count,
            n=layer_count,
            d_model=tf_emb_dim,
            d_ff=tf_ff_size,
            h=heads,
            dropout=dropout
        )

        real_classifier_dims = [tf_emb_dim, mlp_dim, 1]
        self.real_classifier = make_mlp(
            real_classifier_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )
        if d_type == 'global':
            mlp_pool_dims = [tf_emb_dim + pool_emb_dim, mlp_dim, tf_emb_dim]
            self.pool_net = PoolHiddenNet(
                pool_emb_dim=pool_emb_dim,
                h_dim=tf_emb_dim,
                mlp_dim=mlp_pool_dims,
                bottleneck_dim=bottleneck_dim,
                activation=activation,
                batch_norm=batch_norm
            )

    def forward(self, traj, traj_rel, mean, std, seq_start_end=None):
        """
        Inputs:
        - traj: Tensor of shape (obs_len + pred_len, batch, 2)
        - traj_rel: Tensor of shape (obs_len + pred_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch
        Output:
        - scores: Tensor of shape (batch,) with real/fake scores
        """

        inp = (traj_rel.permute(1, 0, 2)[:, 1:, :].to(self.device) - mean) / std
        src_att = torch.ones((inp.shape[0], 1, inp.shape[1])).to(self.device)

        final_h = self.encoder(inp, src_att)
        final_h = final_h.permute(1, 0, 2)

        # Note: In case of 'global' option we are using start_pos as opposed to
        # end_pos. The intution being that hidden state has the whole
        # trajectory and relative postion at the start when combined with
        # trajectory information should help in discriminative behavior.
        if self.d_type == 'local':
            classifier_input = final_h
        else:
            start_positions = traj[:-1, :, :]
            classifier_input = self.pool_net(final_h, seq_start_end, start_positions)
        scores = self.real_classifier(classifier_input)
        return scores
