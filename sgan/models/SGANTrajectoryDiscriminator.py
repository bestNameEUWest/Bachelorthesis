import torch
import torch.nn as nn

from sgan.models.Pooling import PoolHiddenNet
from sgan.models.Utils import make_mlp, log


class TrajectoryDiscriminator(nn.Module):
    def __init__(
        self, device, tf_emb_dim=64, bottleneck_dim=1024,
        pool_emb_dim=64, dropout=0.1, mlp_dim=1024, activation='relu', batch_norm=True, d_type='local',
    ):
        super(TrajectoryDiscriminator, self).__init__()

        self.d_type = d_type
        self.device = device
        self.embedding_dim = tf_emb_dim

        real_classifier_dims = [tf_emb_dim, mlp_dim, 1]
        self.real_classifier = make_mlp(
            real_classifier_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )
        self.spatial_embedding = nn.Linear(2, tf_emb_dim)
        self.encoder = nn.LSTM(tf_emb_dim, 64, 1, dropout=dropout)


        if d_type == 'global':
            mlp_pool_dims = [tf_emb_dim + pool_emb_dim, mlp_dim, tf_emb_dim]
            self.pool_net = PoolHiddenNet(
                pool_emb_dim=pool_emb_dim,
                mlp_dim=mlp_pool_dims,
                bottleneck_dim=bottleneck_dim,
                activation=activation,
                batch_norm=batch_norm
            )

    def init_hidden(self, batch):
        return (
            torch.zeros(1, batch, 64).cuda(), # num_layers and h_dim
            torch.zeros(1, batch, 64).cuda()
        )

    def forward(self, traj, traj_rel, seq_start_end=None):
        """
        Inputs:
        - traj: Tensor of shape (obs_len + pred_len, batch, 2)
        - traj_rel: Tensor of shape (obs_len + pred_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch
        Output:
        - scores: Tensor of shape (batch,) with real/fake scores
        """
        batch = traj_rel.size(1)
        obs_traj_embedding = self.spatial_embedding(traj_rel.view(-1, 2))
        obs_traj_embedding = obs_traj_embedding.view(-1, batch, self.embedding_dim)
        state_tuple = self.init_hidden(batch)
        output, state = self.encoder(obs_traj_embedding, state_tuple)
        final_h = state[0]

        # Note: In case of 'global' option we are using start_pos as opposed to
        # end_pos. The intution being that hidden state has the whole
        # trajectory and relative postion at the start when combined with
        # trajectory information should help in discriminative behavior.
        if self.d_type == 'local':
            classifier_input = final_h.squeeze()
        else:
            classifier_input = self.pool_net(final_h.squeeze(), seq_start_end, traj[0])
        scores = self.real_classifier(classifier_input)
        return scores
