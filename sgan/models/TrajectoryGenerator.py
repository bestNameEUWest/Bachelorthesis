import torch
import torch.nn as nn

from sgan.models.SGANEncoder import SGANEncoder
from sgan.models.SGANDecoder import SGANDecoder
from sgan.models.Pooling import PoolHiddenNet

from sgan.models.Utils import make_mlp, get_noise, log


class TrajectoryGenerator(nn.Module):
    def __init__(
            self, device, feature_count=2, pool_emb_dim=64, tf_emb_dim=64, tf_ff_size=2048, dec_inp_size=3,
            dec_out_size=3, mlp_dim=1024, layer_count=1, noise_dim=(0,),
            noise_type='gaussian', noise_mix_type='ped', pooling_type=None,
            dropout=0.0, heads=8, bottleneck_dim=1024, activation='relu', batch_norm=True,
    ):
        super(TrajectoryGenerator, self).__init__()

        if pooling_type and pooling_type.lower() == 'none':
            pooling_type = None
        self.tmp = pool_emb_dim + tf_emb_dim
        self.device = device
        self.noise_dim = noise_dim
        self.noise_type = noise_type
        self.noise_mix_type = noise_mix_type
        self.pooling_type = pooling_type
        self.noise_first_dim = 0
        self.bottleneck_dim = bottleneck_dim

        self.test = nn.Linear(pool_emb_dim + tf_emb_dim, tf_emb_dim)

        self.encoder = SGANEncoder(
            device=device,
            feature_count=feature_count,
            layer_count=layer_count,
            emb_size=tf_emb_dim,
            ff_size=tf_ff_size,
            heads=heads,
            dropout=dropout
        )
        self.decoder = SGANDecoder(
            dec_inp_size=dec_inp_size,
            dec_out_size=dec_out_size,
            layer_count=layer_count,
            emb_size=tf_emb_dim,
            ff_size=tf_ff_size,
            heads=heads,
            dropout=dropout
        )

        self.pool_net = PoolHiddenNet(
            pool_emb_dim=pool_emb_dim,
            tf_emb_dim=tf_emb_dim,
            mlp_dim=mlp_dim,
            bottleneck_dim=bottleneck_dim,
            activation=activation,
            batch_norm=batch_norm
        )

        if noise_dim[0] == 0:
            self.noise_dim = None
        else:
            self.noise_first_dim = noise_dim[0]

        # Decoder Hidden
        if pooling_type:
            input_dim = tf_emb_dim + bottleneck_dim
        else:
            input_dim = tf_emb_dim

        if self.mlp_decoder_needed():
            mlp_decoder_context_dims = [
                input_dim, mlp_dim, tf_emb_dim - self.noise_first_dim
            ]

            self.mlp_decoder_context = make_mlp(
                mlp_decoder_context_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout
            )

    def add_noise(self, _input, seq_start_end, user_noise=None):
        """
        Inputs:
        - _input: Tensor of shape (_, decoder_h_dim - noise_first_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - user_noise: Generally used for inference when you want to see
        relation between different types of noise and outputs.
        Outputs:
        - decoder_h: Tensor of shape (_, decoder_h_dim)
        """
        if not self.noise_dim:
            return _input
        if self.noise_mix_type == 'global':
            noise_shape = (_input.size(0), seq_start_end.size(0),) + self.noise_dim
        else:
            noise_shape = (_input.size(0), _input.size(1),) + self.noise_dim

        # log('noise_shape', noise_shape)

        if user_noise is not None:
            z_decoder = user_noise
        else:
            z_decoder = get_noise(noise_shape, self.noise_type, self.device)

        # log('z_decoder shape', z_decoder.shape)

        if self.noise_mix_type == 'global':
            _list = []
            for idx, (start, end) in enumerate(seq_start_end):
                start = start.item()
                end = end.item()
                _vec = z_decoder[:, idx].view(_input.size(0), 1, -1)
                _to_cat = _vec.repeat(1, end - start, 1)
                _list.append(torch.cat([_input[:, start:end], _to_cat], dim=2))
            decoder_h = torch.cat(_list, dim=1)
            return decoder_h

        decoder_h = torch.cat([_input, z_decoder], dim=1)

        return decoder_h

    def mlp_decoder_needed(self):
        if (
                self.noise_dim or self.pooling_type
        ):
            return True
        else:
            return False

    def forward(self, obs_traj, obs_traj_rel, seq_start_end, dec_inp, trg_att, user_noise=None):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        - obs_traj_rel: Tensor of shape (obs_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - user_noise: Generally used for inference when you want to see
        relation between different types of noise and outputs.
        Output:
        - pred_traj_rel: Tensor of shape (self.pred_len, batch, 2)
        """

        inp = obs_traj_rel.permute(1, 0, 2)[:, 1:, :].to(self.device)
        src_att = torch.ones((inp.shape[0], 1, inp.shape[1])).to(self.device)
        final_encoder_h = self.encoder(inp, src_att)
        final_encoder_h = final_encoder_h.permute(1, 0, 2)

        # Pool States
        if self.pooling_type:
            end_pos = obs_traj[1:, :, :]
            pool_h = self.pool_net(final_encoder_h, seq_start_end, end_pos)
            # Construct input hidden states for decoder
            mlp_decoder_context_input = torch.cat([final_encoder_h, pool_h], dim=2)
        else:
            mlp_decoder_context_input = final_encoder_h

        # log('mlp_decoder_context_input shape', mlp_decoder_context_input.shape)
        # Add Noise
        if self.mlp_decoder_needed():
            noise_input = self.mlp_decoder_context(mlp_decoder_context_input)
        else:
            noise_input = mlp_decoder_context_input

        final_encoder_h = self.add_noise(noise_input, seq_start_end, user_noise=user_noise)
        final_encoder_h = final_encoder_h.permute(1, 0, 2)

        # Predict Trajectory

        pred_traj_fake_rel = self.decoder(
            final_encoder_h,
            src_att,
            dec_inp,
            trg_att,
        )
        return pred_traj_fake_rel
