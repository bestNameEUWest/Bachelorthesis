import torch
import torch.nn as nn

from sgan.models.transformer.custom_transformer import TransformerEncoder, TransformerDecoder
from sgan.models.SGANEncoder import SGANEncoder
from sgan.models.SGANDecoder import SGANDecoder
from sgan.models.Pooling import PoolHiddenNet, SocialPooling

from sgan.models.Utils import make_mlp, get_noise, log


class TrajectoryGenerator(nn.Module):
    def __init__(
            self, obs_len, pred_len, device, feature_count=2, embedding_dim=64, encoder_h_dim=64,
            decoder_h_dim=128, mlp_dim=1024, layer_count=1, noise_dim=(0,),
            noise_type='gaussian', noise_mix_type='ped', pooling_type=None,
            pool_every_timestep=True, dropout=0.0, heads=8, bottleneck_dim=1024,
            activation='relu', batch_norm=True, neighborhood_size=2.0, grid_size=8
    ):
        super(TrajectoryGenerator, self).__init__()

        if pooling_type and pooling_type.lower() == 'none':
            pooling_type = None

        self.device = device
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.mlp_dim = mlp_dim
        self.encoder_h_dim = encoder_h_dim
        self.decoder_h_dim = decoder_h_dim
        self.embedding_dim = embedding_dim
        self.noise_dim = noise_dim
        self.num_layers = num_layers
        self.noise_type = noise_type
        self.noise_mix_type = noise_mix_type
        self.pooling_type = pooling_type
        self.noise_first_dim = 0
        self.pool_every_timestep = pool_every_timestep
        self.bottleneck_dim = 1024

        # encoder_h_dim is our d_model in the Transformer
        self.custom_transformer = CustomTransformer(
            enc_inp_size=feature_count,
            dec_inp_size=3,
            dec_out_size=3,
            n=num_layers,
            d_model=encoder_h_dim,
            h=heads,
            dropout=dropout
        )
        # log('Trajectory g sgan enc encoder_h_dim size', encoder_h_dim)
        self.encoder = SGANEncoder(
            device=device,
            feature_count=feature_count,
            layer_count=layer_count,
            emb_size=tf_emb_size,
            ff_size=tf_ff_size,
            heads=heads,
            dropout=dropout
        )
        # log('Trajectory g sgan dec decoder_h_dim size', decoder_h_dim)
        self.decoder = SGANDecoder(
            pred_len,
            device=device,
            tf_decoder=self.custom_transformer.decoder,
        )

        if pooling_type == 'pool_net':
            self.pool_net = PoolHiddenNet(
                embedding_dim=self.embedding_dim,
                h_dim=encoder_h_dim,
                mlp_dim=mlp_dim,
                bottleneck_dim=bottleneck_dim,
                activation=activation,
                batch_norm=batch_norm
            )
        elif pooling_type == 'spool':
            self.pool_net = SocialPooling(
                h_dim=encoder_h_dim,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout,
                neighborhood_size=neighborhood_size,
                grid_size=grid_size
            )

        if self.noise_dim[0] == 0:
            self.noise_dim = None
        else:
            # log('noise_dim', noise_dim)
            self.noise_first_dim = noise_dim[0]

        # Decoder Hidden
        if pooling_type:
            input_dim = encoder_h_dim + bottleneck_dim
        else:
            input_dim = encoder_h_dim

        if self.mlp_decoder_needed():
            # log('decoder_h_dim', decoder_h_dim)
            mlp_decoder_context_dims = [
                input_dim, mlp_dim, decoder_h_dim - self.noise_first_dim
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
                self.noise_dim or self.pooling_type or
                self.encoder_h_dim != self.decoder_h_dim
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

        final_encoder_h, src_att = self.encoder(obs_traj_rel)
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

        decoder_h = self.add_noise(noise_input, seq_start_end, user_noise=user_noise)

        test = nn.Linear(self.decoder_h_dim, self.decoder_h_dim).to(self.device)
        decoder_h = test(decoder_h)

        # Predict Trajectory
        state_tuple = (decoder_h, None)
        pred_traj_fake_rel = self.decoder(
            state_tuple,
            src_att,
            dec_inp,
            trg_att,
        )
        return pred_traj_fake_rel
