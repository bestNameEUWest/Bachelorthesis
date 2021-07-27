import torch
import torch.nn as nn

from sgan.models.transformer.custom_transformer import CustomTransformer
from sgan.models.SGANEncoder import SGANEncoder
from sgan.models.SGANDecoder import SGANDecoder
from sgan.models.Pooling import PoolHiddenNet, SocialPooling

from sgan.models.Utils import make_mlp, get_noise, log


class TrajectoryGenerator(nn.Module):
    def __init__(
            self, obs_len, pred_len, encoder_type, feature_count=2, embedding_dim=64, encoder_h_dim=64,
            decoder_h_dim=128, mlp_dim=1024, num_layers=1, noise_dim=(0,),
            noise_type='gaussian', noise_mix_type='ped', pooling_type=None,
            pool_every_timestep=True, dropout=0.0, heads=8, bottleneck_dim=1024,
            activation='relu', batch_norm=True, neighborhood_size=2.0, grid_size=8
    ):
        super(TrajectoryGenerator, self).__init__()

        if pooling_type and pooling_type.lower() == 'none':
            pooling_type = None

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

        self.encoder = SGANEncoder(
            injected_encoder=self.custom_transformer.encoder,
            embedding_dim=embedding_dim,
            h_dim=encoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        self.decoder = SGANDecoder(
            pred_len,
            injected_decoder=self.custom_transformer.decoder,
            embedding_dim=embedding_dim,
            h_dim=decoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            pool_every_timestep=pool_every_timestep,
            dropout=dropout,
            bottleneck_dim=bottleneck_dim,
            activation=activation,
            batch_norm=batch_norm,
            pooling_type=pooling_type,
            grid_size=grid_size,
            neighborhood_size=neighborhood_size
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
            self.noise_first_dim = noise_dim[0]

        # Decoder Hidden
        if pooling_type:
            input_dim = encoder_h_dim + bottleneck_dim
        else:
            input_dim = encoder_h_dim

        if self.mlp_decoder_needed():
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
            noise_shape = (seq_start_end.size(0),) + self.noise_dim
        else:
            noise_shape = (_input.size(0),) + self.noise_dim

        if user_noise is not None:
            z_decoder = user_noise
        else:
            z_decoder = get_noise(noise_shape, self.noise_type)

        if self.noise_mix_type == 'global':
            _list = []
            for idx, (start, end) in enumerate(seq_start_end):
                start = start.item()
                end = end.item()
                _vec = z_decoder[idx].view(1, -1)
                _to_cat = _vec.repeat(end - start, 1)
                _list.append(torch.cat([_input[start:end], _to_cat], dim=1))
            decoder_h = torch.cat(_list, dim=0)
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

    def forward(self, obs_traj, obs_traj_rel, seq_start_end, user_noise=None):
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

        print('=================================================')
        print('TrajGen fw:')

        batch = obs_traj_rel.size(1)
        # log('obs_traj_rel shape', obs_traj_rel.shape)

        # Encode seq
        final_encoder_h, final_encoder_h_lstm = self.encoder(obs_traj_rel)
        final_encoder_h = final_encoder_h.permute(1, 0, 2)
        log('final_encoder_h shape', final_encoder_h.shape)
        log('final_encoder_h_lstm shape', final_encoder_h_lstm.shape)

        # Pool States
        if self.pooling_type:
            end_pos = obs_traj[1:, :, :]

            pool_h_lstm = self.pool_net(final_encoder_h_lstm, seq_start_end, end_pos)
            pool_h = self.pool_net(final_encoder_h, seq_start_end, end_pos)

            log('pool_h_lstm shape:', pool_h_lstm.shape)
            log('pool_h shape:', pool_h.shape)

            # Construct input hidden states for decoder
            mlp_decoder_context_input_lstm = torch.cat(
                [final_encoder_h_lstm.view(-1, self.encoder_h_dim), pool_h_lstm], dim=1)
            # log('...lstm is contiguous', final_encoder_h_lstm.is_contiguous())
            # log('...tf is contiguous', final_encoder_h.is_contiguous())
            log('mlp_decoder_context_input_lstm shape', mlp_decoder_context_input_lstm.shape)
            # log('mlp_decoder_context_input_lstm', mlp_decoder_context_input_lstm)
            mlp_decoder_context_input = torch.cat([final_encoder_h, pool_h], dim=2)
            log('mlp_decoder_context_input shape', mlp_decoder_context_input.shape)

        else:
            mlp_decoder_context_input = final_encoder_h

        # Add Noise
        if self.mlp_decoder_needed():
            noise_input_lstm = self.mlp_decoder_context(mlp_decoder_context_input_lstm)
            noise_input = self.mlp_decoder_context(mlp_decoder_context_input)
            # log('obs_traj:', obs_traj)

        else:
            noise_input = mlp_decoder_context_input
            noise_input_lstm = mlp_decoder_context_input_lstm
            # log('noise_input:', noise_input)

        decoder_h_lstm = self.add_noise(noise_input_lstm, seq_start_end, user_noise=user_noise)
        decoder_h_lstm = torch.unsqueeze(decoder_h_lstm, 0)
        log('decoder_h_lstm shape', decoder_h_lstm.shape)
        decoder_c_lstm = torch.zeros(self.num_layers, batch, self.decoder_h_dim).cuda()

        decoder_h = self.add_noise(noise_input, seq_start_end, user_noise=user_noise)
        log('decoder_h shape', decoder_h.shape)

        state_tuple = (decoder_h_lstm, decoder_c_lstm)
        last_pos = obs_traj[-1]
        last_pos_rel = obs_traj_rel[-1]

        # Predict Trajectory

        decoder_out_lstm = self.decoder(
            last_pos,
            last_pos_rel,
            state_tuple,
            seq_start_end,
        )

        state_tuple = (decoder_h, None)
        decoder_out = self.decoder(
            last_pos,
            last_pos_rel,
            state_tuple,
            seq_start_end,
        )
        # log('decoder_out:', decoder_out)

        pred_traj_fake_rel, final_decoder_h = decoder_out_lstm

        print('=================================================')

        exit()
        return pred_traj_fake_rel
