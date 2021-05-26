import torch
import torch.nn as nn

from sgan.models.Encoder import Encoder
from sgan.models.Decoder import Decoder
from sgan.models.Pooling import PoolHiddenNet, SocialPooling

from sgan.models.Utils import make_mlp, get_noise, log

class TrajectoryGenerator(nn.Module):
    def __init__(
        self, obs_len, pred_len, embedding_dim=64, encoder_h_dim=64,
        decoder_h_dim=128, mlp_dim=1024, num_layers=1, noise_dim=(0, ),
        noise_type='gaussian', noise_mix_type='ped', pooling_type=None,
        pool_every_timestep=True, dropout=0.0, bottleneck_dim=1024,
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

        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=encoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        self.decoder = Decoder(
            pred_len,
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
            noise_shape = (seq_start_end.size(0), ) + self.noise_dim
        else:
            noise_shape = (_input.size(0), ) + self.noise_dim

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
        #log('obs_traj_rel', obs_traj_rel)
        #log('obs_traj_rel type', type(obs_traj_rel))
        #log('obs_traj_rel shape', obs_traj_rel.shape)
        
        
        
        # Encode seq
        final_encoder_h = self.encoder(obs_traj_rel)
        #log('final_encoder_h', final_encoder_h)
        
        # Pool States
        if self.pooling_type:
            end_pos = obs_traj[-1, :, :]
            print('if self.pooling_type == true:')
            #log('end_pos:', end_pos)
        
            pool_h = self.pool_net(final_encoder_h, seq_start_end, end_pos)
            #log('pool_h:', pool_h)
        
            # Construct input hidden states for decoder
            mlp_decoder_context_input = torch.cat(
                [final_encoder_h.view(-1, self.encoder_h_dim), pool_h], dim=1)
            #log('mlp_decoder_context_input:', mlp_decoder_context_input)
        
        else:
            mlp_decoder_context_input = final_encoder_h.view(
                -1, self.encoder_h_dim)
            print('if self.pooling_type == false:')
            #log('mlp_decoder_context_input:', mlp_decoder_context_input)
        
        

        # Add Noise
        if self.mlp_decoder_needed():
            print('if self.mlp_decoder_needed() == true:')
            noise_input = self.mlp_decoder_context(mlp_decoder_context_input)
            #log('obs_traj:', obs_traj)
        
        else:
            print('if self.mlp_decoder_needed() == false:')
            
            noise_input = mlp_decoder_context_input
            #log('noise_input:', noise_input)
        
        decoder_h = self.add_noise(
            noise_input, seq_start_end, user_noise=user_noise)
        #log('decoder_h:', decoder_h)
        
        decoder_h = torch.unsqueeze(decoder_h, 0)
        #log('decoder_h:', decoder_h)
        
        
        decoder_c = torch.zeros(
            self.num_layers, batch, self.decoder_h_dim
        ).cuda()
        #log('decoder_c:', decoder_c)
        
        
        state_tuple = (decoder_h, decoder_c)
        #log('state_tuple:', state_tuple)
        
        last_pos = obs_traj[-1]
        #log('last_pos:', last_pos)
        
        last_pos_rel = obs_traj_rel[-1]
        #log('last_pos_rel:', last_pos_rel)
        
        # Predict Trajectory

        decoder_out = self.decoder(
            last_pos,
            last_pos_rel,
            state_tuple,
            seq_start_end,
        )
        #log('decoder_out:', decoder_out)
        
        pred_traj_fake_rel, final_decoder_h = decoder_out
        #log('pred_traj_fake_rel:', pred_traj_fake_rel)
        #log('final_decoder_h:', final_decoder_h)
        
        
        print('=================================================')
        
        return pred_traj_fake_rel
