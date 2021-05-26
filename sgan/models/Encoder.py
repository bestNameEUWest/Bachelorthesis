import torch
import torch.nn as nn

from sgan.models.Utils import log

class Encoder(nn.Module):
    """Encoder is part of both TrajectoryGenerator and
    TrajectoryDiscriminator"""
    def __init__(
        self, embedding_dim=64, h_dim=64, mlp_dim=1024, num_layers=1,
        dropout=0.0
    ):
        super(Encoder, self).__init__()

        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            embedding_dim, h_dim, num_layers, dropout=dropout
        )

        self.spatial_embedding = nn.Linear(2, embedding_dim)

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
        # Encode observed Trajectory
        print('=================================================')
        print('Encoder fw')
        
        #log('obs_traj', obs_traj)
        #log('obs_traj.shape', obs_traj.shape)        
        
        batch = obs_traj.size(1)
        #log('batch', batch)
        
        #log('obs_traj cont view(-1, 2)', obs_traj.contiguous().view(-1, 2))
        #log('obs_traj cont view(-1, 2) shape', obs_traj.contiguous().view(-1, 2).shape)
        
        obs_traj_embedding = self.spatial_embedding(obs_traj.contiguous().view(-1, 2))
        #log('obs_traj_embedding', obs_traj_embedding)
        #log('obs_traj_embedding.shape', obs_traj_embedding.shape)
        
        obs_traj_embedding = obs_traj_embedding.view(
            -1, batch, self.embedding_dim
        )
        #log('obs_traj_embedding view', obs_traj_embedding)
        #log('obs_traj_embedding view shape', obs_traj_embedding.shape)
        
        
        state_tuple = self.init_hidden(batch)
        #log('state_tuple', state_tuple)
        
        output, state = self.encoder(obs_traj_embedding, state_tuple)
        #log('output', output)
        #log('output.size()', output.size())
        #log('state', state)
        
        final_h = state[0]
        #log('final_h:', final_h)
        #log('final_h.size():', final_h.size())
        print('=================================================')
        
        exit()
        
        return final_h
