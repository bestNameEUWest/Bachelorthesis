# -*- coding: utf-8 -*-
# date: 2018-11-29 19:56
import torch.nn as nn


class HighLVEncoder(nn.Module):
    """
    High level Transformer encoder which returns the hidden state of the last low level encoder.
    """

    def __init__(self, encoder, src_embed):
        super(HighLVEncoder, self).__init__()
        self.encoder = encoder
        self.src_embed = src_embed

    def forward(self, src, src_mask):
        """
        Take in and process masked src and target sequences.
        """
        return self.encoder(self.src_embed(src), src_mask)
