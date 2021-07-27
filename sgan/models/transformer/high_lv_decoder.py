# -*- coding: utf-8 -*-
# date: 2018-11-29 19:56
import torch.nn as nn


class HighLVDecoder(nn.Module):
    """
    High level Transformer decoder to be used in the generation step.
    """

    def __init__(self, decoder, tgt_embed, generator):
        super(HighLVDecoder, self).__init__()
        self.decoder = decoder
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, memory, src_mask, tgt, tgt_mask):
        """
        Take in and process memory, input, masked src and target sequences.
        """
        return self.generator(self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask))
