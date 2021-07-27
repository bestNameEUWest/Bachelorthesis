from sgan.models.transformer.decoder import Decoder
from sgan.models.transformer.multihead_attention import MultiHeadAttention
from sgan.models.transformer.positional_encoding import PositionalEncoding
from sgan.models.transformer.pointerwise_feedforward import PointerwiseFeedforward
from sgan.models.transformer.encoder_decoder import EncoderDecoder
from sgan.models.transformer.encoder import Encoder
from sgan.models.transformer.encoder_layer import EncoderLayer
from sgan.models.transformer.decoder_layer import DecoderLayer
from sgan.models.transformer.high_lv_encoder import HighLVEncoder
from sgan.models.transformer.high_lv_decoder import HighLVDecoder

import torch.nn as nn
import copy
import math


class CustomTransformer:
    """
    Container class for separate calling of encoder and decoder modules.
    """

    def __init__(self, enc_inp_size, dec_inp_size, dec_out_size, n=6,
                 d_model=512, d_ff=2048, h=8, dropout=0.1):
        """Helper: Construct a model from hyperparameters."""
        c = copy.deepcopy
        attn = MultiHeadAttention(h, d_model)
        ff = PointerwiseFeedforward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        self.decoderLayer = DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout)

        self.encoder = HighLVEncoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), n),
            nn.Sequential(LinearEmbedding(enc_inp_size, d_model), c(position))
        )
        self.decoder = HighLVDecoder(
            Decoder(DecoderLayer(d_model, c(attn), c(attn),
                                 c(ff), dropout), n),
            nn.Sequential(LinearEmbedding(dec_inp_size, d_model), c(position)),
            Generator(d_model, dec_out_size)
        )

        # self.model = EncoderDecoder(
        #     Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        #     Decoder(DecoderLayer(d_model, c(attn), c(attn),
        #                          c(ff), dropout), N),
        #     nn.Sequential(LinearEmbedding(enc_inp_size,d_model), c(position)),
        #     nn.Sequential(LinearEmbedding(dec_inp_size,d_model), c(position)),
        #     Generator(d_model, dec_out_size))

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for (p_e, p_d) in zip(self.encoder.parameters(), self.decoder.parameters()):
            if p_e.dim() > 1:
                nn.init.xavier_uniform_(p_e)
            if p_d.dim() > 1:
                nn.init.xavier_uniform_(p_d)


class LinearEmbedding(nn.Module):
    def __init__(self, inp_size, d_model):
        super(LinearEmbedding, self).__init__()
        # lut => lookup table
        self.lut = nn.Linear(inp_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, out_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, out_size)

    def forward(self, x):
        return self.proj(x)
