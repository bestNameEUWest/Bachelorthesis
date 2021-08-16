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
import math


class TransformerEncoder(nn.Module):
    def __init__(self, enc_inp_size, n=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        """Helper: Construct a model from hyperparameters."""

        self.encoder = HighLVEncoder(
            Encoder(EncoderLayer(
                d_model,
                MultiHeadAttention(h, d_model),
                PointerwiseFeedforward(d_model, d_ff, dropout),
                dropout
            ), n),
            nn.Sequential(LinearEmbedding(enc_inp_size, d_model), PositionalEncoding(d_model, dropout))
        )

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p_e in self.encoder.parameters():
            if p_e.dim() > 1:
                nn.init.xavier_uniform_(p_e)

    def forward(self, objs_traj, src_att):
        self.encoder(objs_traj, src_att)


class TransformerDecoder(nn.Module):
    def __init__(self, dec_inp_size, dec_out_size, n=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        """Helper: Construct a model from hyperparameters."""

        self.decoder = HighLVDecoder(
            Decoder(DecoderLayer(
                d_model,
                MultiHeadAttention(h, d_model),
                MultiHeadAttention(h, d_model),
                PointerwiseFeedforward(d_model, d_ff, dropout),
                dropout
            ), n),
            nn.Sequential(LinearEmbedding(dec_inp_size, d_model), PositionalEncoding(d_model, dropout)),
            Generator(d_model, dec_out_size)
        )

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p_d in self.decoder.parameters():
            if p_d.dim() > 1:
                nn.init.xavier_uniform_(p_d)

    def forward(self, encoder_h, src_att, dec_inp, trg_att):
        self.decoder(encoder_h, src_att, dec_inp, trg_att)


class LinearEmbedding(nn.Module):
    def __init__(self, inp_size, d_model):
        super(LinearEmbedding, self).__init__()
        # lut => lookup table
        self.lut = nn.Linear(inp_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class Generator(nn.Module):
    """
    Define standard linear + softmax generation step.
    """

    def __init__(self, d_model, out_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, out_size)

    def forward(self, x):
        return self.proj(x)
