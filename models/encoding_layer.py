# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this open-source project.


""" Define the Seq2Seq Generation Network """
import numpy as np
import torch
import torch.nn as nn
# from v2.utils.pose import BOS_POSE
from models.layers import MultiHeadAttention, PositionwiseFeedForward


def get_non_pad_mask(seq):
    assert seq.dim() == 3
    non_pad_mask = torch.abs(seq).sum(2).ne(0).type(torch.float)
    return non_pad_mask.unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    """ For masking out the padding part of key sequence. """
    len_q = seq_q.size(1)
    padding_mask = torch.abs(seq_k).sum(2).eq(0)  # sum the vector of last dim and then judge
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask


def get_subsequent_mask(seq, sliding_windown_size):
    """ For masking out the subsequent info. """
    batch_size, seq_len, _ = seq.size()
    mask = torch.ones((seq_len, seq_len), device=seq.device, dtype=torch.uint8)
    mask = torch.triu(mask, diagonal=-sliding_windown_size)
    mask = torch.tril(mask, diagonal=sliding_windown_size)
    mask = 1 - mask
    # print(mask)
    return mask.bool()


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


class EncoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, d_condition=None):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout, d_condition=d_condition)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout, d_condition=d_condition)

    def forward(self, enc_input, slf_attn_mask=None, non_pad_mask=None, condition=None):

        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask, condition=condition)
        # enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output, condition=condition)
        # enc_output *= non_pad_mask

        return enc_output, enc_slf_attn
