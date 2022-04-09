#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math
from utils import str2bool
from models.encoding_layer import EncoderLayer, get_sinusoid_encoding_table


class MusicEncoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(self, max_seq_len=100, input_size=35, d_word_vec=640,
                 n_layers=3, n_head=10, d_k=64, d_v=64,
                 d_model=640, d_inner=1920, dropout=0.1):

        super().__init__()

        self.d_model = d_model
        n_position = max_seq_len + 1

        self.src_emb = nn.Linear(input_size, d_word_vec)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.outlen = d_model

    def get_outlen(self):
        return self.outlen

    def forward(self, src_seq, mask=None, return_attns=False):

        enc_slf_attn_list = []
        src_pos = torch.tensor([i for i in range(src_seq.shape[1])]).unsqueeze(0).cuda()

        # -- Forward
        enc_output = self.src_emb(src_seq) + self.position_enc(src_pos)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=mask)

            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output


class DanceModel(nn.Module):
    def __init__(self, args):
        super(DanceModel, self).__init__()
        self.args = args
        self.use_music = args.use_music
        hidden_feature = args.hidden_feature
        p_dropout = args.p_dropout
        num_stage = args.num_stage
        node_n = args.node_n
        d_model = args.d_model
        n_position = args.input_n + args.output_n

        self.num_stage = num_stage

        if self.use_music:
            self.music_encoder = MusicEncoder()
        else:
            self.music_encoder = None

        self.src_emb = nn.Linear(node_n, d_model)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_model, padding_idx=0),
            freeze=True)
        self.mod_embedding = nn.Embedding(3, d_model)

        self.pose_stack = nn.ModuleList([
            EncoderLayer(d_model, hidden_feature, args.n_head,
                         args.d_k, args.d_v,
                         dropout=p_dropout, d_condition=None)
            for _ in range(3)])

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, hidden_feature, args.n_head,
                         args.d_k, args.d_v,
                         dropout=p_dropout, d_condition=None)
            for _ in range(num_stage)])

        self.pose_decode = nn.Linear(d_model, node_n)

        nn.init.normal_(self.mod_embedding.weight, mean=0, std=args.initializer_range)

        self.l2loss = nn.MSELoss()
        self.l1loss = nn.L1Loss()
        ## 重要！模型初始化

    @staticmethod
    def modify_commandline_options(parser):
        parser.add_argument("--d_model", type=int, default=640)
        parser.add_argument("--d_k", type=int, default=64)
        parser.add_argument("--d_v", type=int, default=64)
        parser.add_argument("--n_head", type=int, default=10)
        parser.add_argument("--hidden_feature", type=int, default=1920)  # 256
        parser.add_argument("--p_dropout", type=float, default=0.1)
        parser.add_argument("--num_stage", type=int, default=12)
        parser.add_argument("--node_n", type=int, default=69)  # 18*3, 67
        parser.add_argument("--use_music", type=str2bool, default=True)
        parser.add_argument("--fusion_type", type=str, default="cln")
        parser.add_argument("--condition_step", type=int, default=10)
        parser.add_argument("--lambda_v", type=float, default=0.03)
        parser.add_argument("--initializer_range", type=float, default=0.02)
        # parser = MusicEncoder.modify_commandline_options(parser)
        return parser

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def run(self, x, music):
        if self.use_music:
            music_f = self.music_encoder(music)
        else:
            music_f = None

        input_n = self.args.input_n
        output_n = self.args.output_n
        src_pos = torch.tensor([i for i in range(input_n)]).unsqueeze(0).cuda()
        pos_embed = self.position_enc(src_pos)

        mod_id = torch.tensor([0 for _ in range(input_n)] + [1 for _ in range(input_n * 2)]).unsqueeze(0).cuda()
        mod_embed = self.mod_embedding(mod_id)

        enc_output = self.src_emb(x) + pos_embed
        for enc_layer in self.pose_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, condition=None)

        multimod_output = torch.cat((enc_output, music_f), dim=1)
        multimod_output = multimod_output + mod_embed

        for enc_layer in self.layer_stack:
            multimod_output, enc_slf_attn = enc_layer(multimod_output, condition=None)

        pose_result = self.pose_decode(multimod_output[:, -output_n:, :])
        return pose_result

    def forward(self, music, **kwargs):
        """

        :param x: [b, t, node]
        :param music: [b, t, dim_music]
        :return:
        """
        epoch = kwargs.get("epoch", None)
        if epoch is None:
            epoch = self.train_epoch
        else:
            self.train_epoch = epoch

        targets = kwargs["targets"]

        input_n = self.args.input_n
        output_n = self.args.output_n
        music_input_length = input_n * 2
        seq_len = targets.shape[1]

        groundtruth_mask = torch.ones(seq_len, self.args.condition_step)
        prediction_mask = torch.zeros(seq_len, int(epoch * self.args.lambda_v))
        mask = torch.cat([prediction_mask, groundtruth_mask], 1).view(-1)[:seq_len]  # for random

        total_cir = self.args.total_len - music_input_length + 1
        total_loss = {}
        for i in range(total_cir):
            target_pose = targets[:, i + input_n:i + input_n + output_n, :]
            music_input = music[:, i:i + music_input_length, :]
            if mask[i] == 1 or i == 0:
                input_pose = targets[:, i:i + input_n, :]
            else:
                input_pose = torch.cat([input_pose[:, 1:, :], pred_pose[:, :1, :]], 1)

            pred_pose = self.run(input_pose, music=music_input)
            loss = self.compute_loss(pred_pose, target_pose)
            for k, v in loss.items():
                if k not in total_loss.keys():
                    total_loss[k] = v / total_cir
                else:
                    total_loss[k] += v / total_cir

        return total_loss

    def compute_loss(self, pred_pose, target_pose):
        l2_loss = self.l2loss(pred_pose[:, :, :63], target_pose[:, :, :63])
        root_loss = self.l1loss(pred_pose[:, :, 63:67], target_pose[:, :, 63:67])
        contact_loss = self.l1loss(pred_pose[:, :, 67:], target_pose[:, :, 67:])
        loss = l2_loss + root_loss * 0.3 + contact_loss * 0.1
        return {"l2_loss": l2_loss, "root_loss": root_loss, "contact_loss": contact_loss, "loss": loss}

    def input_padding(self, input_pose):
        i_idx = [self.args.input_n - 1 for _ in range(self.args.output_n)]
        padding = input_pose[:, i_idx, :]
        # set delta value to be zero, no need to move around
        padding[:, 64:67] *= 0.
        input_with_padding = torch.cat([input_pose, padding], dim=1)
        return input_with_padding

    def generate(self, targets, music):
        """
        :param x: [input_n, node]
        :param music: [t_all, dim_music]
        :return:
        """
        input_n = self.args.input_n
        output_n = self.args.output_n
        music_total_length = music.shape[1]
        music_input_length = 2 * input_n

        total_cir = music_total_length - music_input_length + 1
        output_pose = targets[:, :input_n, :]
        for i in range(total_cir):
            music_input = music[:, i:i + music_input_length, :]
            if i == 0:
                input_pose = targets[:, i:i + input_n, :]
            else:
                input_pose = torch.cat([input_pose[:, 1:, :], pred_pose[:, :1, :]], 1)

            # input_pose_padding = self.input_padding(input_pose)
            pred_pose = self.run(input_pose, music=music_input)
            pred_pose = pred_pose.detach()
            output_pose = torch.cat([output_pose, pred_pose[:, :1, :]], 1)

        return output_pose[:, :, :-2]
