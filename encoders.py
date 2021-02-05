import torch
import torch.nn as nn
import torch.nn.functional as F
from entmax import entmax_bisect


class encoder_base(nn.Module):
    def __init__(self, vocab_size, type_vocab_size,
                 embed_dim, dropout, max_pos_len,
                 *args, **kwargs):
        super(encoder_base, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.type_embed = nn.Embedding(type_vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_pos_len, embed_dim)
        self.if_hyp_embed = nn.Embedding(2, embed_dim)


class transformer(encoder_base):
    """encoder module for the input linearized table"""
    def __init__(self, vocab_size, type_vocab_size, embed_dim, dropout,
                 nlayer, nhead, hidden_size, act_fn,
                 max_pos_len, *args, **kwargs):
        super(transformer, self).__init__(
            vocab_size, type_vocab_size, embed_dim, dropout, max_pos_len)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation=act_fn) for n in range(nlayer)])

    def forward(self, data, data_mask, data_pos, data_type, if_hyp):
        data_vec = self.embed(data.long())
        data_pos_vec = self.pos_embed(data_pos.long())
        data_type_vec = self.type_embed(data_type.long())
        data_if_hyp_vec = self.if_hyp_embed(if_hyp.long())
        output = self.dropout(
            data_vec + data_pos_vec + data_type_vec + data_if_hyp_vec)\
            .transpose(0, 1)
        data_mask = (1 - data_mask).bool()
        for layer in self.layers:
            output = layer(src=output, src_key_padding_mask=data_mask)
        return output.transpose(0, 1)


class ie_transformer(nn.Module):
    """encoder module for the backward model in cyclic training"""
    def __init__(self, vocab_size, embed_dim, dropout,
                 nlayer, nhead, hidden_size, act_fn, max_len, *args, **kwargs):
        super(ie_transformer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation=act_fn) for n in range(nlayer)])

    def forward(self, data, data_mask):
        pos_ids = torch.arange(data.size(1), device=data.device)\
            .unsqueeze(0).expand(data.size(0), -1)
        data_vec = self.embed(data.long())
        data_pos_vec = self.pos_embed(pos_ids.long())
        output = self.dropout(data_vec + data_pos_vec).transpose(0, 1)
        data_mask = (1 - data_mask).bool()
        for layer in self.layers:
            output = layer(src=output, src_key_padding_mask=data_mask)

        return output.transpose(0, 1)

    def softmax_forward(self, logits, data_mask):
        """logits are softmaxed"""
        pos_ids = torch.arange(data_mask.size(1), device=data_mask.device)\
            .unsqueeze(0).expand(data_mask.size(0), -1)

        data_vec = torch.matmul(logits, self.embed.weight)

        data_pos_vec = self.pos_embed(pos_ids.long())
        output = self.dropout(data_vec + data_pos_vec).transpose(0, 1)
        data_mask = (1 - data_mask).bool()
        for layer in self.layers:
            output = layer(src=output, src_key_padding_mask=data_mask)
        return output.transpose(0, 1)

    def mix_forward(self, logits, data_mask):
        """logits have not been softmaxed"""
        pos_ids = torch.arange(data_mask.size(1), device=data_mask.device)\
            .unsqueeze(0).expand(data_mask.size(0), -1)

        data_vec = torch.matmul(F.softmax(logits, -1), self.embed.weight)

        data_pos_vec = self.pos_embed(pos_ids.long())
        output = self.dropout(data_vec + data_pos_vec).transpose(0, 1)
        data_mask = (1 - data_mask).bool()
        for layer in self.layers:
            output = layer(src=output, src_key_padding_mask=data_mask)
        return output.transpose(0, 1)

    def entmax_forward(self, logits, data_mask):
        pos_ids = torch.arange(data_mask.size(1), device=data_mask.device)\
            .unsqueeze(0).expand(data_mask.size(0), -1)

        bs, sq_len, dim = logits.shape
        logits = entmax_bisect(logits.reshape(bs * sq_len, dim), 1.2)
        data_vec = torch.matmul(logits.reshape(bs, sq_len, dim),
                                self.embed.weight)

        data_pos_vec = self.pos_embed(pos_ids.long())
        output = self.dropout(data_vec + data_pos_vec).transpose(0, 1)
        data_mask = (1 - data_mask).bool()
        for layer in self.layers:
            output = layer(src=output, src_key_padding_mask=data_mask)
        return output.transpose(0, 1)
