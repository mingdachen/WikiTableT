import torch.nn as nn

from config import UNK_IDX, UNK_WORD, BOS_IDX, EOS_IDX, \
    BOC_IDX, BOV_IDX, BOQK_IDX, BOQV_IDX, SUBSEC_IDX, EOC_IDX
from transformer_xlm import CacheTransformer


class decoder_base(nn.Module):
    def __init__(self, vocab_size, embed_dim, dropout):
        super(decoder_base, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.embed = nn.Embedding(vocab_size, embed_dim)


class ctransformer(CacheTransformer):
    def __init__(self, vocab_size, encoder_size,
                 decoder_size, embed_dim, dropout, nlayer,
                 nhead, act_fn, max_len, share_embedding,
                 use_copy, use_entmax, *args, **kwargs):
        super(ctransformer, self).__init__(
            n_words=vocab_size,
            bos_index=BOS_IDX,
            eos_index=EOS_IDX,
            pad_index=UNK_IDX,
            emb_dim=embed_dim,
            n_heads=nhead,
            n_layers=nlayer,
            max_len=max_len,
            share_embedding=share_embedding,
            use_entmax=use_entmax,
            dropout=dropout,
            attention_dropout=dropout,
            use_copy=use_copy)

    def forward(self, encoder_outputs, encoder_mask,
                tgts, src_map, *args, **kwargs):
        logits = self.fwd(x=tgts, src_enc=encoder_outputs,
                          src_mask=encoder_mask, src_map=src_map)
        return logits

    def generate(self, encoder_outputs, encoder_mask, max_len,
                 min_len, top_k, top_p, src_map, src_tgt_vocab_map,
                 *args, **kwargs):
        return self._generate(
            src_enc=encoder_outputs, src_mask=encoder_mask,
            max_len=max_len, min_len=min_len,
            top_p=top_p, src_map=src_map, src_tgt_vocab_map=src_tgt_vocab_map)

    def generate_beam(self, encoder_output, encoder_mask,
                      beam_size, length_penalty=0.0,
                      early_stopping=False, trigram_blocking=False,
                      bos_index=1, eos_index=2, pad_index=1,
                      min_len=0, max_len=100, return_all=False,
                      src_map=None, src_tgt_vocab_map=None):
        return self._generate_beam(
            src_enc=encoder_output, src_mask=encoder_mask,
            beam_size=beam_size, length_penalty=length_penalty,
            early_stopping=early_stopping, min_len=min_len,
            max_len=max_len, trigram_blocking=trigram_blocking,
            return_all=return_all, src_map=src_map,
            src_tgt_vocab_map=src_tgt_vocab_map)


class ie_mask_transformer(decoder_base):
    def __init__(self, vocab_size, encoder_size,
                 decoder_size, embed_dim, dropout, nlayer,
                 nhead, act_fn, *args, **kwargs):
        super(ie_mask_transformer, self).__init__(
            vocab_size, embed_dim, dropout)

        # Define layers
        assert encoder_size == decoder_size, \
            "encoder size: {} != decoder size: {}"\
            .format(encoder_size, decoder_size)
        self.pos_embed = nn.Embedding(200, embed_dim)
        self.type_embed = nn.Embedding(500, embed_dim)
        self.if_hyp_embed = nn.Embedding(2, embed_dim)
        self.layers = nn.ModuleList(
            [nn.TransformerDecoderLayer(
                d_model=decoder_size,
                nhead=nhead,
                dim_feedforward=decoder_size * 4,
                activation=act_fn) for n in range(nlayer)])
        self.hid2vocab = nn.Linear(decoder_size, vocab_size)

    def forward(self, encoder_outputs, encoder_mask, inp, inp_mask, inp_pos,
                inp_type, inp_if_hyp, *args, **kwargs):

        tgt_vecs = self.embed(inp.long()) + \
            self.pos_embed(inp_pos.long()) + \
            self.type_embed(inp_type.long()) + \
            self.if_hyp_embed(inp_if_hyp.long())
        output = self.dropout(tgt_vecs.transpose(0, 1))
        encoder_outputs = encoder_outputs.transpose(0, 1)
        encoder_mask = (1 - encoder_mask).bool()
        inp_mask = (1 - inp_mask).bool()
        for layer in self.layers:
            output = layer(
                tgt=output,
                memory=encoder_outputs,
                tgt_key_padding_mask=inp_mask,
                memory_key_padding_mask=encoder_mask)

        logits = self.hid2vocab(output)
        return logits.transpose(0, 1)
