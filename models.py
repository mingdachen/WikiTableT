import os

import torch
import torch.nn.functional as F
import torch.nn as nn

import model_utils
import encoders
import decoders

from decorators import auto_init_args, auto_init_pytorch


class Base(nn.Module):
    def __init__(self, iter_per_epoch, experiment):
        super(Base, self).__init__()
        self.expe = experiment
        self.iter_per_epoch = iter_per_epoch
        self.eps = self.expe.config.eps
        self.expe.log.info("use_entmax: {}"
                           .format(self.expe.config.use_entmax))
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version
            # which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def to_tensor(self, inputs):
        if torch.is_tensor(inputs):
            return inputs.clone().detach().to(self.device)
        else:
            return torch.tensor(inputs, device=self.device)

    def to_tensors(self, *inputs):
        return [self.to_tensor(inputs_) if inputs_ is not None and inputs_.size
                else None for inputs_ in inputs]

    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_all_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def optimize(self, loss, update_param):
        loss.backward()
        if update_param:
            if self.expe.config.gclip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.parameters(), self.expe.config.gclip)
            self.opt.step()
            if self.expe.config.wstep:
                self.scheduler.step()
            self.opt.zero_grad()

    def init_optimizer(self, opt_type, learning_rate, weight_decay):
        if opt_type.lower() == "adam":
            optimizer = torch.optim.Adam
        elif opt_type.lower() == "rmsprop":
            optimizer = torch.optim.RMSprop
        elif opt_type.lower() == "sgd":
            optimizer = torch.optim.SGD
        else:
            raise NotImplementedError("invalid optimizer: {}".format(opt_type))

        opt = optimizer(
            params=filter(
                lambda p: p.requires_grad, self.parameters()
            ),
            weight_decay=weight_decay,
            lr=learning_rate)

        if self.expe.config.wstep:
            self.scheduler = \
                model_utils.get_linear_schedule_with_warmup(
                    opt, self.expe.config.wstep,
                    self.expe.config.n_epoch * self.iter_per_epoch)
            self.expe.log.info(
                "training with learning rate scheduler - "
                "iterations per epoch: {}, total epochs: {}"
                .format(self.iter_per_epoch, self.expe.config.n_epoch))
        return opt

    def save(self, dev_bleu, test_bleu, epoch, iteration=None, name="best"):
        save_path = os.path.join(self.expe.experiment_dir, name + ".ckpt")
        checkpoint = {
            "dev_bleu": dev_bleu,
            "test_bleu": test_bleu,
            "epoch": epoch,
            "iteration": iteration,
            "state_dict": self.state_dict(),
            "opt_state_dict": self.opt.state_dict(),
            "config": self.expe.config
        }
        if self.expe.config.wstep:
            checkpoint["lr_scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(checkpoint, save_path)
        self.expe.log.info("model saved to {}".format(save_path))

    def load(self, checkpointed_state_dict=None, name="best", path=None):
        if checkpointed_state_dict is None:
            base_path = self.expe.experiment_dir if path is None else path
            save_path = os.path.join(base_path, name + ".ckpt")
            checkpoint = torch.load(save_path,
                                    map_location=lambda storage, loc: storage)
            self.load_state_dict(checkpoint['state_dict'])
            self.opt.load_state_dict(checkpoint.get("opt_state_dict"))
            if self.expe.config.wstep:
                self.scheduler.load_state_dict(
                    checkpoint["lr_scheduler_state_dict"])
            self.expe.log.info("model loaded from {}".format(save_path))
            self.to(self.device)
            for state in self.opt.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
            self.expe.log.info("transferred model to {}".format(self.device))
            return checkpoint.get('epoch', 0), \
                checkpoint.get('iteration', 0), \
                checkpoint.get('dev_bleu', 0), \
                checkpoint.get('test_bleu', 0)
        else:
            self.load_state_dict(checkpointed_state_dict)
            self.expe.log.info("model loaded from checkpoint.")
            self.to(self.device)
            self.expe.log.info("transferred model to {}".format(self.device))


class BasicCyclicAttnSplitMask(Base):
    @auto_init_pytorch
    @auto_init_args
    def __init__(
            self, vocab_size, type_vocab_size,
            embed_dim, iter_per_epoch, use_entmax, experiment):
        super(BasicCyclicAttnSplitMask, self).__init__(
            iter_per_epoch, experiment)
        self.encode = getattr(encoders, self.expe.config.encoder_type)(
            embed_dim=embed_dim,
            nlayer=self.expe.config.elayer,
            nhead=self.expe.config.nhead,
            hidden_size=self.expe.config.ensize,
            dropout=self.expe.config.dp,
            vocab_size=vocab_size,
            type_vocab_size=type_vocab_size,
            max_pos_len=self.expe.config.max_encoder_len,
            act_fn=self.expe.config.act_fn)

        self.decode = getattr(decoders, self.expe.config.decoder_type)(
            embed_dim=embed_dim,
            encoder_size=self.expe.config.ensize,
            decoder_size=self.expe.config.desize,
            dropout=self.expe.config.dp,
            nlayer=self.expe.config.dlayer,
            nhead=self.expe.config.nhead,
            type_vocab_size=type_vocab_size,
            vocab_size=vocab_size,
            max_len=self.expe.config.max_decoder_len,
            act_fn=self.expe.config.act_fn,
            use_copy=self.expe.config.use_copy,
            use_entmax=use_entmax,
            share_embedding=self.expe.config.share_decoder_embedding)

        self.cyclic_encode = encoders.ie_transformer(
            embed_dim=embed_dim,
            nlayer=self.expe.config.bwdelayer,
            nhead=self.expe.config.bwdnhead,
            hidden_size=self.expe.config.ensize,
            dropout=self.expe.config.dp,
            vocab_size=vocab_size,
            type_vocab_size=type_vocab_size,
            max_len=self.expe.config.max_decoder_len,
            act_fn=self.expe.config.act_fn)

        self.cyclic_decode = decoders.ie_mask_transformer(
            embed_dim=embed_dim,
            encoder_size=self.expe.config.ensize,
            decoder_size=self.expe.config.desize,
            attn_size=self.expe.config.asize,
            dropout=self.expe.config.dp,
            nlayer=self.expe.config.bwddlayer,
            nhead=self.expe.config.bwdnhead,
            max_len=self.expe.config.max_encoder_len,
            vocab_size=vocab_size,
            act_fn=self.expe.config.act_fn)

        if self.expe.config.use_copy:
            self.loss_fn = model_utils.CopyGeneratorLoss(
                vocab_size=vocab_size,
                force_copy=self.expe.config.force_copy)

    def forward(
            self, data, data_mask, data_pos, data_type, data_if_hyp,
            data_src_vocab, data_src_tgt_vocab_map,
            tgt_input_data, tgt_input_data_mask, tgt_input_data_pos,
            tgt_input_data_type, tgt_input_data_if_hyp,
            tgt_output_data, tgt_output_mask,
            tgt_input, tgt_label, tgt_mask, tgt_src_vocab):

        data, data_mask, data_pos, data_type, data_if_hyp, \
            data_src_vocab, data_src_tgt_vocab_map, \
            tgt_input_data, tgt_input_data_mask, tgt_input_data_pos, \
            tgt_input_data_type, tgt_input_if_hyp, \
            tgt_output_data, tgt_output_mask, \
            tgt_input, tgt_label, tgt_mask, tgt_src_vocab = \
            self.to_tensors(data, data_mask, data_pos, data_type,
                            data_if_hyp, data_src_vocab,
                            data_src_tgt_vocab_map,
                            tgt_input_data, tgt_input_data_mask,
                            tgt_input_data_pos, tgt_input_data_type,
                            tgt_input_data_if_hyp,
                            tgt_output_data, tgt_output_mask,
                            tgt_input, tgt_label, tgt_mask, tgt_src_vocab)
        data_vec = self.encode(
            data, data_mask, data_pos, data_type, data_if_hyp)

        pred_probs, _ = self.decode(
            encoder_outputs=data_vec,
            encoder_mask=data_mask,
            tgts=tgt_input,
            src_map=data_src_vocab)

        if self.expe.config.use_copy:
            loss = self.loss_fn(
                scores=pred_probs,
                align=tgt_src_vocab,
                target=tgt_label,
                src_tgt_map=data_src_tgt_vocab_map,
                label_smoothing=self.expe.config.lm)
            batch_size, seq_len = tgt_mask.shape
            flat_tgt_mask = tgt_mask.reshape(-1)
            loss = loss * flat_tgt_mask
            gloss = loss.reshape(batch_size, seq_len).sum(1) / \
                (tgt_mask.reshape(batch_size, seq_len)).sum(1)

        elif self.expe.config.lm:
            loss_fn = model_utils.LabelSmoothingLoss(
                classes=self.vocab_size,
                smoothing=self.expe.config.lm,
                dim=-1)
            loss = loss_fn(pred_probs, tgt_label.long())
            loss = loss * tgt_mask
            gloss = loss.sum(1) / tgt_mask.sum(1)
        else:
            batch_size, seq_len, vocab_size = pred_probs.shape
            flat_tgt_mask = tgt_mask.reshape(-1)
            flat_pred_probs = pred_probs.reshape(
                batch_size * seq_len, vocab_size)
            tgt = tgt_label.reshape(-1)
            gloss = F.cross_entropy(
                flat_pred_probs, tgt.long(),
                reduction="none")
            gloss = gloss * flat_tgt_mask
            gloss = gloss.reshape(batch_size, seq_len).sum(1) / tgt_mask.sum(1)
        gloss = gloss.mean(0)

        if self.expe.config.floss:
            if self.expe.config.use_copy:
                floss_tgt_vec = self.cyclic_encode.softmax_forward(
                    model_utils.collapse_copy_scores(
                        scores=torch.cat(pred_probs, 1),
                        src_tgt_vocab_map=data_src_tgt_vocab_map,
                        vocab_size=self.vocab_size,
                        keep_src_vocab_unk=False)[0]
                    [:, :self.vocab_size].reshape(
                        batch_size, seq_len, self.vocab_size),
                    tgt_mask)
            else:
                floss_tgt_vec = self.cyclic_encode.mix_forward(
                    pred_probs, tgt_mask)

            floss_pred_probs = self.cyclic_decode(
                encoder_outputs=floss_tgt_vec,
                encoder_mask=tgt_mask,
                inp=tgt_input_data,
                inp_mask=tgt_input_data_mask,
                inp_pos=tgt_input_data_pos,
                inp_type=tgt_input_data_type,
                inp_if_hyp=tgt_input_if_hyp)

            batch_size, seq_len, vocab_size = floss_pred_probs.shape
            if self.expe.config.bwd_lm:
                loss_fn = model_utils.LabelSmoothingLoss(
                    classes=self.vocab_size,
                    smoothing=self.expe.config.bwd_lm,
                    dim=-1)
                loss = loss_fn(floss_pred_probs, tgt_output_data.long())
                loss = loss * tgt_output_mask
                floss = loss.sum(1) / tgt_output_mask.sum(1)
            else:
                floss_tgt_mask = tgt_output_mask.reshape(-1)
                floss_pred_probs = floss_pred_probs.reshape(
                    batch_size * seq_len, vocab_size)
                floss_tgt = tgt_output_data.reshape(-1)
                floss = F.cross_entropy(
                    floss_pred_probs, floss_tgt.long(),
                    reduction="none")
                floss = floss * floss_tgt_mask
                floss = floss.reshape(batch_size, seq_len).sum(1) / \
                    (floss_tgt_mask.reshape(batch_size, seq_len)).sum(1)
            floss = floss.mean(0)
        else:
            floss = torch.zeros_like(gloss)

        if self.expe.config.tloss:
            tloss_tgt_vec = self.cyclic_encode(tgt_input, tgt_mask)

            tloss_pred_probs = self.cyclic_decode(
                encoder_outputs=tloss_tgt_vec,
                encoder_mask=tgt_mask,
                inp=tgt_input_data,
                inp_mask=tgt_input_data_mask,
                inp_pos=tgt_input_data_pos,
                inp_type=tgt_input_data_type,
                inp_if_hyp=tgt_input_if_hyp)

            batch_size, seq_len, vocab_size = tloss_pred_probs.shape
            if self.expe.config.bwd_lm:
                loss_fn = model_utils.LabelSmoothingLoss(
                    classes=self.vocab_size,
                    smoothing=self.expe.config.bwd_lm, dim=-1)
                loss = loss_fn(tloss_pred_probs, tgt_output_data.long())
                loss = loss * tgt_output_mask
                tloss = loss.sum(1) / tgt_output_mask.sum(1)
            else:
                tloss_tgt_mask = tgt_output_mask.reshape(-1)
                tloss_pred_probs = tloss_pred_probs.reshape(
                    batch_size * seq_len, vocab_size)
                tloss_tgt = tgt_output_data.reshape(-1)
                tloss = F.cross_entropy(
                    tloss_pred_probs, tloss_tgt.long(),
                    reduction="none")
                tloss = tloss * tloss_tgt_mask
                tloss = tloss.reshape(batch_size, seq_len).sum(1) / \
                    (tloss_tgt_mask.reshape(batch_size, seq_len)).sum(1)
            tloss = tloss.mean(0)
        else:
            tloss = torch.zeros_like(gloss)
        loss = gloss + self.expe.config.floss * floss + \
            self.expe.config.tloss * tloss
        return loss, gloss, floss, tloss

    def greedy_decode(
            self, data, data_mask, data_pos,
            data_type, data_if_hyp, data_src_vocab,
            data_src_tgt_vocab_map,
            max_len, min_len, top_p, top_k):
        self.eval()
        data, data_mask, data_pos, data_type, \
            data_if_hyp, data_src_vocab, \
            data_src_tgt_vocab_map = \
            self.to_tensors(
                data, data_mask, data_pos,
                data_type, data_if_hyp, data_src_vocab, data_src_tgt_vocab_map)
        data_vec = self.encode(
            data, data_mask, data_pos, data_type, data_if_hyp)

        return self.decode.generate(
            encoder_outputs=data_vec,
            encoder_mask=data_mask,
            max_len=max_len,
            min_len=min_len,
            top_p=top_p,
            top_k=top_k,
            src_map=data_src_vocab,
            src_tgt_vocab_map=data_src_tgt_vocab_map)
