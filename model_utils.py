import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from entmax import entmax_bisect


def init_embedding(input_embedding):
    """
    Initialize embedding
    """
    bias = np.sqrt(3.0 / input_embedding.size(1))
    nn.init.uniform_(input_embedding, -bias, bias)


def init_linear(input_linear):
    """
    Initialize linear transformation
    """
    bias = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
    nn.init.uniform_(input_linear.weight, -bias, bias)
    if input_linear.bias is not None:
        input_linear.bias.data.zero_()


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = F.one_hot(target.long(), self.cls)
            true_dist = true_dist * self.confidence
            true_dist = true_dist + self.smoothing / (self.cls - 1)
        return torch.sum(-true_dist * pred, dim=self.dim)


def get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, need_softmax=True, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    # assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = \
            logits < torch.topk(logits, top_k, dim=-1)[0]\
            .min(-1)[0].unsqueeze(-1)
        logits[indices_to_remove] = filter_value

    if top_p > 0.0 and top_p < 1.0:
        sorted_logits, sorted_indices = \
            torch.sort(logits, dim=-1, descending=True)
        if need_softmax:
            softmax_sorted_logits = F.softmax(sorted_logits, dim=-1)
        else:
            softmax_sorted_logits = sorted_logits
        cumulative_probs = torch.cumsum(softmax_sorted_logits, dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 0] = False
        # Shift the indices to the right to keep also the first token above the threshold
        # sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        # sorted_indices_to_remove[..., 0] = 0

        # indices_to_remove = sorted_indices[sorted_indices_to_remove]
        # unsorted
        sorted_logits[sorted_indices_to_remove] = filter_value
        logits = sorted_logits.gather(-1, torch.sort(sorted_indices, descending=False)[1])
    return logits


class CopyGenerator(nn.Module):
    """An implementation of pointer-generator networks
    :cite:`DBLP:journals/corr/SeeLM17`.
    These networks consider copying words
    directly from the source sequence.
    The copy generator is an extended version of the standard
    generator that computes three values.
    * :math:`p_{softmax}` the standard softmax over `tgt_dict`
    * :math:`p(z)` the probability of copying a word from
      the source
    * :math:`p_{copy}` the probility of copying a particular word.
      taken from the attention distribution directly.
    The model returns a distribution over the extend dictionary,
    computed as
    :math:`p(w) = p(z=1)  p_{copy}(w)  +  p(z=0)  p_{softmax}(w)`
    .. mermaid::
       graph BT
          A[input]
          S[src_map]
          B[softmax]
          BB[switch]
          C[attn]
          D[copy]
          O[output]
          A --> B
          A --> BB
          S --> D
          C --> D
          D --> O
          B --> O
          BB --> O
    Args:
       input_size (int): size of input representation
       output_size (int): size of output vocabulary
       pad_idx (int)
    based on an implementation from
    https://github.com/OpenNMT/OpenNMT-py
    """

    def __init__(self, input_size, use_entmax, pad_idx=0):
        super(CopyGenerator, self).__init__()
        # self.linear = nn.Linear(input_size, output_size)
        self.linear_copy = nn.Linear(input_size, 1)
        self.pad_idx = pad_idx
        self.use_entmax = use_entmax

    def _bottle(self, _v):
        return _v.view(-1, _v.size(2))

    def _unbottle(self, _v, batch_size):
        return _v.view(-1, batch_size, _v.size(1))

    def forward(self, hidden, orig_prob, attn, src_map):
        """
        Compute a distribution over the target dictionary
        extended by the dynamic dictionary implied by copying
        source words.
        Args:
           hidden (FloatTensor): hidden outputs ``(batch x tlen, input_size)``
           attn (FloatTensor): attn for each ``(batch x tlen, input_size)``
           src_map (FloatTensor):
               A sparse indicator matrix mapping each source word to
               its index in the "extended" vocab containing.
               ``(src_len, batch, extra_words)``
        """

        # CHECKS
        # batch_by_tlen, _ = hidden.size()
        # batch_by_tlen_, slen = attn.size()
        onehot_src_map = \
            F.one_hot(src_map.long(), torch.max(src_map).long() + 1)
        batch, slen, cvocab = onehot_src_map.size()

        if self.use_entmax:
            prob = entmax_bisect(orig_prob, 1.2)
        else:
            prob = torch.softmax(orig_prob, 1)

        # Probability of copying p(z=1) batch.
        p_copy = torch.sigmoid(self.linear_copy(hidden))
        # Probability of not copying: p_{word}(w) * (1 - p(z))
        out_prob = torch.mul(prob, 1 - p_copy)
        mul_attn = torch.mul(attn, p_copy)
        copy_prob = torch.bmm(
            mul_attn.view(batch, -1, slen),  # batch size x tgt len x src len
            onehot_src_map.float())  # batch size x src len x cvocab
        copy_prob = copy_prob.contiguous().view(-1, cvocab)
        return out_prob, copy_prob


class CopyGeneratorLoss(nn.Module):
    """Copy generator criterion.

    based on an implementation from
    https://github.com/OpenNMT/OpenNMT-py
    """
    def __init__(self, vocab_size, force_copy, unk_index=0,
                 ignore_index=0, eps=1e-10):
        super(CopyGeneratorLoss, self).__init__()
        self.force_copy = force_copy
        self.eps = eps
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.unk_index = unk_index

    def _bottle(self, _v):
        return _v.view(-1, _v.size(2))

    def _unbottle(self, _v, batch_size):
        return _v.view(-1, batch_size, _v.size(1))

    def forward(self, scores, align, target, src_tgt_map, label_smoothing):
        """
        Args:
            scores (FloatTensor): ``(batch_size*tgt_len)`` x dynamic vocab size
                whose sum along dim 1 is less than or equal to 1, i.e. cols
                softmaxed.
            src_tgt_map: batch size x extended vocab size
                         ([b, src vocab idx] = tgt vocab idx)
            align (LongTensor): ``(batch_size x tgt_len)``
            target (LongTensor): ``(batch_size x tgt_len)``
        """
        bs, sqlen = align.shape
        flat_align = align.reshape(-1)
        flat_target = target.reshape(-1)

        if label_smoothing:
            out_prob, copy_prob = scores

            scores, copy_mask = collapse_copy_scores(
                torch.cat([out_prob, copy_prob], 1),
                src_tgt_map, self.vocab_size)

            label_mask = copy_mask

            confidence = 1 - label_smoothing
            smoothing = label_smoothing / label_mask.sum(1, keepdim=True)

            tgt_labels = torch.zeros_like(scores)
            copy_labels = torch.zeros_like(scores)

            tgt_labels.scatter_(1, flat_target.unsqueeze(1).long(), 1)

            copy_ix = flat_align.unsqueeze(1) + self.vocab_size
            copy_labels.scatter_(1, copy_ix.long(), 1)
            non_copy = flat_align == self.unk_index
            if not self.force_copy:
                non_copy = non_copy | (flat_target != self.unk_index)

            final_labels = torch.where(
                non_copy.unsqueeze(1), tgt_labels, copy_labels
            )

            final_labels = final_labels * (confidence - smoothing) + smoothing
            final_labels = final_labels * label_mask

            # final_labels = final_labels * label_mask
            loss = torch.sum(- (scores + self.eps).log() * final_labels, dim=1)
        else:
            scores = torch.cat(scores, 1)
            # probabilities assigned by the model to the gold targets
            vocab_probs = scores.gather(
                1, flat_target.unsqueeze(1).long()).squeeze(1)

            # probability of tokens copied from source
            copy_ix = flat_align.unsqueeze(1) + self.vocab_size
            copy_tok_probs = scores.gather(1, copy_ix.long()).squeeze(1)
            # Set scores for unk to 0 and add eps
            copy_tok_probs[flat_align == self.unk_index] = 0
            copy_tok_probs = copy_tok_probs + self.eps  # to avoid -inf logs

            # find the indices in which you do not use the copy mechanism
            non_copy = flat_align == self.unk_index
            if not self.force_copy:
                non_copy = non_copy | (flat_target != self.unk_index)

            probs = torch.where(
                non_copy, copy_tok_probs + vocab_probs, copy_tok_probs
            )
            # just NLLLoss; can the module be incorporated?
            loss = -(probs + self.eps).log()
        # Drop padding.
        loss[flat_target == self.ignore_index] = 0
        return loss


class CopyGeneratorLossCompute(nn.Module):
    """Copy Generator Loss Computation.

    based on an implementation from
    https://github.com/OpenNMT/OpenNMT-py
    """
    def __init__(self, criterion, generator, tgt_vocab, normalize_by_length):
        super(CopyGeneratorLossCompute, self).__init__()
        self.criterion = criterion
        self.generator = generator
        self.tgt_vocab = tgt_vocab
        self.normalize_by_length = normalize_by_length

    def _bottle(self, _v):
        return _v.view(-1, _v.size(2))

    def _unbottle(self, _v, batch_size):
        return _v.view(-1, batch_size, _v.size(1))

    def compute_loss(self, batch, output, target, copy_attn, align,
                     std_attn=None, coverage_attn=None):
        """Compute the loss.
        The args must match :func:`self._make_shard_state()`.
        Args:
            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            copy_attn: the copy attention value.
            align: the align info.
        """
        target = target.view(-1)
        align = align.view(-1)
        scores = self.generator(
            self._bottle(output), self._bottle(copy_attn), batch.src_map
        )
        loss = self.criterion(scores, align, target)

        # this part looks like it belongs in CopyGeneratorLoss
        if self.normalize_by_length:
            # Compute Loss as NLL divided by seq length
            tgt_lens = batch.tgt[:, :, 0].ne(self.padding_idx).sum(0).float()
            # Compute Total Loss per sequence in batch
            loss = loss.view(-1, batch.batch_size).sum(0)
            # Divide by length of each sequence and sum
            loss = torch.div(loss, tgt_lens).sum()
        else:
            loss = loss.sum()

        return loss


def collapse_copy_scores(
        scores, src_tgt_vocab_map, vocab_size,
        keep_src_vocab_unk=True):
    """
    Given scores from an expanded dictionary
    corresponeding to a batch, sums together copies,
    with a dictionary word when it is ambiguous.

    src_tgt_vocab_map: batch size x src tgt vocab map size
    scores: (batch size * seq len) x dynamic vocab size

    based on an implementation from
    https://github.com/OpenNMT/OpenNMT-py
    """
    batch_size = src_tgt_vocab_map.shape[0]
    batch_size_by_seq_len = scores.shape[0]
    assert batch_size_by_seq_len % batch_size == 0, \
        batch_size_by_seq_len % batch_size

    seq_len = batch_size_by_seq_len // batch_size
    offset = vocab_size

    fill = src_tgt_vocab_map[:, 1:].unsqueeze(1)\
        .expand(-1, seq_len, -1).reshape(batch_size * seq_len, -1)
    pad = torch.ones(batch_size_by_seq_len,
                     scores.shape[1] - fill.shape[1]).to(fill.device)
    padded_fill = torch.cat([pad, fill], 1)
    scores[padded_fill == -1] = 0

    non_neg_src_tgt_vocab_map = src_tgt_vocab_map.clone()
    non_neg_src_tgt_vocab_map[non_neg_src_tgt_vocab_map == -1] = 0

    blank = (offset + torch.arange(1, non_neg_src_tgt_vocab_map.shape[1])
             .unsqueeze(0).expand(batch_size_by_seq_len, -1)).long()
    blank = blank.to(scores.device)
    fill = non_neg_src_tgt_vocab_map[:, 1:].long().unsqueeze(1)\
        .expand(-1, seq_len, -1).reshape(batch_size * seq_len, -1)

    add_scores = torch.zeros_like(scores)
    indexed_scores = scores.gather(1, blank)
    add_scores.scatter_(1, fill, indexed_scores)
    if keep_src_vocab_unk:
        add_scores[:, 0] = 0
    scores = scores + add_scores

    scores_mask = torch.ones_like(scores)
    scores_mask.scatter_(1, blank, 0.0)

    if keep_src_vocab_unk:
        scores_mask[padded_fill == 0] = 1
    scores = scores * scores_mask

    return scores, scores_mask
