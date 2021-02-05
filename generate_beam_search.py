import train_helper
import data_utils
import config
import models
import torch
import sys
import nltk
import os

from train_helper import run_multi_bleu
from config import EOS_IDX
from tqdm import tqdm

BEST_DEV_BLEU = TEST_BLEU = 0


def run(e):
    global BEST_DEV_BLEU, TEST_BLEU

    checkpoint = torch.load(e.config.model_file,
                            map_location=lambda storage, loc: storage)
    e.log.info("loaded from: {}".format(e.config.model_file))

    class dummy_exp:
        pass
    model_exp = dummy_exp()
    model_exp.log = e.log
    checkpoint["config"].debug = False
    checkpoint["config"].resume = True

    model_exp.config = checkpoint["config"]
    model_exp.experiment_dir = e.config.gen_dir \
        if e.config.gen_dir else e.experiment_dir
    model_exp.config.top_p = e.config.top_p
    model_exp.config.max_gen_len = e.config.max_gen_len
    model_exp.config.min_gen_len = e.config.min_gen_len
    for name in dir(e.config):
        if name.startswith("__"):
            continue
        if name not in dir(model_exp.config):
            value = getattr(e.config, name)
            e.log.info("update {} to {}".format(name, value))
            setattr(model_exp.config, name, value)

    e.log.info("*" * 25 + " DATA PREPARATION " + "*" * 25)
    data_processor = data_utils.DataProcessor(
        train_path=e.config.train_path,
        dev_path=e.config.dev_path,
        test_path=e.config.test_path,
        wikidata_path=e.config.wikidata_path,
        infobox_path=e.config.infobox_path,
        bpe_vocab=e.config.bpe_vocab,
        bpe_codes=e.config.bpe_codes,
        experiment=model_exp)
    data = data_processor.process()

    e.log.info("*" * 25 + " DATA PREPARATION " + "*" * 25)
    e.log.info("*" * 25 + " MODEL INITIALIZATION " + "*" * 25)

    model = models.BasicCyclicAttnSplitMask(
        vocab_size=len(data.vocab),
        type_vocab_size=500,
        embed_dim=model_exp.config.edim,
        iter_per_epoch=100,
        use_entmax=False,
        experiment=model_exp)

    model.load(checkpointed_state_dict=checkpoint["state_dict"])

    e.log.info(model)
    e.log.info("*" * 25 + " MODEL INITIALIZATION " + "*" * 25)

    dev_eval = train_helper.SplitMaskEvaluator(
        model=model,
        data=data.dev_data,
        inv_vocab=data.inv_vocab,
        vocab=data.vocab,
        return_wikidata=e.config.return_wikidata,
        return_hyperlink=e.config.return_hyperlink,
        input_wikidata=e.config.input_wikidata,
        input_hyperlink=e.config.input_hyperlink,
        eval_batch_size=e.config.eval_batch_size,
        experiment=model_exp)

    model.eval()
    gen_fn = e.config.gen_prefix
    output_path = e.config.gen_dir \
        if e.config.gen_dir else e.experiment_dir

    if not os.path.isdir(output_path):
        print("make dirs", output_path)
        os.makedirs(output_path)
    print("e.config.max_gen_len", e.config.max_gen_len,
          "e.config.min_gen_len", e.config.min_gen_len)
    all_gen = {}
    for nbatch, (input_data, input_data_mask, input_data_pos,
                 input_data_type, input_if_hyp,
                 input_data_src_vocab, input_data_src_tgt_vocab_map,
                 tgt_inp_data, tgt_inp_data_mask, tgt_inp_data_pos,
                 tgt_inp_data_type, tgt_inp_data_if_hyp,
                 tgt_out_data, tgt_out_data_mask,
                 tgt_input, tgt_label, tgt_mask,
                 tgt_src_vocab, batch_idx) in tqdm(
            enumerate(dev_eval.data_iterator),
            total=len(dev_eval.data_iterator)):
        if nbatch and nbatch % (len(dev_eval.data_iterator) // 10 + 1) == 0:
            e.log.info("evaluating progress: {}/{} = {:.2f} %"
                       .format(nbatch,
                               len(dev_eval.data_iterator),
                               nbatch / (len(dev_eval.data_iterator) + 1) * 100)
                       )
        with torch.no_grad():
            data, data_mask, data_pos, \
                data_type, data_if_hyp, data_src_vocab,\
                data_src_tgt_vocab_map = \
                model.to_tensors(input_data, input_data_mask,
                                 input_data_pos, input_data_type,
                                 input_if_hyp,
                                 input_data_src_vocab,
                                 input_data_src_tgt_vocab_map)
            data_vec = model.encode(
                data, data_mask, data_pos, data_type, data_if_hyp)

            batch_gen, batch_nll, batch_len = model.decode.generate_beam(
                encoder_output=data_vec,
                encoder_mask=data_mask,
                beam_size=e.config.beam_size,
                trigram_blocking=e.config.trigram_blocking,
                min_len=e.config.min_gen_len,
                max_len=e.config.max_gen_len,
                src_map=data_src_vocab,
                src_tgt_vocab_map=data_src_tgt_vocab_map)

        for batch_gen_, batch_null_, batch_len_, idx in \
                zip(batch_gen, batch_nll, batch_len, batch_idx):
            curr_gen = []
            for i in batch_gen_[1:batch_len_]:
                if i == EOS_IDX:
                    break
                if i >= len(dev_eval.inv_vocab):
                    curr_gen.append(
                        dev_eval.data_iterator.data[idx]["inv_src_vocab"][int(i) - len(dev_eval.inv_vocab)]
                    )
                else:
                    curr_gen.append(dev_eval.inv_vocab[int(i)])
            all_gen[idx] = " ".join(curr_gen)\
                .replace("@@ ", "").replace("@@", "")

    assert len(all_gen) == len(dev_eval.data_iterator.data), \
        "{} != {}".format(len(all_gen), len(dev_eval.data_iterator.data))
    file_name = os.path.join(output_path, gen_fn + ".txt")
    ref_file_name = os.path.join(output_path, gen_fn + "_ref.txt")

    file_name_untok = os.path.join(output_path, gen_fn + "_untok.txt")
    ref_file_name_untok = os.path.join(output_path, gen_fn + "_untok_ref.txt")

    gen_len_list = []
    with open(file_name, "w") as fp, open(ref_file_name, "w") as fp2, \
            open(file_name_untok, "w") as fpu, \
            open(ref_file_name_untok, "w") as fpu2:
        for hyp_idx, ref in zip(sorted(all_gen),
                                sorted(dev_eval.data_iterator.data,
                                       key=lambda x: x["idx"])):
            assert hyp_idx == ref["idx"], \
                "hyp_idx={} != ref[\"idx\"]={}".format(hyp_idx, ref["idx"])
            hyp = all_gen[hyp_idx]
            fp2.write(ref["tok_text"] + "\n")
            fpu2.write(ref["untok_text"] + "\n")
            if hyp:
                tok_hyp = nltk.word_tokenize(hyp)
                gen_len_list.append(len(tok_hyp))
                tok_hyp = " ".join(tok_hyp)
                fp.write(tok_hyp + "\n")
                fpu.write(hyp + "\n")
            else:
                gen_len_list.append(0)
                fp.write("<placeholder>\n")
                fpu.write("<placeholder>\n")
    bleu_score = run_multi_bleu(file_name, ref_file_name)
    e.log.info("generated sentences saved to: {}".format(file_name))

    e.log.info(
        "#Data: {}, bleu: {:.3f}, loss: {:.3f}, gloss: {:.3f}, "
        "floss: {:.3f}, tloss: {:.3f}, "
        "avg gen len: {:.2f}"
        .format(len(all_gen), bleu_score, dev_eval.eval_stats["loss"],
                dev_eval.eval_stats["gen_loss"],
                dev_eval.eval_stats["fake_cyclic_loss"],
                dev_eval.eval_stats["true_cyclic_loss"],
                sum(gen_len_list) / len(gen_len_list)
                )
    )
    dev_eval.eval_stats.reset()


if __name__ == '__main__':

    PARSED_CONFIG = config.get_base_parser().parse_args()

    def exit_handler(*args):
        print(PARSED_CONFIG)
        print("best dev bleu: {:.4f}, test bleu: {:.4f}"
              .format(BEST_DEV_BLEU, TEST_BLEU))
        sys.exit()

    train_helper.register_exit_handler(exit_handler)

    with train_helper.Experiment(PARSED_CONFIG,
                                 PARSED_CONFIG.save_prefix,
                                 forced_debug=True) as exp:

        exp.log.info("*" * 25 + " ARGS " + "*" * 25)
        exp.log.info(PARSED_CONFIG)
        exp.log.info("*" * 25 + " ARGS " + "*" * 25)

        run(exp)
