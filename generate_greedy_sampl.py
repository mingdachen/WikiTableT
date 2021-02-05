import train_helper
import data_utils
import config
import models
import torch
import sys
import os

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
    # rev_model_exp = copy.deepcopy(experiment)
    model_exp.config = checkpoint["config"]
    model_exp.experiment_dir = \
        e.config.gen_dir if e.config.gen_dir else e.experiment_dir
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

    dev_eval = train_helper.Evaluator(
        model=model,
        data=data.dev_data,
        inv_vocab=data.inv_vocab,
        vocab=data.vocab,
        eval_batch_size=e.config.eval_batch_size,
        return_wikidata=model_exp.config.return_wikidata,
        return_hyperlink=model_exp.config.return_hyperlink,
        input_wikidata=e.config.input_wikidata,
        input_hyperlink=e.config.input_hyperlink,
        experiment=model_exp)

    if not os.path.isdir(model_exp.experiment_dir):
        print("make dirs", model_exp.experiment_dir)
        os.makedirs(model_exp.experiment_dir)
    dev_eval.evaluate(e.config.gen_prefix)


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
