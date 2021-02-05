import sys

import train_helper
import data_utils
import config

import models

BEST_DEV_BLEU = TEST_BLEU = 0


def run(e):
    global BEST_DEV_BLEU, TEST_BLEU

    e.log.info("*" * 25 + " DATA PREPARATION " + "*" * 25)
    data_processor = data_utils.DataProcessor(
        train_path=e.config.train_path,
        dev_path=e.config.dev_path,
        test_path=e.config.test_path,
        wikidata_path=e.config.wikidata_path,
        infobox_path=e.config.infobox_path,
        bpe_vocab=e.config.bpe_vocab,
        bpe_codes=e.config.bpe_codes,
        experiment=e)
    data = data_processor.process()

    train_batch = data_utils.Minibatcher(
        data=data.train_data,
        batch_size=e.config.batch_size,
        save_dir=e.experiment_dir,
        filename="minibatcher.ckpt",
        log=e.log,
        is_eval=False,
        vocab_size=len(data.vocab),
        vocab=data.vocab,
        return_wikidata=e.config.return_wikidata,
        return_hyperlink=e.config.return_hyperlink,
        input_wikidata=e.config.input_wikidata,
        input_hyperlink=e.config.input_hyperlink,
        verbose=True)

    e.log.info("*" * 25 + " DATA PREPARATION " + "*" * 25)
    e.log.info("*" * 25 + " MODEL INITIALIZATION " + "*" * 25)

    model = models.BasicCyclicAttnSplitMask(
        vocab_size=len(data.vocab),
        type_vocab_size=500,
        embed_dim=e.config.edim,
        iter_per_epoch=len(train_batch.idx_pool) // e.config.gcs,
        use_entmax=False,
        experiment=e)

    start_epoch = true_it = 0
    if e.config.resume:
        start_epoch, _, BEST_DEV_BLEU, TEST_BLEU = \
            model.load(name="latest")
        e.log.info(
            "resumed from previous checkpoint: start epoch: {}, "
            "iteration: {}, best dev bleu: {:.3f}, test bleu: {:.3f}."
            .format(start_epoch, true_it, BEST_DEV_BLEU, TEST_BLEU))

    e.log.info(model)
    e.log.info("*" * 25 + " MODEL INITIALIZATION " + "*" * 25)

    dev_eval = train_helper.Evaluator(
        model=model,
        data=data.dev_data,
        inv_vocab=data.inv_vocab,
        vocab=data.vocab,
        eval_batch_size=e.config.eval_batch_size,
        return_wikidata=e.config.return_wikidata,
        return_hyperlink=e.config.return_hyperlink,
        input_wikidata=e.config.input_wikidata,
        input_hyperlink=e.config.input_hyperlink,
        experiment=e)

    e.log.info("Training start ...")
    train_stats = train_helper.Tracker(
        ["loss", "gen_loss", "cyclic_loss", "reference_loss"])

    for epoch in range(start_epoch, e.config.n_epoch):
        for it, (input_data, input_data_mask, input_data_pos,
                 input_data_type, input_if_hyp,
                 input_data_src_vocab, input_data_src_tgt_vocab_map,
                 tgt_inp_data, tgt_inp_data_mask,
                 tgt_inp_data_pos, tgt_inp_data_type, tgt_inp_data_if_hyp,
                 tgt_out_data, tgt_out_data_mask,
                 tgt_input, tgt_label, tgt_mask, tgt_src_vocab, _) in \
                enumerate(train_batch):
            model.train()
            curr_it = train_batch.init_pointer + it + \
                1 + epoch * len(train_batch.idx_pool)
            true_it = curr_it // e.config.gcs
            full_division = ((curr_it % e.config.gcs) == 0) or \
                (curr_it % len(train_batch.idx_pool) == 0)

            loss, gloss, floss, tloss = \
                model(input_data, input_data_mask, input_data_pos,
                      input_data_type, input_if_hyp,
                      input_data_src_vocab, input_data_src_tgt_vocab_map,
                      tgt_inp_data, tgt_inp_data_mask,
                      tgt_inp_data_pos, tgt_inp_data_type, tgt_inp_data_if_hyp,
                      tgt_out_data, tgt_out_data_mask,
                      tgt_input, tgt_label, tgt_mask, tgt_src_vocab)
            model.optimize(loss / e.config.gcs, update_param=full_division)
            train_stats.update(
                {"loss": loss, "gen_loss": gloss, "cyclic_loss": floss,
                 "reference_loss": tloss}, len(input_data))

            if e.config.auto_disconnect and full_division:
                if e.elapsed_time > 3.5:
                    e.log.info("elapsed time: {:.3}(h), "
                               "automatically exiting the program..."
                               .format(e.elapsed_time))
                    train_batch.save()
                    model.save(
                        dev_bleu=BEST_DEV_BLEU,
                        test_bleu=TEST_BLEU,
                        iteration=true_it,
                        epoch=epoch,
                        name="latest")
                    sys.exit()
            if ((true_it + 1) % e.config.print_every == 0 or
                    (curr_it + 1) % len(train_batch.idx_pool) == 0) \
                    and full_division:
                curr_lr = model.scheduler.get_last_lr()[0] if e.config.wstep \
                    else e.config.lr
                summarization = train_stats.summarize(
                    "epoch: {}, it: {} (max: {}), lr: {:.4e}"
                    .format(epoch, it, len(train_batch), curr_lr))
                e.log.info(summarization)
                train_stats.reset()

            if ((true_it + 1) % e.config.eval_every == 0 or
                    curr_it % len(train_batch.idx_pool) == 0) \
                    and full_division:

                train_batch.save()
                model.save(
                    dev_bleu=BEST_DEV_BLEU,
                    test_bleu=TEST_BLEU,
                    iteration=true_it,
                    epoch=epoch,
                    name="latest")

                e.log.info("*" * 25 + " DEV SET EVALUATION " + "*" * 25)

                dev_bleu = dev_eval.evaluate("gen_dev")

                e.log.info("*" * 25 + " DEV SET EVALUATION " + "*" * 25)

                if BEST_DEV_BLEU < dev_bleu:
                    BEST_DEV_BLEU = dev_bleu

                    model.save(
                        dev_bleu=BEST_DEV_BLEU,
                        test_bleu=TEST_BLEU,
                        iteration=true_it,
                        epoch=epoch)
                e.log.info("best dev bleu: {:.4f}, test bleu: {:.4f}"
                           .format(BEST_DEV_BLEU, TEST_BLEU))
                train_stats.reset()
            if ((true_it + 1) % e.config.save_every == 0 or \
                    curr_it % len(train_batch.idx_pool) == 0) \
                    and full_division:
                train_batch.save()
                model.save(
                    dev_bleu=BEST_DEV_BLEU,
                    test_bleu=TEST_BLEU,
                    iteration=true_it,
                    epoch=epoch,
                    name="latest")

        train_batch.save()
        model.save(
            dev_bleu=BEST_DEV_BLEU,
            test_bleu=TEST_BLEU,
            iteration=true_it,
            epoch=epoch + 1,
            name="latest")

        time_per_epoch = (e.elapsed_time / (epoch - start_epoch + 1))
        time_in_need = time_per_epoch * (e.config.n_epoch - epoch - 1)
        e.log.info("elapsed time: {:.2f}(h), "
                   "time per epoch: {:.2f}(h), "
                   "time needed to finish: {:.2f}(h)"
                   .format(e.elapsed_time, time_per_epoch, time_in_need))
        train_stats.reset()


if __name__ == '__main__':

    PARSED_CONFIG = config.get_base_parser().parse_args()

    def exit_handler(*args):
        print(PARSED_CONFIG)
        print("best dev bleu: {:.4f}, test bleu: {:.4f}"
              .format(BEST_DEV_BLEU, TEST_BLEU))
        sys.exit()

    train_helper.register_exit_handler(exit_handler)

    with train_helper.Experiment(PARSED_CONFIG,
                                 PARSED_CONFIG.save_prefix) as exp:

        exp.log.info("*" * 25 + " ARGS " + "*" * 25)
        exp.log.info(PARSED_CONFIG)
        exp.log.info("*" * 25 + " ARGS " + "*" * 25)

        run(exp)
