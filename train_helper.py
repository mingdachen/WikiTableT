# pylint: disable=no-member
import subprocess
import data_utils
import threading
import argparse
import logging
import signal
import torch
import time
import nltk
import os

from config import get_base_parser, MULTI_BLEU_PERL, \
    EOS_IDX, RESOURCE_LINK, METEOR_DATA, METEOR_JAR
from decorators import auto_init_args


def register_exit_handler(exit_handler):
    import atexit

    atexit.register(exit_handler)
    signal.signal(signal.SIGTERM, exit_handler)
    signal.signal(signal.SIGINT, exit_handler)


def run_multi_bleu(input_file, reference_file):
    bleu_output = subprocess.check_output(
        "./{} {} < {}".format(MULTI_BLEU_PERL, reference_file, input_file),
        stderr=subprocess.STDOUT, shell=True).decode('utf-8')
    bleu = float(
        bleu_output.strip().split("\n")[-1]
        .split(",")[0].split("=")[1][1:])
    return bleu


class Tracker:
    @auto_init_args
    def __init__(self, names):
        assert len(names) > 0
        self.reset()

    def __getitem__(self, name):
        return self.values.get(name, 0) / self.counter if self.counter else 0

    def __len__(self):
        return len(self.names)

    def reset(self):
        self.values = dict({name: 0. for name in self.names})
        self.counter = 0
        self.create_time = time.time()

    def update(self, named_values, count):
        """
        named_values: dictionary with each item as name: value
        """
        self.counter += count
        for name, value in named_values.items():
            self.values[name] += value.item() * count

    def summarize(self, output=""):
        if output:
            output += ", "
        for name in self.names:
            output += "{}: {:.3f}, ".format(
                name, self.values[name] / self.counter if self.counter else 0)
        output += "elapsed time: {:.1f}(s)".format(
            time.time() - self.create_time)
        return output

    @property
    def stats(self):
        return {n: v / self.counter if self.counter else 0
                for n, v in self.values.items()}


class Experiment:
    @auto_init_args
    def __init__(self, config, experiments_prefix,
                 forced_debug=False, logfile_name="log"):
        """Create a new Experiment instance.

        Modified based on: https://github.com/ex4sperans/mag

        Args:
            logfile_name: str, naming for log file. This can be useful to
                separate logs for different runs on the same experiment
            experiments_prefix: str, a prefix to the path where
                experiment will be saved
        """

        # get all defaults
        all_defaults = {}
        for key in vars(config):
            all_defaults[key] = get_base_parser().get_default(key)

        self.default_config = all_defaults

        config.resume = False
        if not config.debug and not forced_debug:
            if os.path.isdir(self.experiment_dir):
                print("log exists: {}".format(self.experiment_dir))
                config.resume = True

            print(config)
            self._makedir()

        # self._make_misc_dir()

    def _makedir(self):
        os.makedirs(self.experiment_dir, exist_ok=True)

    def _make_misc_dir(self):
        os.makedirs(self.config.vocab_file, exist_ok=True)

    @property
    def log_file(self):
        return os.path.join(self.experiment_dir, self.logfile_name)

    @property
    def experiment_dir(self):
        if self.config.debug or self.forced_debug:
            return "./"
        else:
            # get namespace for each group of args
            arg_g = dict()
            for group in get_base_parser()._action_groups:
                group_d = {a.dest: self.default_config.get(a.dest, None)
                           for a in group._group_actions}
                arg_g[group.title] = argparse.Namespace(**group_d)

            # skip default value
            identifier = ""
            for key, value in sorted(vars(arg_g["model_configs"]).items()):
                if getattr(self.config, key) != value:
                    identifier += key + str(getattr(self.config, key))
            return os.path.join(self.experiments_prefix, identifier)

    def register_directory(self, dirname):
        directory = os.path.join(self.experiment_dir, dirname)
        os.makedirs(directory, exist_ok=True)
        setattr(self, dirname, directory)

    def _register_existing_directories(self):
        for item in os.listdir(self.experiment_dir):
            fullpath = os.path.join(self.experiment_dir, item)
            if os.path.isdir(fullpath):
                setattr(self, item, fullpath)

    def __enter__(self):
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        if self.config.debug or self.forced_debug:
            logging.basicConfig(
                level=logging.DEBUG,
                format='%(asctime)s %(levelname)s: %(message)s',
                datefmt='%m-%d %H:%M')
        else:
            print("log saving to", self.log_file)
            logging.basicConfig(
                filename=self.log_file,
                filemode='a+', level=logging.INFO,
                format='%(asctime)s %(levelname)s: %(message)s',
                datefmt='%m-%d %H:%M')

        self.log = logging.getLogger()
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        logging.shutdown()

    @property
    def elapsed_time(self):
        return (time.time() - self.start_time) / 3600


class Evaluator:
    def __init__(self, model, eval_batch_size, data, inv_vocab, vocab,
                 return_wikidata, return_hyperlink,
                 input_wikidata, input_hyperlink, experiment):
        self.model = model
        self.expe = experiment
        self.inv_vocab = inv_vocab

        self.data_iterator = data_utils.Minibatcher(
            batch_size=eval_batch_size,
            data=data,
            is_eval=True,
            save_dir=None,
            verbose=False,
            vocab_size=len(inv_vocab),
            vocab=vocab,
            return_wikidata=return_wikidata,
            return_hyperlink=return_hyperlink,
            input_wikidata=input_wikidata,
            input_hyperlink=input_hyperlink,
            filename="devtesteval_minibatcher.ckpt",
            log=self.expe.log)
        self.eval_stats = Tracker(
            ["loss", "gen_loss", "fake_cyclic_loss", "true_cyclic_loss"])

    def evaluate(self, gen_fn):
        self.model.eval()
        all_gen = {}
        self.expe.log.info("max gen len: {}, min gen len: {}, top p: {}"
                           .format(self.expe.config.max_gen_len,
                                   self.expe.config.min_gen_len,
                                   self.expe.config.top_p))
        for nbatch, (input_data, input_data_mask, input_data_pos,
                     input_data_type, input_if_hyp, input_data_src_vocab,
                     input_data_src_tgt_vocab_map,
                     tgt_inp_data, tgt_inp_data_mask, tgt_inp_data_pos,
                     tgt_inp_data_type, tgt_inp_data_if_hyp,
                     tgt_out_data, tgt_out_data_mask,
                     tgt_input, tgt_label, tgt_mask,
                     tgt_src_vocab, batch_idx) in \
                enumerate(self.data_iterator):
            if nbatch and nbatch % (len(self.data_iterator) // 10 + 1) == 0:
                self.expe.log.info(
                    "evaluating progress: {}/{} = {:.2f} %"
                    .format(nbatch, len(self.data_iterator),
                            nbatch / (len(self.data_iterator) + 1) * 100)
                )
            with torch.no_grad():
                batch_gen, _ = self.model.greedy_decode(
                    input_data, input_data_mask, input_data_pos,
                    input_data_type, input_if_hyp,
                    input_data_src_vocab, input_data_src_tgt_vocab_map,
                    self.expe.config.max_gen_len, self.expe.config.min_gen_len,
                    self.expe.config.top_p, self.expe.config.top_k)

            assert len(batch_gen) == len(batch_idx), \
                "len(batch_gen)={} != len(batch_idx)={}"\
                .format(len(batch_gen), len(batch_idx))
            for gen, idx in zip(batch_gen, batch_idx):
                curr_gen = []
                for i in gen:
                    if i == EOS_IDX:
                        break
                    if int(i) >= len(self.inv_vocab):
                        curr_gen.append(
                            self.data_iterator.data[idx]["inv_src_vocab"]
                            [int(i) - len(self.inv_vocab)])
                    else:
                        curr_gen.append(self.inv_vocab[int(i)])
                all_gen[idx] = " ".join(curr_gen)\
                    .replace("@@ ", "").replace("@@", "")
        assert len(all_gen) == len(self.data_iterator.data), \
            "{} != {}".format(len(all_gen), len(self.data_iterator.data))
        file_name = os.path.join(self.expe.experiment_dir, gen_fn + ".txt")
        ref_file_name = \
            os.path.join(self.expe.experiment_dir, gen_fn + "_ref.txt")

        file_name_untok = \
            os.path.join(self.expe.experiment_dir, gen_fn + "_untok.txt")
        ref_file_name_untok = \
            os.path.join(self.expe.experiment_dir, gen_fn + "_untok_ref.txt")

        gen_len_list = []
        with open(file_name, "w") as fp, open(ref_file_name, "w") as fp2, \
                open(file_name_untok, "w") as fpu, \
                open(ref_file_name_untok, "w") as fpu2:
            for hyp_idx, ref in zip(sorted(all_gen),
                                    sorted(self.data_iterator.data,
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
        self.expe.log.info(
            "generated sentences saved to: {}".format(file_name))

        self.expe.log.info(
            "#Data: {}, bleu: {:.3f}, loss: {:.3f}, "
            "gloss: {:.3f}, floss: {:.3f}, tloss: {:.3f}, "
            "avg gen len: {:.1f}"
            .format(len(all_gen), bleu_score, self.eval_stats["loss"],
                    self.eval_stats["gen_loss"],
                    self.eval_stats["fake_cyclic_loss"],
                    self.eval_stats["true_cyclic_loss"],
                    sum(gen_len_list) / len(gen_len_list))
        )
        self.eval_stats.reset()
        return bleu_score


def enc(s):
    return s.encode('utf-8')


def dec(s):
    return s.decode('utf-8')


class Meteor:
    def __init__(self):
        self.meteor_cmd = ['java', '-jar', '-Xmx2G', METEOR_JAR,
                           '-', '-', '-stdio', '-l', 'en', '-norm', '-a',
                           METEOR_DATA]
        for file in [METEOR_JAR, METEOR_DATA]:
            assert os.path.isfile(file), \
                "{} not exsit! Please download it from {}"\
                .format(file, RESOURCE_LINK)
        self.meteor_p = subprocess.Popen(
            self.meteor_cmd,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        # Used to guarantee thread safety
        self.lock = threading.Lock()

    def compute_score(self, gts, res):
        assert(gts.keys() == res.keys())
        imgIds = gts.keys()
        scores = []

        eval_line = 'EVAL'
        self.lock.acquire()
        for i in imgIds:
            assert(len(res[i]) == 1)
            stat = self._stat(res[i][0], gts[i])
            eval_line += ' ||| {}'.format(stat)

        self.meteor_p.stdin.write(enc('{}\n'.format(eval_line)))
        self.meteor_p.stdin.flush()
        for i in range(0, len(imgIds)):
            scores.append(dec(float(self.meteor_p.stdout.readline().strip())))
        score = float(dec(self.meteor_p.stdout.readline().strip()))
        self.lock.release()

        return score, scores

    def _stat(self, hypothesis_str, reference_list):
        # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
        hypothesis_str = hypothesis_str.replace('|||','').replace('  ',' ')
        score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
        self.meteor_p.stdin.write(enc(score_line + "\n"))
        self.meteor_p.stdin.flush()
        return dec(self.meteor_p.stdout.readline()).strip()

    def _score(self, hypothesis_str, reference_list):
        # self.lock.acquire()
        with self.lock:
            # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
            hypothesis_str = hypothesis_str.replace('|||','').replace('  ',' ')
            score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
            self.meteor_p.stdin.write(enc(score_line + "\n"))
            self.meteor_p.stdin.flush()
            stats = dec(self.meteor_p.stdout.readline().strip())
            eval_line = 'EVAL ||| {}'.format(stats)
            # EVAL ||| stats
            self.meteor_p.stdin.write(enc('{}\n'.format(eval_line)))
            self.meteor_p.stdin.flush()
            score = float(dec(self.meteor_p.stdout.readline()).strip())
            # bug fix: there are two values returned by the jar file, one average, and one all, so do it twice
            # thanks for Andrej for pointing this out
            score = float(dec(self.meteor_p.stdout.readline().strip()))
        # self.lock.release()
        return score

    def __del__(self):
        with self.lock:
            if self.meteor_p:
                self.meteor_p.stdin.close()
                self.meteor_p.kill()
                self.meteor_p.wait()
                self.meteor_p = None
