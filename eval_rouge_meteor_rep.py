from train_helper import run_multi_bleu, Meteor
from collections import Counter
from tqdm import tqdm

import statistics
import rouge
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--references', type=str, default=None)
parser.add_argument('--generations', type=str, default=None)

args = parser.parse_args()

gen_path = args.generations
ref_path = args.references

rouge_eval = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
                         max_n=2,
                         limit_length=True,
                         length_limit=5000,
                         length_limit_type='words',
                         apply_avg=False,
                         apply_best=False,
                         alpha=0.5,  # Default F1_score
                         weight_factor=1.2,
                         stemming=True)

max_rep_n = 4
min_rep_n = 2
intra_rep = {"mean": 0.0, "median": 0.0, "min": 0.0,
             "max": 0.0, "std": 0.0, "count_ratio": 0.0}
intra_rep_per_gram = {}
for i in range(1, max_rep_n + 1):
    intra_rep_per_gram[i] = {"mean": 0.0, "median": 0.0,
                             "min": 0.0, "max": 0.0,
                             "std": 0.0, "count_ratio": 0.0}

n_dev = 0
stats = {"rouge1": 0, "rouge2": 0, "rougel": 0, "meteor": 0}
meteor = Meteor()
with open(gen_path) as gen_fp, open(ref_path) as ref_fp:
    for nline, (gen_line, ref_line) in tqdm(enumerate(zip(gen_fp, ref_fp))):
        if gen_line.strip():
            curr_intra_rep = Counter()
            curr_intra_rep_per_gram = \
                {i: Counter() for i in range(1, max_rep_n + 1)}

            n_dev += 1
            gen_tok = gen_line.strip().split()
            for i in range(len(gen_tok)):
                for l in range(1, min(max_rep_n + 1, len(gen_tok[i:]) + 1)):
                    if l >= min_rep_n:
                        curr_intra_rep[" ".join(gen_tok[i: i + l])] += 1
                    curr_intra_rep_per_gram[len(gen_tok[i: i + l])][" ".join(gen_tok[i: i + l])] += 1

            curr_intra_rep_count = []
            curr_total_ngram = len(curr_intra_rep)
            curr_total_rep_ngram_count = 0
            curr_total_rep_ngram_type = 0
            for p, c in curr_intra_rep.items():
                if c >= 3:
                    curr_intra_rep_count.append(c)
                    curr_total_rep_ngram_count += c
                    curr_total_rep_ngram_type += 1

            curr_intra_rep_total_count_per_gram = {}
            curr_intra_rep_total_type_per_gram = {}
            curr_intra_rep_count_per_gram = {}
            curr_total_ngram_per_gram = {}
            curr_total_rep_ngram_count_per_gram = {}
            curr_total_rep_ngram_type_per_gram = {}

            for i in range(1, max_rep_n + 1):
                curr_intra_rep_total_count_per_gram[i] = sum(curr_intra_rep_per_gram[i].values())
                curr_intra_rep_total_type_per_gram[i] = len(curr_intra_rep_per_gram[i])

                curr_intra_rep_count_per_gram[i] = []
                curr_total_ngram_per_gram[i] = len(curr_intra_rep_per_gram[i])
                curr_total_rep_ngram_count_per_gram[i] = 0
                curr_total_rep_ngram_type_per_gram[i] = 0
                for p, c in curr_intra_rep_per_gram[i].items():
                    if c >= 3:
                        curr_intra_rep_count_per_gram[i].append(c)
                        curr_total_rep_ngram_count_per_gram[i] += c
                        curr_total_rep_ngram_type_per_gram[i] += 1

                intra_rep_per_gram[i]["max"] += max(curr_intra_rep_count_per_gram[i]) if len(curr_intra_rep_count_per_gram[i]) else 0
                intra_rep_per_gram[i]["min"] += min(curr_intra_rep_count_per_gram[i]) if len(curr_intra_rep_count_per_gram[i]) else 0
                intra_rep_per_gram[i]["median"] += statistics.median(curr_intra_rep_count_per_gram[i]) if len(curr_intra_rep_count_per_gram[i]) else 0
                intra_rep_per_gram[i]["mean"] += statistics.mean(curr_intra_rep_count_per_gram[i]) if len(curr_intra_rep_count_per_gram[i]) else 0
                intra_rep_per_gram[i]["std"] += statistics.stdev(curr_intra_rep_count_per_gram[i]) if len(curr_intra_rep_count_per_gram[i]) > 1 else 0
                intra_rep_per_gram[i]["count_ratio"] += curr_total_rep_ngram_count_per_gram[i] / curr_intra_rep_total_count_per_gram[i] if curr_intra_rep_total_count_per_gram[i] else 0

            curr_intra_rep_total_count = sum(curr_intra_rep.values())
            curr_intra_rep_total_type = len(curr_intra_rep)

            intra_rep["max"] += max(curr_intra_rep_count) if len(curr_intra_rep_count) else 0
            intra_rep["min"] += min(curr_intra_rep_count) if len(curr_intra_rep_count) else 0
            intra_rep["median"] += statistics.median(curr_intra_rep_count) if len(curr_intra_rep_count) else 0
            intra_rep["mean"] += statistics.mean(curr_intra_rep_count) if len(curr_intra_rep_count) else 0
            intra_rep["std"] += statistics.stdev(curr_intra_rep_count) if len(curr_intra_rep_count) > 1 else 0
            intra_rep["count_ratio"] += curr_total_rep_ngram_count / curr_intra_rep_total_count if curr_intra_rep_total_count else 0

            ms = meteor._score(gen_line.strip(), [ref_line.strip()])
            rouge_scores = rouge_eval.get_scores([gen_line.strip()], [ref_line.strip()])
            stats['rouge1'] += rouge_scores['rouge-1'][0]['f'][0]
            stats['rouge2'] += rouge_scores['rouge-2'][0]['f'][0]
            stats['rougel'] += rouge_scores['rouge-l'][0]['f'][0]
            stats['meteor'] += ms

dev_rouge1 = stats['rouge1'] / n_dev * 100
dev_rouge2 = stats['rouge2'] / n_dev * 100
dev_rougel = stats['rougel'] / n_dev * 100
dev_meteor = stats['meteor'] / n_dev * 100

intra_rep["max"] = intra_rep["max"] / n_dev
intra_rep["min"] = intra_rep["min"] / n_dev
intra_rep["median"] = intra_rep["median"] / n_dev
intra_rep["mean"] = intra_rep["mean"] / n_dev
intra_rep["std"] = intra_rep["std"] / n_dev
intra_rep["count_ratio"] = intra_rep["count_ratio"] / n_dev * 100


dev_bleu_score = run_multi_bleu(gen_path, ref_path)

print("REP -", ", ".join(["{} : {:.2f}".format(k, v) for k, v
                          in sorted(intra_rep.items())]))

print("NOTE: for REP, We report count ratio.")

print("=== REP ngram breakdown ===")
for i in range(1, max_rep_n + 1):
    print("*** {}-gram ***".format(i))
    print(", ".join(["{} : {:.2f}".format(k, v / n_dev if k != "count_ratio" else v / n_dev * 100)
                     for k, v in sorted(intra_rep_per_gram[i].items())]))

print("BLEU: {:.2f}".format(dev_bleu_score))
print("REOUG-1: {:.2f}".format(dev_rouge1))
print("ROUGE-2: {:.2f}".format(dev_rouge2))
print("ROUGE-L: {:.2f}".format(dev_rougel))
print("METEOR: {:.2f}".format(dev_meteor))
