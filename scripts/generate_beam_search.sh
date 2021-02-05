#!/bin/bash

python generate_beam_search.py \
    --gen_dir generations \
    --gen_prefix gen_dev_beam_search_size5 \
    --model_file large+copy+cyc.ckpt \
    --train_path dev-trim-shuf.json \
    --dev_path dev-trim-shuf.json \
    --wikidata_path wikidata.json.devtest \
    --infobox_path infobox.json.devtest \
    --bpe_vocab cased_30k.vocab \
    --bpe_codes cased_30k.codes \
    --eval_batch_size 10 \
    --max_gen_len 300 \
    --min_gen_len 100 \
    --beam_size 5
