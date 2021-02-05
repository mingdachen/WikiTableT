#!/bin/bash

python train.py \
    --debug 0 \
    --auto_disconnect 1 \
    --save_prefix large_copy_cyc \
    --decoder_type ctransformer \
    --encoder_type transformer \
    --n_epoch 5 \
    --train_path train-trim-shuf.json \
    --dev_path dev-trim-shuf.json \
    --wikidata_path wikidata.json \
    --infobox_path infobox.json \
    --bpe_vocab cased_30k.vocab \
    --bpe_codes cased_30k.codes \
    --batch_size 4 \
    --max_num_value 10 \
    --eval_batch_size 20 \
    --use_copy 1 \
    --max_gen_len 150 \
    --min_gen_len 50 \
    --warmup_steps 2000 \
    --learning_rate 3e-4 \
    --gradient_accumulation_steps 50 \
    --encoder_num_layer 1 \
    --decoder_num_layer 12 \
    --num_head 8 \
    --bwd_num_head 8 \
    --true_cyclic_loss 1.0 \
    --fake_cyclic_loss 0.1 \
    --dropout 0.1 \
    --l2 0.0 \
    --input_hyperlink 1 \
    --input_wikidata 1 \
    --embed_dim 1024 \
    --encoder_size 1024 \
    --decoder_size 1024 \
    --print_every 100 \
    --save_every 1000 \
    --eval_every 2000
