#!/bin/bash

python eval_rouge_meteor_rep.py --references $1 --generations $2
python eval_parent.py --tables parent_references/parent_dev.txt --references $1 --generations $2
