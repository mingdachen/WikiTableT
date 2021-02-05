#!/bin/bash

python convert_to_parent_reference.py \
    dev-trim-shuf.json \
    infobox.json.devtest \
    wikidata.json.devtest \
    parent_dev.txt
