#!/bin/bash

python make_corpus_wiki.py -x --language "${1}" "../cirrus/${1}wiki-20250303-cirrussearch-content.json.gz"
