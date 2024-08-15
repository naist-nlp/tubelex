#!/bin/bash

#SBATCH -J tubelex-id1
#SBATCH -p gpu_long
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH -c4

echo '[stanza]' >> tubelex-id.out
python tubelex.py --lang id -x --frequencies -o tubelex-id%.tsv.xz >> tubelex-id.out

# TEST 10 files
# python tubelex.py --lang id -x --frequencies --verbose --limit 10 -o zz-testtubelex-id%.tsv.xz > tubelex-test-id.txt
# python tubelex.py --lang id --tokenization regex -x --frequencies --verbose --limit 10 -o zz-testtubelex-id%.tsv.xz > test-id-regex.txt