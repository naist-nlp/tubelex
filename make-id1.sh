#!/bin/bash

#SBATCH -J tubelex-id1
#SBATCH -p gpu_long
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH -c4

DIR=frequencies

echo '[stanza]' >> ${DIR}/tubelex-id.out
python tubelex.py --lang id -x --frequencies -o ${DIR}/tubelex-id%.tsv.xz >> ${DIR}/tubelex-id.out

# TEST 10 files
# python tubelex.py --lang id -x --frequencies --verbose --limit 10 -o ${DIR}/zz-testtubelex-id%.tsv.xz > ${DIR}/tubelex-test-id.txt
# python tubelex.py --lang id --tokenization regex -x --frequencies --verbose --limit 10 -o ${DIR}/zz-testtubelex-id%.tsv.xz > test-id-regex.txt