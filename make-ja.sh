#!/bin/bash

#SBATCH -J tubelex-ja
#SBATCH -A lang
#SBATCH -p lang_short
#SBATCH -c1

python tubelex.py --lang ja -x --clean --unique										 	>  tubelex-ja.out
echo '[unidic-lite]' 																	>> tubelex-ja.out
python tubelex.py --lang ja -x --frequencies -o tubelex-ja%.tsv.xz 						>> tubelex-ja.out
echo '[unidic-310]' 																	>> tubelex-ja.out
python tubelex.py --lang ja -x --frequencies -D unidic -o tubelex-ja-310%.tsv.xz 		>> tubelex-ja.out

echo '[unidic-lite] --form base --pos' 															>> tubelex-ja.out
python tubelex.py --lang ja -x --frequencies --form base --pos -o tubelex-ja-base-pos%.tsv.xz 	>> tubelex-ja.out
echo '[unidic-lite] --form lemma --pos' 														>> tubelex-ja.out
python tubelex.py --lang ja -x --frequencies --form lemma --pos -o tubelex-ja-lemma-pos%.tsv.xz >> tubelex-ja.out

echo '[unidic-310] --form base --pos' 																			>> tubelex-ja.out
python tubelex.py --lang ja -x --frequencies -D unidic --form base --pos -o tubelex-ja-310-base-pos%.tsv.xz 	>> tubelex-ja.out
echo '[unidic-310] --form lemma --pos' 																			>> tubelex-ja.out
python tubelex.py --lang ja -x --frequencies -D unidic --form lemma --pos -o tubelex-ja-310-lemma-pos%.tsv.xz 	>> tubelex-ja.out
