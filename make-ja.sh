#!/bin/bash

#SBATCH -J tubelex-ja
#SBATCH -A lang
#SBATCH -p lang_short
#SBATCH -c1

DIR=frequencies

python tubelex.py --lang ja -x --clean --unique										 	>  tubelex-ja.out
echo '[unidic-lite]' 																	>> ${DIR}/tubelex-ja.out
python tubelex.py --lang ja -x --frequencies -o ${DIR}/${DIR}/tubelex-ja%.tsv.xz 						>> ${DIR}/tubelex-ja.out
echo '[unidic-310]' 																	>> ${DIR}/tubelex-ja.out
python tubelex.py --lang ja -x --frequencies -D unidic -o ${DIR}/tubelex-ja-310%.tsv.xz 		>> ${DIR}/tubelex-ja.out

echo '[unidic-lite] --form base --pos' 															>> ${DIR}/tubelex-ja.out
python tubelex.py --lang ja -x --frequencies --form base --pos -o ${DIR}/tubelex-ja-base-pos%.tsv.xz 	>> ${DIR}/tubelex-ja.out
echo '[unidic-lite] --form lemma --pos' 														>> ${DIR}/tubelex-ja.out
python tubelex.py --lang ja -x --frequencies --form lemma --pos -o ${DIR}/tubelex-ja-lemma-pos%.tsv.xz >> ${DIR}/tubelex-ja.out

echo '[unidic-310] --form base --pos' 																			>> ${DIR}/tubelex-ja.out
python tubelex.py --lang ja -x --frequencies -D unidic --form base --pos -o ${DIR}/tubelex-ja-310-base-pos%.tsv.xz 	>> ${DIR}/tubelex-ja.out
echo '[unidic-310] --form lemma --pos' 																			>> ${DIR}/tubelex-ja.out
python tubelex.py --lang ja -x --frequencies -D unidic --form lemma --pos -o ${DIR}/tubelex-ja-310-lemma-pos%.tsv.xz 	>> ${DIR}/tubelex-ja.out
