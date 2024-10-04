# About TUBELEX

TUBELEX is a YouTube subtitle corpus currently available for Chinese, English, Indonesian, Japanese, and Spanish.

See our [paper preprint](paper).

This repository provides full source code for the project and [word frequency lists](frequencies). We also provide the following models on Hugging Face Hub:

- [fastText word embeddings](https://huggingface.co/naist-nlp/tubelex-fasttext)
- [KenLM n-gram models](https://huggingface.co/naist-nlp/tubelex-kenlm)


# Replicating Experiments

To replicate our experiments you will need the following files placed in the data directory. We could not distribute them because their license wasn't clear or didn't allow redistribution:

- [Word GINI](https://sociocom.naist.jp/word-gini-en/) files `GINI_en.csv` and `GINI_ja.csv`,
- `elexicon.csv` file available via word generation form at the [English Lexicon Project](https://elexicon.wustl.edu),
- `MELD-SCH.csv`, [MELD-SCH](https://link.springer.com/article/10.3758/s13428-017-0944-0#Sec13) database, available online as a supplementary Excel file "ESM 1", converted to UTF-8 CSV (using Excel),
- `Clark-BRMIC-2004`, (we use only the `Clark-BRMIC-2004/cp2004b.txt` file), expanded zip archive, available online as a supplementary material for [English norms (Clark and Paivio, 2004)](https://link.springer.com/article/10.3758/BF03195584#SecESM1),
- `en-glasgow.csv` [Glasgow norms](https://link.springer.com/article/10.3758/s13428-018-1099-3#Sec13), available online as a supplementary CSV file "ESM 2",
- `es-alonso-oral-freq.tsv`, available online as a supplementary material for [Spanish oral frequencies by Alonso et al. 2011](https://link.springer.com/article/10.3758/s13428-011-0062-3#SecESM1), concatenated two “columns” into one and exported to UTF-8 TSV,
- `es-guasch.csv`, [Spanish norms (Guasch et al., 2014)](https://link.springer.com/article/10.3758/s13428-015-0684-y#Sec13) database, available online as a supplementary Excel file "ESM 1", converted to UTF-8 CSV (using Excel), 
- `es-moreno-martinez.csv`, [Spanish norms (Moreno-Martínez et al., 2014)](https://link.springer.com/article/10.3758/s13428-013-0435-x#Sec22) database, available online as a supplementary Excel file "ESM 1", converted to UTF-8 CSV (using Excel),
- `Lexeed.txt`, file available from the CD-ROM accompanying [NTT Database Series: Lexical Properties of Japanese](https://ci.nii.ac.jp/ncid/BA44537988) by Amano Shigeaki and Kondo Tadahisa (1999-2022), i.e. the Heisei edition of the database.
- `subimdb.tsv` file, which you can generate by first downloading and extracting the [SubIMDB](https://zenodo.org/records/2552407/files/SubIMDB_All_Individual.tar?download=1) corpus into the `SubIMDB_All_Individual` directory, and then compiling the frequency list with the following command:

    ```python tubelex.py --lang en --frequencies --tokenized-files SubIMDB_All_Individual/subtitles -o data/subimdb.tsv```
    
- `laborotvspeech.tsv` file, which you can generate by first downloading and extracting the [LaboroTVSpeech](https://laboro.ai/activity/column/engineer/eg-laboro-tv-corpus-jp/) and [LaboroTVSpeech2](https://laboro.ai/activity/column/engineer/laborotvspeech2/) (both are free for academic use; you do not need to extract the `*.wav` files) as `laborotvspeech/LaboroTVSpeech_v1.0b` and `laborotvspeech/LaboroTVSpeech_v2.0b` directories, and then compiling the frequency list with the following command:

    ```python tubelex.py --lang en --frequencies --laborotv --tokenized-files laborotvspeech -o data/laborotvspeech.tsv```

- `hkust-mtcs.tsv` file, which you can generate by first downloading and extracting [transcripts of the HKUST/MTSC corpus](https://catalog.ldc.upenn.edu/LDC2005T32), into the `LDC2005T32` directory, and then compiling the frequency list with the following command:

	```python tubelex.py --lang zh --frequencies --hkust-mtsc --tokenized-files LDC2005T32/hkust_mcts_p1tr/data -o data/hkust-mtsc.tsv```

- `espal.tsv` file created by following these steps:
	1. Go to the [EsPal website](https://www.bcbl.eu/databases/espal/).
	2. Select "Subtitle Tokens (2012-10-05)". (Phonology doesn't matter.)
	3. Click "Words to Properties".
	4. Select "Word Frequency" > "Count"
	5. For N in 1...5 repeat steps 6 to 8:
	6. - Click "File with Items: Choose File" and select the file `data/es-words.`*N*`.txt`.
	7. - Click "Download"
	8. - Click "Search Again..."
	9. Remove UTF-8 BOM (bytes 0xEFBBBF) from each file, and the header line `word\tcnt` from each file except the first one.
	10. Concatenate the edited files to `data/espal.txt`.
	11. Remove lines not containing any count.
	11. Add `[TOTAL]\t462611693` as the last line (`\t` is the tab character).
We use a number of other files (e.g. SPALEX, Wikipedia frequencies, SUBTLEX-US, SUBTLEX-ESP), which are either included or downloaded automatically.
	12. Remove trailing tabs from all lines.
	13. The resulting file should have 35285 lines and 448608 bytes.
