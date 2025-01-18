# About TUBELEX

TUBELEX is a multi-lingual YouTube subtitle corpus. It currently provides data for Chinese, English, Indonesian, Japanese, and Spanish.

Word frequency in TUBELEX provides an approximation of everyday languages exposure comparable to, and often better than other resources, such as written corpora, Wikipedia, or film subtitle corpora (SUBTLEX, OpenSubtitles).

You may find TUBELEX useful for NLP applications modeling human familiarity with words, e.g. readability, text simplification, or language learning applications. TUBELEX log-frequencies are highly correlated with psycholinguistic data (lexical decision time, word familiarity) and lexical complexity.

Read [our paper](https://arxiv.org/abs/2410.03240) (to be presented at COLING 2025) for more details:
```
@article{nohejl_etal_2024_film,
  title={Beyond {{Film Subtitles}}: {{Is YouTube}} the {{Best Approximation}} of {{Spoken Vocabulary}}?},
  author={Nohejl, Adam and Hudi, Frederikus and Kardinata, Eunike Andriani and Ozaki, Shintaro and Riera Machin, Maria Angelica and Sun, Hongyu and Vasselli, Justin and Watanabe, Taro},
  year={2024}, eprint={2410.03240}, archiveprefix={arXiv}, primaryclass={cs.CL},
  url={https://arxiv.org/abs/2410.03240v1}, journal={ArXiv preprint}, volume={arXiv:2410.03240v1 [cs]}
}
```

This repository provides full source code for the project and word frequency lists. We also provide the following models on Hugging Face Hub:

- [fastText word embeddings](https://huggingface.co/naist-nlp/tubelex-fasttext)
- [KenLM n-gram models](https://huggingface.co/naist-nlp/tubelex-kenlm)

Note that the full text of the corpus cannot be published for copyright reasons. To enable use of TUBELEX in a wide range of applications, we offer frequency lists in multiple variants and the two above-mentioned types of basic language models. The frequency lists also include frequencies by video category, and dispersion (range or “contextual diversity”).

## Word Frequency Lists

Frequency lists in the default tokenization and normalization for the impatient:

- [Chinese](frequency/tubelex-zh.tsv.xz)
- [English](frequency/tubelex-en.tsv.xz)
- [Indonesian](frequency/tubelex-id.tsv.xz)
- [Japanese](frequency/tubelex-ja.tsv.xz)
- [Spanish](frequency/tubelex-es.tsv.xz)

All TUBELEX frequency files are TSV files compressed with LZMA (`xz`) with the following columns:

- `word` – the word (see below for tokenization/lemmatization and normalization),
- `count` – number of occurrences of the word,
- `videos` – number of videos containing the word,
- `channels` – number of channels the word occurrs in,
- `count:`*C* – number of occurrences of the word in the YouTube video category *C*.

All files also provide a row of totals as the last row (`[TOTAL]`).

The columns `videos`, `channels` provide dispersion information as count of corpus parts, in which each word occurs in. It can also be easily determined for categories using the columns `count:`*C*. This measure of dispersion is called range, contextual diversity, or document frequency. You might want to try the logarithm of `channels` as a feature for your model instead of log-frequency.

We provide the following [word frequency lists](frequencies) described in our paper for each language identified with a 2-letter ISO code *L*:

- default: `tubelex-`*L*`.tsv.xz` (for *L* in `en`, `es`, `id`, `ja`, `zh`; direct links to each above)
- base: `tubelex-`*L*`-base-pos.tsv.xz` (for *L* = `ja`)
- lemma: `tubelex-`*L*`-lemma-pos.tsv.xz` (for *L* in `en`, `es`, `id`, `ja`)
- regex: `tubelex-`*L*`-regex.tsv.xz` (for *L* in `en`, `es`, `id`)

Note that the `lemma` and `base` variants contain the majority POS for each lemma (base form) in the additional column `pos`.

Additionally, we provide:

- segmentation using UniDic 3.1.0 (instead of `unidic-lite`) for Japanese as `tubelex-ja-310.tsv.xz`, `tubelex-ja-310-lemma-pos.tsv.xz`, `tubelex-ja-310-base-pos.tsv.xz`,
- Penn Treebank segmentation for English as `tubelex-en-treebank.tsv.xz`,
- frequencies with majority POS information for each word for Chinese as `tubelex-zh-pos.tsv.xz`.

All of the above are lowercased and Unicode NFKC normalized (as described in our paper). We also provide [variants of the above files with alternative normalizations](frequencies/alternative-normalizations) with filenames with the following suffixes:

- only lowercased: `_lower.tsv.xz`,
- only normalized to NFKC: `_nfkc.tsv.xz`,
- no normalization: `_no-normalization.tsv.xz`.

## Ongoing Work

We are currently working on:

- extending TUBELEX to more languages,
- acquiring larger corpora for each language,
- acquiring more metadata and information that we could make public,
- investigating dispersion measures based on TUBELEX ([preprint](https://arxiv.org/abs/2501.06536)).


## How to (Re)Constructing the Corpus

You can re-construct TUBELEX by following the steps below. By modifying the scripts, it is possible to construct corpora for other languages or with different parameters (larger size, different tokenizations etc.)

1. Install the Git submodule for scraping data from YouTube (forked from JTubeSpeech):

    ```git submodule init && git submodule update```
    
    
    Note that the forked submodule is substantially different from JTubeSpeech, and can run most of the steps in parallel.
    
2. Install requirements (see [requirements.txt](requirements.txt)). The `unidic` package (as opposed to `unidic-lite`) requires an additional installation step:
	
	```python -m unidic download```

3. Scrape manual subtitles. The process consists of several substeps, which we have parallelized using shell scripts and GNU `parallel`. To adjust it to your environment, inspect the shell scripts and change the parameters as necessary. Although we have changed the internal workings of the [original JTubeSpeech](https://github.com/sarulab-speech/jtubespeech) scripts a little, you may also find their outline of the process helpful.

Do the following substeps in the `jtubespeech-subtitles` subdirectory:

	a. Make search words based on Wikipedia:
  
	  ```
	  bash make_search_words.sh
	  ```
		
	  This will create a file `word/tasks.csv` being created with chunks of the search words lists for the next step.
		
	b. Get video IDs by searching for the collected words:
	
	  ```
	  bash obtain_video_id_parallel.sh
	  ```
	  
	  This will automatically run Python scripts in parallel (using GNU `parallel`), one for each of your CPUs.
	  
	c. Prepare tasks for the next step:
	
	  ```
	  bash prepare_tasks_from_obtained.sh
	  ```
	  
	  This will create files named `videoid/tasks_enesidjazh_part`*XXXXXX*, where *XXXXXX* are numbers from 0 to *N* - 1 (depending on the number of videos found).
	
	d. Retrieve subtitle metadata: As this takes a relatively long time, we have divided this step into many tasks, that you can run (optionally in paralell). Each task can be expected to run a few hours. For *i* in 0 to *N* - 1, run the tasks prepared in the previous step:
	
	  ```
	  bash retrieve_subtitle_exists.sh *i*
	  ```
	
	e. Sample 120.000 subtitle files fulfilling the inclusion criteria for each language:
	
	  ```
	  sample.sh
	  ```
	
	f. Download the subtitles:

	  ```
	  bash download_video_parallel.sh
	  ```

5. Clean, remove duplicates and compute frequencies saving output with LZMA compression in the current directory:
    
    ```
    python tubelex.py -x --clean --unique
    python tubelex.py -x --frequencies -o tubelex-ja%.tsv.xz
    python tubelex.py -x --frequencies -D unidic -o tubelex-ja-310%.tsv.xz
    ```

6. Alternatively, consult the help and process the files as you see fit:

    ```python tubelex.py --help```
    
7. Optionally remove the language identification model, intermediate files, and the downloaded subtitles to save disk space:

    ```rm *.ftz *.zip; rm -r jtubespeech/video```


## How to Replicate the Experiments

To replicate the experiments in our paper you will need the following files placed in the `data` directory. We could not distribute them because their license wasn't clear or didn't allow redistribution:

- [Word GINI](https://sociocom.naist.jp/word-gini-en/) files `GINI_en.csv` and `GINI_ja.csv`,
- `elexicon.csv` file available via word generation form at the [English Lexicon Project](https://elexicon.wustl.edu),
- `MELD-SCH.csv`, [MELD-SCH](https://link.springer.com/article/10.3758/s13428-017-0944-0#Sec13) database, available online as a supplementary Excel file "ESM 1", converted to UTF-8 CSV (using Excel),
- `Clark-BRMIC-2004`, (we use only the `Clark-BRMIC-2004/cp2004b.txt` file), expanded zip archive, available online as a supplementary material for [English norms (Clark and Paivio, 2004)](https://link.springer.com/article/10.3758/BF03195584#SecESM1),
- `en-glasgow.csv` [Glasgow norms](https://link.springer.com/article/10.3758/s13428-018-1099-3#Sec13), available online as a supplementary CSV file "ESM 2",
- `es-alonso-oral-freq.tsv`, available online as a supplementary material for [Spanish oral frequencies by Alonso et al. 2011](https://link.springer.com/article/10.3758/s13428-011-0062-3#SecESM1), concatenated two “columns” into one and exported to UTF-8 TSV,
- `es-guasch.csv`, [Spanish norms (Guasch et al., 2014)](https://link.springer.com/article/10.3758/s13428-015-0684-y#Sec13) database, available online as a supplementary Excel file "ESM 1", converted to UTF-8 CSV (using Excel), 
- `es-moreno-martinez.csv`, [Spanish norms (Moreno-Martínez et al., 2014)](https://link.springer.com/article/10.3758/s13428-013-0435-x#Sec22) database, available online as a supplementary Excel file "ESM 1", converted to UTF-8 CSV (using Excel),
- `amano-kondo-1999-ntt/*.csv`, CSV files of the tables from the [Amano-Kondo NTT database (1999)](https://ci.nii.ac.jp/ncid/BA44537988). You can extract the files from the first CD-ROM containing a Windows installer like this:

  ```
  # The CD contains a Win98 installer, which decompresses and installs files on your
  # computer. The files are a database and a program to browse the database.
  # Here we just decompress the database (DB0001.MDB) and extract tables from it as CSV.
  # To do so, you will first need to install two software packages (via brew).
  
  # Install 7zz (sevenzip) and mdbtools:
  brew install sevenzip mdbtools
  
  # Decompress:
  7zz x CD1/DB0001.MD_ -so > DB0001.MDB
  
  # Extract tables as CSV:
  mkdir amano-kondo-1999-ntt
  for t in $(mdb-tables DB0001.MDB); do mdb-export DB0001.MDB $t > "amano-kondo-1999-ntt/${t}.csv"; done
  ```

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
