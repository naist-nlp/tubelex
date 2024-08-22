# About TUBELEX


## TODO: This is slightly out of date, and mostly talks about Japanese, the paper draft currently contains more accurate (although less detailed) description of the process.

Inspired by the SUBTLEX word lists, TUBELEX-JA is a large word list based on Japanese subtitles for YouTube videos (93M tokens from 72k subtitle files).

The project consists mainly of:

- [tubelex.py](tubelex.py): a script to create the word list
- the word list in several mutations:
  - segmented with Unidic Lite:
	- [tubelex-ja.tsv.xz](results/tubelex-ja.tsv.xz): no normalization,
	- [tubelex-ja-lower.tsv.xz](results/tubelex-ja-lower.tsv.xz): no normalization, lowercased,
	- [tubelex-ja-nfkc.tsv.xz](results/tubelex-ja-lower.tsv.xz): NFKC normalization
	- [tubelex-ja-nfkc-lower.tsv.xz](results/tubelex-ja-lower.tsv.xz): NFKC normalization, lowercased
  - segmented with Unidic 3.1.0:
	- [tubelex-ja-310.tsv.xz](results/tubelex-ja.tsv.xz): no normalization,
	- [tubelex-ja-310-lower.tsv.xz](results/tubelex-ja-lower.tsv.xz): no normalization, lowercased,
	- [tubelex-ja-310-nfkc.tsv.xz](results/tubelex-ja-lower.tsv.xz): NFKC normalization
	- [tubelex-ja-310-nfkc-lower.tsv.xz](results/tubelex-ja-lower.tsv.xz): NFKC normalization, lowercased
	
For each word, we count:
- number of occurrences,
- number of videos,
- number of channels.

For a small number of videos, there is no channel information, so we count them as separate single-video channels. Words occurring in less than 3 videos are not included. The list is sorted by the number of occurrences. The data is tab-separated with a header, and the file is compressed with LZMA2 (`xz`).

**Important:** The last row labeled `[TOTAL]` lists the **total numbers** of tokens, videos, and channels, and thus may **require special handling**. Also, note that the totals are not sums of the previous rows' values.

## About mutations

Words are segmented with MeCab using [Unidic Lite](https://github.com/polm/unidic-lite), or alternatively with Unidic 3.1.0. Words that contain decimal digits (except kanji characters for numbers) and words that start or end with a non-word character (e.g. punctuation) are ignored in both cases. The [full-width tilde](https://ja.wikipedia.org/wiki/チルダ#全角チルダ) character (0xFF5E, '～') is replaced by the [wave dash](https://ja.wikipedia.org/wiki/波ダッシュ) character (0x301C, '〜') before tokenization. The former is almost always a visually indistinguishable typo. We consider the wave dash a word-forming character similarly to alphabet, kanji etc.

Unidic Lite is often used for Japanese segmentation in Python due to its ease of installation (see  [Fugashi](https://pypi.org/project/fugashi/) for more info). Unidic Lite is also used for tokenization of [a commonly used Japanese BERT model](https://huggingface.co/cl-tohoku/bert-base-japanese-v2). That said, Unidic 3.1.0 is, of course, larger and more up-to-date.

One of the possible motivations for using [NFKC](http://unicode.org/reports/tr15/)-normalized data is that `BertJapaneseTokenizer` class used by the aforementioned model performs it by default. Among others, letters of the Latin alphabet and katakana are normalized (to ordinary/half-width, and full-width, respectively) in NFKC, which is quite reasonable. On the other hand, since we do it after segmentation, it reintroduces tokens containing decimal digits (e.g. by converting "⑮" to "15").

The lowercasing concerns not only the letters A-Z but also accented characters and letters of any cased alphabet (e.g. Ω is lowercased to ω).

## About corpus and processing

As a basis for the corpus, we used manual subtitles listed in the file `data/ja/202103.csv` from the [JTubeSpeech](https://github.com/sarulab-speech/jtubespeech) repository that were still available as of 30 November 2022. (The script for downloading is also part of that repository.) The download subtitles were then processed using the [tubelex.py](tubelex.py) script according to the following steps:

1. Extract lines of subtitles and convert HTML (e.g. `&amp;`) entities to characters.

2. Remove the following text sequences:
    - formatting tags,
    - addresses (`http`(`s`), e-mail, domain names starting with `www.`, social network handles starting with `@`).

3. Remove the following lines:
    - empty lines,
    - lines repeating the previous line,
    - lines composed entirely of non-Japanese characters,

4. Remove the following files:
    - < 3 lines,
    - < 70 % Japanese characters,
    - < 95 % lines identified as Japanese language using a FastText model,

5. Remove near-duplicates of other files.

6. Create the word list (both raw and normalized) as described initially.
  
Near-duplicates are files with cosine similarity >= 0.95 between their 1-gram TF-IDF vectors. We make a reasonable effort to minimize the number of duplicates removed. See the source code for more details on this and other points. For consistency, we have used Unidic Lite for building TF-IDF regardless of the final segmentation used for frequency counting.

Note that the script saves intermediate files after cleaning and removing duplicates, and has various options we do not describe here (see `python tubelex.py --help`).

# Usage

Note that the output of the script is already included in the repository. You can, however, reproduce it by following the steps below. Results will vary based on the YouTube videos/subtitles still available for download.

1. Install the Git submodule for JTubeSpeech-subtitles:

    ```git submodule init && git submodule update```
    
2. Install requirements for both tubelex and JTubeSpeech:

    - JTubeSpeech:
        - Install `youtube-dl`, e.g. using Homebrew:
	
          ```brew install youtube-dl```
	
          Python packages for JTubeSpeech are covered by tubelex requirements.
	  
    - Tubelex:
        - See [requirements.txt](requirements.txt).
        - The `unidic` package (as opposed to `unidic-lite`) requires an additional installation step:
	
	  ```python -m unidic download```

3. Download manual subtitles using the download script:

    ```cd jtubespeech; python scripts/download_video.py ja data/ja/202103.csv; cd ..```

4. Clean, remove duplicates and compute frequencies saving output with LZMA compression in the current directory:
    
    ```
    python tubelex.py -x --clean --unique
    python tubelex.py -x --frequencies -o tubelex-ja%.tsv.xz
    python tubelex.py -x --frequencies -D unidic -o tubelex-ja-310%.tsv.xz
    ```

5. Alternatively, consult the help and process the files as you see fit:

    ```python tubelex.py --help```
    
6. Optionally remove the language identification model, intermediate files, and the downloaded subtitles to save disk space:

    ```rm *.ftz *.zip; rm -r jtubespeech/video```

# Results

After cleaning and duplicate removal, there are **93,215,459 tokens**. The word list consists of **127,421 words** occurring in at least 3 videos (Unidic Lite segmentation, no normalization, no lowercasing). The number of words differs slightly for other mutations.

## Cleaning statistics (steps 2-4 above):

* files (determined after sequence removal and line filtering):
  - 103887 total
  - 7689 too short (<3 lines)
  - 4925 not enough J. characters (<0.7 characters)
  - 16941 not enough J. lines (<0.95 identified as Japanese using a FastText model)
  - 74332 valid files after cleaning
* sequences removed from valid files:
  - 129280 tags
  - 1659 addresses
* lines in valid files:
  - 8108028 total lines
  - 2004 whitespace-only lines
  - 41200 repeated lines
  - 61780 lines composed of non-Japanese characters
  - 8003044 valid lines after cleaning

## Duplicate removal statistics (step 5 above):
  - 74332 total
  - 1767 duplicates removed
  - 72565 valid files

# Further work and similar lists

The word list contains only the surface forms of the words (segments). For many purposes, lemmas, POS and other information would be more useful. We plan to add further processing later.

We have yet to attempt to analyze the corpus/word list, or compare it with word lists based on smaller but more carefully curated corpora of spoken Japanese. The largest corpus of such kind would be [CSJ](https://clrd.ninjal.ac.jp/csj/index.html) (7M tokens, with publicly available [word lists](https://clrd.ninjal.ac.jp/csj/chunagon.html#data)). Smaller corpora include [CEJC](https://www2.ninjal.ac.jp/conversation/corpus.html), [NUCC](https://mmsrv.ninjal.ac.jp/nucc/), [J-TOCC](http://nakamata.info/database/), and [BTSJ](https://ninjal-usamilab.info/btsj_corpus/).

Note that there is also a large corpus based on TV subtitles [LaboroTVSpeech](https://laboro.ai/activity/column/engineer/eg-laboro-tv-corpus-jp/) (22M tokens), which can be used for free for academic purposes (application necessary).

You may also like [wikipedia-word-frequency-clean](https://github.com/adno/wikipedia-word-frequency-clean), a repository of word frequency lists for multiple languages from cleaned-up Wikipedia dumps, which are processed in a similar way to TUBELEX-JA and also include Japanese.



# Replicating experiments

To replicate our experiments you will need the following files placed in the data directory. We could not distribute them because their license wasn't clear or didn't allow redistribution:

- [Word GINI](https://sociocom.naist.jp/word-gini-en/) files `GINI_en.csv` and `GINI_ja.csv`,
- `elexicon.csv` file available via word generation form at the [English Lexicon Project](https://elexicon.wustl.edu),
- `MELD-SCH.csv`, [MELD-SCH](https://link.springer.com/article/10.3758/s13428-017-0944-0#Sec13) database, available online as a supplementary Excel file "ESM 1", converted to UTF-8 CSV (using Excel),
- `es-moreno-martinez.csv`, [Spanish norms (Moreno-Martínez et al., 2014)](https://link.springer.com/article/10.3758/s13428-013-0435-x#Sec22) database, available online as a supplementary Excel file "ESM 1", converted to UTF-8 CSV (using Excel), 
- `Lexeed.txt`, file available from the CD-ROM accompanying [NTT Database Series: Lexical Properties of Japanese](https://ci.nii.ac.jp/ncid/BA44537988) by Amano Shigeaki and Kondo Tadahisa (1999-2022), i.e. the Heisei edition of the database.
- `subimdb.tsv` file, which you can generate by first downloading and extracting the [SubIMDB](https://zenodo.org/records/2552407/files/SubIMDB_All_Individual.tar?download=1) corpus into the `SubIMDB_All_Individual` directory, and then compiling the frequency list with the following command:


    ```python tubelex.py --lang en -x --frequencies --tokenized-files SubIMDB_All_Individual/subtitles -o data/subimdb.tsv```

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