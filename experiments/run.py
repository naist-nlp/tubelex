import argparse
from collections import defaultdict
from typing import Optional
from collections.abc import Callable
import sys
import os
import re
from itertools import zip_longest
import pandas as pd
from csv import QUOTE_NONE
import numpy as np
import wordfreq as wf  # word_frequency, tokenize, get_frequency_dict
from itertools import chain
from typing import NamedTuple, Any
from tqdm import tqdm

from sklearn.linear_model import RidgeCV
from sklearn.metrics import (
    mean_squared_error as mse_score, mean_absolute_error as mae_score, r2_score
    )
from joblib import dump, load
from frequency_data import FrequencyData, download_if_necessary
from datasets import load_dataset, Dataset
from spalex import get_spalex

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

from tubelex import add_tokenizer_arg_group, get_tokenizers, nfkc_lower
from lang_utils import get_re_split


LANG2FULL_NAME: dict[str, str] = {
    'en': 'English',
    'ca': 'Catalan',
    'fil': 'Filipino',
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'ja': 'Japanese',
    'pt': 'Portuguese',
    'si': 'Sinhala',
    'es': 'Spanish',
    # non-MLSP:
    'zh': 'Chinese',
    'id': 'Indonesian'
    }

CAT_ID2CATEGORY = {
    'autos': 'Autos & Vehicles',
    'comedy': 'Comedy',
    'education': 'Education',
    'entertainment': 'Entertainment',
    'film': 'Film & Animation',
    'gaming': 'Gaming',
    'howto': 'Howto & Style',
    'music': 'Music',
    'news': 'News & Politics',
    'nonprofits': 'Nonprofits & Activism',
    'people': 'People & Blogs',
    'pets': 'Pets & Animals',
    'science': 'Science & Technology',
    'sports': 'Sports',
    'travel': 'Travel & Events'
    }


def pearson_r(x: np.ndarray, y: np.ndarray):
    return np.corrcoef(x, y)[0][1]


DATASET_NAME = 'MLSP2024/MLSP2024'
LANG2DATASET_ID = {
    'es': 'spanish_lcp_labels',
    'en': 'english_lcp_labels',
    'ja': 'japanese_lcp_labels'
    }


def get_mlsp_dataset(
    lang_or_id: str,
    train: bool = False,
    token: Optional[str] = None
    ) -> Dataset:
    input_id = LANG2DATASET_ID.get(lang_or_id, lang_or_id)
    return load_dataset(
        DATASET_NAME, input_id,
        split=('trial' if train else 'test'),
        token=token
        )


def get_sub_freq_data(language: str) -> FrequencyData:
    return FrequencyData.from_subtitles(language)


def get_wiki_freq_data(language: str) -> FrequencyData:
    return FrequencyData.from_wiki(language)


def get_gini_data(language: str) -> pd.Series:
    df = pd.read_csv(f'data/GINI_{language}.csv', na_filter=False, quoting=QUOTE_NONE)
    return df.set_index('Word')['GINI']


def get_gini_missing_func(language: str) -> Callable[[str], tuple[float, bool]]:
    gini = get_gini_data(language)
    max_gini = gini.max()

    def gini_missing_func(word: str) -> tuple[float, bool]:
        g = gini.get(word)
        return (
            (max_gini, True) if (g is None) else
            (g, False)
            )

    return gini_missing_func


ACTIV_ES_PATH = 'data/downloads/activ-es.csv'


def get_activ_es_data() -> pd.Series:
    download_if_necessary(
        'https://github.com/francojc/activ-es/raw/master/activ-es-v.02/'
        'wordlists/plain/aes1grams.csv', ACTIV_ES_PATH
        )
    df      = pd.read_csv(ACTIV_ES_PATH, index_col='word')
    return df['aes_orf'] / 100_000


def get_activ_es_freq_missing_func() -> Callable[[str], tuple[float, bool]]:
    freq    = get_activ_es_data()
    min_freq = freq.min()

    def activ_es_freq_missing_func(word: str) -> tuple[float, bool]:
        f = freq.get(word)
        return (
            (min_freq, True) if (f is None) else
            (f, False)
            )

    return activ_es_freq_missing_func


def get_ldt_data(language: str, prefer_zscore: bool = False) -> pd.Series:
    if language == 'en':
        ldt_col = 'I_Zscore' if prefer_zscore else 'I_Mean_RT'
        df = pd.read_csv(
            'data/elexicon.csv', na_values=['#'], keep_default_na=False, thousands=','
            )
        df = df.loc[df['Word'] == df['Word'].str.lower()]
        df = df[~df[ldt_col].isna()]
        series = df.set_index('Word')[ldt_col]
    elif language == 'es':
        df = get_spalex()
        series = df.set_index('word')['rt']
    elif language == 'zh':
        ldt_col = 'zRT' if prefer_zscore else 'RT'
        df = pd.read_csv('data/MELD-SCH.csv')
        is_w = (df['lexicality'] == 1)
        df = df[~df[ldt_col].isna() & is_w]
        series = df.set_index('word')[ldt_col]
    else:
        raise Exception(f'No LDT data for {language}')

    print(f'Loaded LDT:{language}[{series.name}]: {len(series)}', file=sys.stderr)
    return series


def get_familiarity_data(
    # IMPORTANT: STATS_DATASETS depends on the number/order of arguments!
    language: str,
    glasgow: bool = False,
    clark_paivio: bool = False,
    moreno_martinez: bool = False,
    amano: bool = False
    # IMPORTANT: STATS_DATASETS depends on the number/order of arguments!
    ) -> pd.Series:
    if language == 'zh':
        fam_col = 'FAM_M'
        df = pd.read_table('data/chinese-familiarity.tsv.xz')
        df = df[~df[fam_col].isna()]    # removes comments
        series = df.set_index('WORD')[fam_col]
    elif language == 'id':
        fam_col = "ALL_Frequency_Mean"
        df = pd.read_csv('data/indonesian-subjective-frequency.csv.xz')
        df = df[~df[fam_col].isna()]    # removes comments
        series = df.set_index('Words (Indonesian)')[fam_col]
    elif language == 'en':
        assert not (glasgow and clark_paivio)
        if glasgow:
            fam_col = 'FAM'
            df = pd.read_csv('data/en-glasgow.csv', header=[0, 1])
            # Drop the second level, we just need to axcess the Words and (FAM) means,
            # (and the CSV format is an irreparable mess anyway):
            df.columns = df.columns.droplevel(1)
            # There are no N/A values.
            # Words contains multi-sense items such as:
            # 'shell', 'shell (military)', 'shell (sea)'
            # in which case we only keep the unspecified ones ('shell'):
            df = df[~df['Words'].str.contains('(', regex=False)]
            series = df.set_index('Words')[fam_col]
        elif clark_paivio:
            fam_col = "FAM"
            df = pd.read_csv('data/Clark-BRMIC-2004/cp2004b.txt', delimiter=r'\s+')
            df = df[~df[fam_col].isna()]            # removes N/A values
            df['WORD'] = df['WORD'].str.lower()    # lowercase (words are all UPPERCASE)
            series = df.set_index('WORD')[fam_col]
        else:
            # MRC (largest/default)
            fam_col = "FAM"
            df = pd.read_csv('data/mrc.csv')
            df = df[~df[fam_col].isna()]    # removes N/A values
            series = df.set_index('WORD')[fam_col]
    elif language == 'es':
        if moreno_martinez:
            fam_col = "Fam"
            df = pd.read_csv('data/es-moreno-martinez.csv')
            df = df[~df[fam_col].isna()]                # remove N/A values
            df['Spanish'] = df['Spanish'].str.strip()   # strip spaces
            series = df.set_index('Spanish')[fam_col]
        else:
            # Guasch+2016 (largest/default)
            fam_col = "FAM_M"
            df = pd.read_csv('data/es-guasch.csv')
            df = df[~df[fam_col].isna()]                # remove N/A values
            df['Word'] = df['Word'].str.strip()   # strip spaces
            series = df.set_index('Word')[fam_col]
    elif language == 'ja':
        if amano:
            fam_col = '文字単語親密度'
            df = pd.read_csv('data/amano-kondo-1999-ntt/単語親密度.csv')
            # Remove -1 values (N/A): 88569 -> 88494 entries:
            df = df.loc[df[fam_col] >= 0]
            # We only take 文字単語親密度 (not e.g. 文字音声単語親密度, 音声単語親密度)
            # Entries only differing in written form (表記), if there are multiple
            # values, take mean (-> 76883 entries).
            series = df.set_index('表記').groupby(level=0)[fam_col].mean()
        else:
            # Larger and more recent WLSP:
            fam_col = '知っている'
            download_if_necessary(
                'https://github.com/masayu-a/WLSP-familiarity/raw/4.0/bunruidb-fam.csv',
                'data/downloads/bunruidb-fam.csv'
                )
            df = pd.read_csv('data/downloads/bunruidb-fam.csv')
            df = df[~df[fam_col].isna()]                # remove N/A values
            # Entries only differing in written form (見出し本体), if there are multiple
            # values, take mean (-> 81271 entries).
            series = df.set_index('見出し本体').groupby(level=0)[fam_col].mean()
    else:
        raise Exception(f'No familiarity data for {language}')

    print(
        f'Loaded familiarity:{language}[{series.name}]: {len(series)}',
        file=sys.stderr
        )
    return series


def get_tubelex_freq_data(
    language: str,
    tokenization: Optional[str] = None,     # regex, treebank
    form: str = 'surface',                  # surface, base, lemma
    category: Optional[str] = None
    ) -> FrequencyData:

    if tokenization is not None:
        assert form == 'surface'
        assert tokenization in {'regex', 'treebank'}
        language = f'{language}-{tokenization}'  # e.g. 'en-regex'
    else:
        assert form in {'surface', 'base', 'lemma'}
        assert form != 'base' or language == 'ja'  # 'base' only available for Japanese
        if form == 'lemma' and language == 'zh':
            form == 'surface'                    # consider lemma == surface for Chinese
        if form != 'surface':
            language = f'{language}-{form}-pos'  # note: we ignore POS

    cols = (
        ('word', f'count:{category}') if category is not None
        else None
        )

    if language == 'es':
        language = 'es-regex'
    return FrequencyData.from_file_url(
        filename=f'frequencies/tubelex-{language}-nfkc-lower.tsv.xz',
        total_row=True,
        cols=cols
        )


def get_opensubtitles_freq_data(
    language: str,
    ) -> FrequencyData:
    return FrequencyData.from_file_url(
        filename=(
            'data/os_zh_cn.tsv.xz' if (language == 'zh') else
            f'data/os_{language}.tsv.xz'
            ),
        total_row=False, to_lower=True,
        cols=('word', 'count')
        )


def get_csj_freq_data(
    tokenization: Optional[str] = None,
    form: str = 'lemma'
    ) -> FrequencyData:
    if (tokenization is not None) or (form != 'lemma'):
        raise Exception(
            'CSJ data is only available in lemma form and default (MeCab) tokenization.'
            )
    return FrequencyData.from_file_url(
        filename='data/downloads/CSJ_frequencylist_suw_ver201803.zip',
        url=('https://repository.ninjal.ac.jp/record/3276/files/'
             'CSJ_frequencylist_suw_ver201803.zip'),
        total_row=False,
        cols=['lemma', 'frequency']
        )


def get_laborotv_freq_data(
    dictionary: str = 'ipadic',
    tokenization: Optional[str] = None,
    form: str = 'surface'
    ) -> FrequencyData:
    if (
        (dictionary != 'ipadic') or
        (tokenization is not None) or (form != 'surface')
        ):
        raise Exception(
            'LaboroTVSpeech data is only available in IPADIC tokenization (-d IPADIC).'
            )
    return FrequencyData.from_file_url(
        filename='data/laborotvspeech.tsv', total_row=True
        )


def get_subimdb_freq_data() -> FrequencyData:
    return FrequencyData.from_file_url(filename='data/subimdb.tsv', total_row=True)


def get_hkust_mtsc_freq_data() -> FrequencyData:
    return FrequencyData.from_file_url(filename='data/hkust-mtsc.tsv', total_row=True)


def get_bnc_spoken_written_freq_data(
    spoken: bool = False,
    written: bool = False
    ) -> FrequencyData:

    assert written or spoken    # at least one of them

    if spoken:
        fd  = FrequencyData.from_file_url(
            url='https://www.kilgarriff.co.uk/BNClists/all.num.gz',
            filename='data/downloads/bnc_all.num.gz',
            header=['freq', 'word', 'pos', 'cd'], cols=['word', 'freq', 'cd'],
            delimiter=' ', total_row='!!WHOLE_CORPUS'
            )
        if written:
            return fd

    assert written != spoken  # either spoken or written

    fdw = FrequencyData.from_file_url(
        url='https://www.kilgarriff.co.uk/BNClists/written.num.gz',
        filename='data/downloads/bnc_written.num.gz',
        header=['freq', 'word', 'pos', 'cd'], cols=['word', 'freq', 'cd'],
        delimiter=' ', total_row='!!ANY'
        )
    if written:
        return fdw

    assert spoken and not written

    return fd.difference(fdw)


def get_espal_freq_data() -> FrequencyData:
    return FrequencyData.from_file_url(
        filename='data/espal.tsv',
        cols=['word', 'cnt'],
        total_row=True
        )


def get_alonso_freq_data() -> FrequencyData:
    return FrequencyData.from_file_url(
        filename='data/es-alonso-oral-freq.tsv',
        cols=['Word', 'Frequency'],
        total_row=False
        )


wf_min_freq = {}


def get_wf_min_freq(lang: str) -> float:
    if (f := wf_min_freq.get(lang)) is None:
        f = min(wf.get_frequency_dict(lang).values())
        wf_min_freq[lang] = f
    return f


LANG2KYTEA_MODEL = {
    'ja': 'data/downloads/kytea/jp-0.4.7-5.mod',
    'zh': 'data/downloads/kytea/lcmc-0.4.0-5.mod'
    }
LANG2KYTEA_MODEL_GZ = {
    'ja': 'data/downloads/kytea/jp-0.4.7-5.mod.gz',
    'zh': 'data/downloads/kytea/lcmc-0.4.0-5.mod.gz'
    }
LANG2KYTEA_MODEL_URL = {
    'ja': 'http://www.phontron.com/kytea/download/model/jp-0.4.7-5.mod.gz',
    'zh': 'http://www.phontron.com/kytea/download/model/lcmc-0.4.0-5.mod.gz'
    }


def get_kytea_tokenizer(lang: str) -> Callable[[str], list[str]]:
    model_path = LANG2KYTEA_MODEL[lang]
    if not os.path.exists(model_path):
        gz_path = LANG2KYTEA_MODEL_GZ[lang]
        download_if_necessary(LANG2KYTEA_MODEL_URL[lang], gz_path)
        import gzip
        import shutil
        with gzip.open(gz_path, 'rb') as f_in, open(model_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    import Mykytea
    mk = Mykytea.Mykytea(f'-model {model_path}')

    def mk_tokenize(s: str) -> list[str]:
        return list(mk.getWS(s))

    return mk_tokenize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--stats', action='store_true',
                        help='Write stats for corpora and datasets.')
    action.add_argument('--train', action='store_true',
                        help='Train linear regression.')
    action.add_argument('--correlation', action='store_true',
                        help='Compute correlation.')

    parser.add_argument('--stats-datasets', default='experiments/stats-datasets.csv',
                        help='Output CSV file for dataset stats.')
    parser.add_argument('--stats-corpora', default='experiments/stats-corpora.csv',
                        help='Output CSV file for corpus stats.')

    parser.add_argument('--cache-tubelex', action='store_true', help=(
        'Cache TUBELEX frequencies for correlation pvalue computation.'
        ))
    parser.add_argument('--tubelex-cache', default='experiments/cache', help=(
        'Cache directory for TUBELEX frequencies (--cache-tubelex).'
        ))

    parser.add_argument(
        '--metrics', '-m', action='store_true',
        help='Print metrics when doing inference.'
        )
    input_data = parser.add_mutually_exclusive_group()
    input_data.add_argument(
        '--mlsp', nargs='+',
        help='MLSP (lexical complexity) subset names (Hugging Face)'
        )
    input_data.add_argument(
        '--ldt', nargs='+',
        help='Language codes for LDT (only for --correlation)'
        )
    input_data.add_argument(
        '--familiarity', nargs='+',
        help='Language codes for familiarity (only for --correlation)'
        )
    parser.add_argument(
        '--clark-paivio', action='store_true',
        help='Use Clark-Paivio for English familiarity.'
        )
    parser.add_argument(
        '--glasgow', action='store_true',
        help='Use Glasgow norms for English familiarity.'
        )
    parser.add_argument(
        '--moreno-martinez', action='store_true',
        help='Use Moreno-Martinez+2014 for Spanish familiarity.'
        )
    parser.add_argument(
        '--amano', action='store_true',
        help='Use the Heisei NTT DB (Amano and Kondo, 1999) for Japanese familiarity.'
        )
    parser.add_argument(
        '--zscore', '-z', action='store_true',
        help='Prefer z-score for LDT (applies to en, zh).'
        )
    parser.add_argument(
        '--wordfreq-retokenize', action='store_true',
        help='Allow wordfreq to retokenize words.'
        )
    parser.add_argument(
        '--no-c-j-tokenize', action='store_true',
        help='Do not tokenize Chinese and Japanese.'
        )
    parser.add_argument('--output-files', '-o', nargs='*', default=[], help=(
        'Output files in LCP format when doing inference. '
        'If single name is given for multiple input files, '
        'it is understood as a directory name for multiple files. '
        'Default: experiments/output (experiments/output.tsv).'
        ))
    parser.add_argument(
        '--subtlex', nargs='*',
        default=[], help=(
            'Use SUBTLEX for these language codes.'
            )
        )
    parser.add_argument(
        '--subtlex-uk', action='store_true',
        default=[], help='Use SUBTLEX-UK for English.'
        )
    parser.add_argument(
        '--opensubtitles', '--os', nargs='*',
        default=[], help=(
            'Use OpenSubtitles2018 for these language codes (zh->zh_cn). '
            'Overrides --subtlex.'
            )
        )
    parser.add_argument(
        '--tubelex', nargs='*',
        default=[], help=(
            'Use TUBELEX for these language codes. Overrides --opensubtitles.'
            )
        )
    parser.add_argument(
        '--wordfreq', nargs='*',
        default=[], help=(
            'Use wordfreq for these language codes. '
            '(Already default, specify to compute correlation with TUBELEX.)'
            )
        )
    parser.add_argument(
        '--category', default=None, choices=CAT_ID2CATEGORY,
        help=(
            'Limit TUBELEX to only one category'
            )
        )
    parser.add_argument(
        '--wikipedia', nargs='*',
        default=[], help=(
            'Use Wikipedia for these language codes. Overrides --tubelex'
            )
        )
    parser.add_argument(
        '--gini', nargs='*',
        default=[], help=(
            'Use GINI for these language codes (en, ja). Overrides --wikipedia'
            )
        )
    parser.add_argument(
        '--subimdb', action='store_true', help='Use SubIMDB for English.'
        )
    parser.add_argument(
        '--spoken-bnc', action='store_true',
        help='Use spoken portion of BNC for English.'
        )
    parser.add_argument(
        '--espal', action='store_true', help='Use EsPal for Spanish.'
        )
    parser.add_argument(
        '--activ-es', action='store_true', help='Use ACTIV-ES for Spanish.'
        )
    parser.add_argument(
        '--alonso', '--spoken-crea', action='store_true', help=(
            'Use oral frequencies from CREA by Alonso et al. (2011) for Spanish.'
            )
        )
    parser.add_argument(
        '--csj', action='store_true', help=(
            'Use CSJ for Japanese (requires --form lemma).'
            )
        )
    parser.add_argument(
        '--laborotvspeech', action='store_true', help=(
            'Use LaboroTVSpeech for Japanese (requires --D ipadic).'
            )
        )
    parser.add_argument(
        '--hkust-mtsc', action='store_true', help='Use HKUST/MCTS for Chinese.'
        )
    parser.add_argument(
        '--minus', action='store_true', help='Use opposite value instead of log.'
        )
    parser.add_argument(
        '--minus-log', action='store_true', help='Use minus log instead of log.'
        )
    parser.add_argument('--models', default='experiments/models',
                        help='Model directory.')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument(
        '--log-lookups', help='Log frequency lookups to a file.'
        )
    add_tokenizer_arg_group(parser)  # calls add_tagger_arg_group(parser)
    parser.add_argument('--token', default=None,
                        help='Hugging Face access token')

    return parser.parse_args()


class StatData(NamedTuple):
    get: Callable[[str], Any]
    langs: list[str] | dict[str, str]

    def data(self) -> dict['str', Any]:
        langs = self.langs
        args_langs = (
            langs.items() if isinstance(langs, dict) else
            zip(map(lambda lang: (lang,), langs), langs)
            )
        return {lang: self.get(*args) for args, lang in args_langs}

    def types(corpus: Any) -> int:
        if isinstance(corpus, FrequencyData):
            return len(corpus.f)
        assert isinstance(corpus, (pd.Series, dict))    # any Series or a wordfreq dict
        return len(corpus)

    def tokens(corpus: Any) -> Optional[int]:
        if isinstance(corpus, FrequencyData):
            return corpus.f_total
        if isinstance(corpus, dict):        # wordfreq dict
            return None
        assert isinstance(corpus, pd.Series)
        if corpus.name == 'aes_orf':        # ACTIV-ES (Francom et al., 2014)
            return 3_897_234
        if corpus.name == 'GINI':           # GINI: unknown
            return None
        n = corpus.sum()
        assert n % 1 == 0
        return int(n)


ALL_LANGS = ['en', 'es', 'zh', 'id', 'ja']

STATS_CORPORA: dict[str, StatData] = {
    'BNC-Spoken':       StatData(
        get_bnc_spoken_written_freq_data,                       {(True, False): 'en'}),
    'CREA-Spoken':      StatData(get_alonso_freq_data,          {(): 'es'}),
    'CSJ':              StatData(get_csj_freq_data,             {(): 'ja'}),
    'HKUST/MTS':        StatData(get_hkust_mtsc_freq_data,      {(): 'zh'}),

    'ACTIV-ES':         StatData(get_activ_es_data,             {(): 'es'}),
    'EsPal':            StatData(get_espal_freq_data,           {(): 'es'}),
    'LaboroTV1+2':      StatData(get_laborotv_freq_data,       {(): 'ja'}),
    'OpenSubtitles':    StatData(get_opensubtitles_freq_data,   ALL_LANGS),
    'SUBTLEX':          StatData(get_sub_freq_data,             ['en', 'es', 'zh']),
    'SUBTLEX-UK':       StatData(get_sub_freq_data,             {('en-uk',): 'en'}),

    'GINI':             StatData(get_gini_data,                 ['en', 'ja']),
    'Wikipedia':        StatData(get_wiki_freq_data,            ALL_LANGS),
    'wordfreq':         StatData(wf.get_frequency_dict,         ALL_LANGS),

    'TUBELEX\\textsubscript{default}': StatData(get_tubelex_freq_data, ALL_LANGS)
    }

STATS_DATASETS: dict[str, StatData] = {
    'Lexical Decision Time':    StatData(get_ldt_data,      ['en', 'es', 'zh']),
    'Lexical Complexity':       StatData(get_mlsp_dataset,  ['en', 'es', 'ja']),
    'Word Familiarity':         StatData(get_familiarity_data, ALL_LANGS),
    'Word Familiarity (Alternative)':          StatData(get_familiarity_data, {
        ('en', True): 'en',
        ('es', False, False, True): 'es',
        ('ja', False, False, False, True): 'ja',
        })
    }


def do_stats(path_datasets: str, path_corpora: str) -> None:
    sizes = {}
    tokens = {}
    types = {}

    for path, name2stat_data, is_dataset in (
        (path_corpora, STATS_CORPORA, False),
        (path_datasets, STATS_DATASETS, True),
        ):
        for name, stat_data in tqdm(iterable=name2stat_data.items(), desc=path):
            ld = stat_data.data()
            if is_dataset:
                sizes[name] = {lang: len(data) for lang, data in ld.items()}
            else:
                tokens[name] = {
                    lang: StatData.tokens(data) for lang, data in ld.items()
                    }
                types[name] = {lang: StatData.types(data) for lang, data in ld.items()}

        if is_dataset:
            df = pd.DataFrame(sizes)
        else:
            corpora_order = list(STATS_CORPORA.keys())
            df = pd.concat({
                'tokens': pd.DataFrame(tokens),
                'types': pd.DataFrame(types),
                }, axis=1).swaplevel(axis=1).sort_index(
                    # Keep columns in `corpora_order`:
                    axis=1,
                    key=lambda cols: pd.Index(map(
                        lambda col: corpora_order.index(col) if (col in corpora_order)
                        else col,
                        cols
                        ))
                    )
        # languages as rows->columns, sort alphabetically:
        df = df.rename(
            lambda lang: LANG2FULL_NAME.get(lang, lang)
            ).sort_index().transpose()

        # prevent .0 floats
        df.to_csv(path, float_format='%d')


def main(args: argparse.Namespace) -> None:
    # Input data:
    mlsp_subsets = args.mlsp
    ldt_langs = args.ldt
    fam_langs = args.familiarity

    output_files = args.output_files
    model_dir = args.models
    cache_tubelex = args.cache_tubelex
    tubelex_cache_dir = args.tubelex_cache
    train = args.train
    correlation = args.correlation
    read_gold = train or correlation
    subimdb = args.subimdb
    hkust_mtsc = args.hkust_mtsc
    spoken_bnc = args.spoken_bnc
    espal = args.espal
    alonso = args.alonso
    csj = args.csj
    laborotvspeech = args.laborotvspeech
    f_lookups = None

    if args.stats:
        do_stats(path_datasets=args.stats_datasets, path_corpora=args.stats_corpora)
        return

    if args.log_lookups is not None:
        f_lookups = open(args.log_lookups, 'a')

    # Tokenization:
    tokenize_c_j = not args.no_c_j_tokenize

    input_sets = mlsp_subsets or ldt_langs or fam_langs

    if not output_files and not train:
        output_files = [
            'experiments/output.tsv' if (len(input_sets) == 1) else
            'experiments/output'
            ]

    if cache_tubelex:
        if not (args.tubelex and
                args.category is None and
                args.tokenization is None and
                args.dictionary is None and
                args.form == 'surface'
                ):
            raise Exception(
                f'Non-default arguments incompatible with --cache-tubelex:\n'
                f'--tubelex: {args.tubelex}\n'
                f'--category: {args.category}\n'
                f'--tokenization: {args.tokenization}\n'
                f'--dictionary: {args.dictionary}\n'
                f'--form: {args.form}'
                )
        os.makedirs(tubelex_cache_dir, exist_ok=True)

    if train:
        if output_files:
            raise Exception('Supplied --train with --output-files.')
        os.makedirs(model_dir, exist_ok=True)
    elif len(output_files) == 1 and len(input_sets) > 1:
        output_dir = output_files[0]
        os.makedirs(output_dir, exist_ok=True)
        if mlsp_subsets:
            output_files = [
                os.path.join(
                    output_dir,
                    re.sub(r'_unlabelled|_labels', '', subset) + '.tsv'
                    )
                for subset in mlsp_subsets
                ]
        else:
            assert ldt_langs or fam_langs
            assert correlation
            output_files = [
                os.path.join(output_dir, lang + '.tsv') for lang in input_sets
                ]

    assert (
        train or
        (len(input_sets) == len(output_files))
        ), (train, len(input_sets), len(output_files))

    lang2freq_data = {}
    lang2gini_func = {}
    activ_es_func = None

    subtlex = args.subtlex
    for lang in args.subtlex:
        lang2freq_data[lang] = get_sub_freq_data(lang)
    if args.subtlex_uk:
        assert 'en' not in lang2freq_data
        subtlex.append('en')
        lang2freq_data['en'] = get_sub_freq_data('en-uk')

    lang2corpus_specific_tokenizer = {}
    for lang in args.opensubtitles:
        lang2freq_data[lang] = get_opensubtitles_freq_data(lang)
        if tokenize_c_j and (lang in LANG2KYTEA_MODEL):
            lang2corpus_specific_tokenizer[lang] = get_kytea_tokenizer(lang)
    for lang in chain(args.wikipedia, subtlex, args.gini):
        # These corpora use simple regex tokenization (except for zh/ja)
        # OpenSubtitles, SubIMDB, BNC use something more advanced (similar to Stanza)
        if lang not in ('zh', 'ja'):
            lang2corpus_specific_tokenizer[lang] = get_re_split().split
    if args.activ_es:
        lang2corpus_specific_tokenizer['es'] = get_re_split().split

    assert not args.category or args.tubelex, '--category requires --tubelex'
    for lang in args.tubelex:
        lang2freq_data[lang] = get_tubelex_freq_data(
            lang,
            form=args.form, tokenization=args.tokenization,
            category=args.category
            )

    for lang in args.wikipedia:
        lang2freq_data[lang] = get_wiki_freq_data(lang)

    for lang in args.gini:
        lang2gini_func[lang] = get_gini_missing_func(lang)

    if subimdb:
        assert 'en' not in lang2freq_data
        lang2freq_data['en'] = get_subimdb_freq_data()
    if spoken_bnc:
        assert 'en' not in lang2freq_data
        lang2freq_data['en'] = get_bnc_spoken_written_freq_data(spoken=True)

    if espal:
        assert 'es' not in lang2freq_data
        lang2freq_data['es'] = get_espal_freq_data()
    if alonso:
        assert 'es' not in lang2freq_data
        lang2freq_data['es'] = get_alonso_freq_data()
    if args.activ_es:
        assert 'es' not in lang2freq_data
        activ_es_func = get_activ_es_freq_missing_func()

    if csj:
        assert 'ja' not in lang2freq_data
        lang2freq_data['ja'] = get_csj_freq_data(
            form=args.form, tokenization=args.tokenization
            )
    if laborotvspeech:
        assert 'ja' not in lang2freq_data
        lang2freq_data['ja'] = get_laborotv_freq_data(
            form=args.form, tokenization=args.tokenization, dictionary=args.dictionary
            )

    if hkust_mtsc:
        assert 'zh' not in lang2freq_data
        lang2freq_data['zh'] = get_hkust_mtsc_freq_data()

    all_langs = set(chain(
        subtlex, args.opensubtitles, args.tubelex,
        args.wikipedia, args.gini, args.wordfreq,
        ['en'] if (subimdb or spoken_bnc) else [],
        ['zh'] if (hkust_mtsc) else [],
        ['es'] if (espal or alonso or (activ_es_func is not None)) else [],
        ['ja'] if (csj or laborotvspeech) else [],
        ))
    lang2tokenize = {
        lang: get_tokenizers(
            lang=lang, tokenization=args.tokenization, full=True, args=args
            )[1] for lang in all_langs
        }

    def tokenize(s, lang):
        if (tokenize := lang2corpus_specific_tokenizer.get(lang)):
            return tokenize(s)
        if (
            ((lang in lang2freq_data) or (lang in lang2gini_func)) and
            (tokenize_c_j or lang not in {'ja', 'zh'})
            ):
            tokenize = lang2tokenize.get(lang)
            return (
                [s] if (tokenize is None) else tokenize(s)
                )
        # Wordfreq:
        return (
            [s] if (lang in {'ja', 'zh'} and not tokenize_c_j) else
            wf.tokenize(s, lang)
            )

    if args.wordfreq_retokenize:
        def wf_frequency_missing(w: str, lang: str, minimum: float = 0) -> float:
            f = wf.word_frequency(w, lang)
            return (f, False) if f else (minimum, True)
    else:
        lang2wordfreq_dict = {}

        def wf_frequency_missing(w: str, lang: str, minimum: float = 0) -> float:
            if (d := lang2wordfreq_dict.get(lang)) is None:
                d = wf.get_frequency_dict(lang)
                lang2wordfreq_dict[lang] = d
            f = d.get(w)
            return (f, False) if (f is not None) else (minimum, True)

    def frequency_missing(w: str, lang: str) -> [float, bool]:
        if (frequency_missing_func := (activ_es_func if lang == 'es' else
                                       lang2gini_func.get(lang))) is not None:
            return frequency_missing_func(w)
        if (freq_data := lang2freq_data.get(lang)) is not None:
            return freq_data.smooth_frequency_missing(w)
        # We cannot smooth frequencies from wordfreq, so we use minimum instead:
        return wf_frequency_missing(w, lang, minimum=get_wf_min_freq(lang))

    def agg_frequency_missing(s: str, lang: str) -> [float, bool]:
        s_norm = nfkc_lower(s)
        if f_lookups is not None:
            print(f'lookup\t{s}', file=f_lookups)
            print(f'normalized\t{s_norm}', file=f_lookups)
        s = s_norm
        is_gini = lang in lang2gini_func
        tokens = tokenize(s, lang)
        if f_lookups is not None:
            for i, t in enumerate(tokens):
                print(f'token_{i}\t{t}', file=f_lookups)
        fs, ms = (
            zip(*(frequency_missing(t, lang) for t in tokens)) if tokens else
            ((), ())
            )
        (f, m) = (
            # Accept empty tokens => return f1
            max(fs, default=np.inf) if is_gini else min(fs, default=0),
            any(ms)
            )

        # Without tokenization:
        f1, m1 = frequency_missing(s, lang)
        better_without_tokenization = ((f1 < f) if is_gini else (f1 > f))

        return ((f1, m1) if better_without_tokenization else (f, m))

    if correlation:
        print(
            'file\tlanguage\tcorrelation\tcorr_tubelex\t'
            'n\tn_missing\tcorr_without_missing'
            )
    elif not train and args.metrics:
        print('file\tlanguage\tPearson\'s r\tMAE\tMSE\tR2')

    for input_id, path_out in zip_longest(input_sets, output_files):
        lang2data_targets_freq_tubelex_missing_gold = defaultdict(
            lambda: ([], [], [], [], [])
            )
        if mlsp_subsets:
            try:
                dataset = get_mlsp_dataset(input_id, train=train, token=args.token)
            except Exception:
                raise Exception(
                    f'Cannot retrieve dataset. Check the above exception, that you '
                    f'have requested access to {DATASET_NAME}, and that you are logged '
                    f'in with the correct token using `huggingface-cli login`. '
                    f'Alternatively you can supply the access token directly using '
                    f'the --token option.'
                    )

            for instance in dataset:
                if read_gold:
                    assert len(instance) == 5, (
                        f'Unexpected #fields: {len(instance)}'
                        )
                else:
                    assert 4 <= len(instance) <= 5, (
                        f'Unexpected #fields: {len(instance)}'
                        )
                lang, idx_str = instance['id'].split('_', 1)
                assert 2 <= len(lang) <= 3, f'Unexpected language: {lang}'
                (
                    data, targets, frequencies, missing, gold
                    ) = lang2data_targets_freq_tubelex_missing_gold[lang]
                t = instance['target']
                if len(instance) == 5:
                    g = instance['complexity']
                    assert isinstance(g, float)
                f, miss = agg_frequency_missing(t, lang)
                data.append([str(v) for v in instance.values()])
                targets.append(t)
                frequencies.append(f)
                missing.append(miss)
                if len(instance) == 5:
                    gold.append(g)
        else:
            assert ldt_langs or fam_langs
            lang = input_id
            dataset = (
                get_ldt_data(lang, prefer_zscore=args.zscore) if ldt_langs else
                get_familiarity_data(
                    lang,
                    glasgow=args.glasgow,
                    clark_paivio=args.clark_paivio,
                    moreno_martinez=args.moreno_martinez,
                    amano=args.amano
                    )
                )
            (
                data, targets, frequencies, missing, gold
                ) = lang2data_targets_freq_tubelex_missing_gold[lang]
            for word, ldt in dataset.items():
                f, miss = agg_frequency_missing(word, lang)
                data.append([word])
                targets.append(word)
                frequencies.append(f)
                assert correlation
                missing.append(miss)
                gold.append(ldt)

        if not train:
            parent, __ = os.path.split(path_out)
            if parent:  # may be ''
                os.makedirs(parent, exist_ok=True)
            fo = open(path_out, 'w')
        else:
            fo = None

        for lang, (
            data, targets, frequencies, missing, gold
            ) in lang2data_targets_freq_tubelex_missing_gold.items():
            miss_t = [t for t, m in zip(targets, missing) if m]
            if args.verbose:
                print(
                    f'{lang}: #={len(targets)}, '
                    f'missing frequency #={len(miss_t)}: {miss_t}'
                    )
            assert all(frequencies)  # all should be non-zero
            f = np.array(frequencies)
            f_valid = ~np.array(missing, dtype=bool)
            logf = (
                (-f) if args.minus else
                (-np.log10(f)) if args.minus_log else
                np.log10(f)
                )
            c = np.array(gold) if gold else None

            if correlation:
                assert c is not None

                cache_name = f'tubelex-{lang}-' + (
                    'mlsp' if mlsp_subsets else
                    'ldt' if ldt_langs else
                    'fam')
                if ldt_langs:
                    if args.zscore:
                        cache_name += '.zscore'
                elif not mlsp_subsets:
                    if args.glasgow:
                        cache_name += '.glasgow'
                    if args.clark_paivio:
                        cache_name += '.clark_paivio'
                    if args.moreno_martinez:
                        cache_name += '.moreno_martinez'
                    if args.amano:
                        cache_name += '.amano'
                cache_path = os.path.join(tubelex_cache_dir, cache_name + '.npy')

                if cache_tubelex:
                    np.save(cache_path, logf, allow_pickle=False)
                    logf_tubelex = logf
                else:
                    if not os.path.exists(cache_path):
                        print(
                            f'Warning: No cached TUBELEX frequencies at {cache_path}. '
                            f'Will use NA values to compare with instead.',
                            file=sys.stderr
                            )
                        logf_tubelex = np.full_like(logf, np.nan)
                    else:
                        logf_tubelex = np.load(cache_path, allow_pickle=False)
                        if len(logf_tubelex) != len(logf):
                            print(
                                f'Warning: Cached TUBELEX frequencies at {cache_path} '
                                f'differ in length {len(logf_tubelex)} != {len(logf)}. '
                                f'Will use NA values to compare with instead.',
                                file=sys.stderr
                                )
                            logf_tubelex = np.full_like(logf, np.nan)

                r = pearson_r(logf, c)
                r_tubelex = pearson_r(logf, logf_tubelex)
                n = len(logf)
                n_missing = sum(missing)
                r_valid = pearson_r(logf[f_valid], c[f_valid])
                print(
                    f'{input_id}\t{LANG2FULL_NAME[lang]}\t{r}\t{r_tubelex}\t'
                    f'{n}\t{n_missing}\t{r_valid}'
                    )
                for fields, p in zip(data, logf):
                    print('\t'.join((
                        *fields[:4], str(p)
                        )), file=fo)
            elif train:
                assert c is not None
                linear_regression = RidgeCV().fit(
                    logf.reshape(-1, 1), c
                    )
                model_path = os.path.join(model_dir, f'{lang}.model')
                dump(linear_regression, model_path)
            else:
                model_path = os.path.join(model_dir, f'{lang}.model')
                linear_regression = load(model_path)
                pred    = linear_regression.predict(logf.reshape(-1, 1))
                pred    = np.minimum(1.0, np.maximum(0.0, pred))    # clip to 0...1
                if args.metrics:
                    if c is None:
                        raise Exception(
                            'Missing gold labels, cannot compute metrics.'
                            )
                    r = pearson_r(c, pred)
                    mae = mae_score(c, pred)
                    mse = mse_score(c, pred)
                    r2 = r2_score(c, pred)
                    print(
                        f'{input_id}\t{LANG2FULL_NAME[lang]}\t{r}\t{mae}\t'
                        f'{mse}\t{r2}'
                        )
                for fields, p in zip(data, pred):
                    print('\t'.join((
                        *fields[:4], str(p)
                        )), file=fo)

        if fo is not None:
            fo.close()
    if f_lookups:
        f_lookups.close()


if __name__ == '__main__':
    main(parse_args())
