import re
import os
import sys
from collections.abc import Iterator, Iterable, Callable
from typing import Optional
from collections import Counter
from collections.abc import Container
from urllib.request import urlretrieve
from contextlib import contextmanager
from zipfile import ZipFile
from itertools import chain, groupby, compress, islice
import argparse
from os.path import splitext
from tqdm import tqdm  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.metrics.pairwise import linear_kernel  # type: ignore
import fasttext  # type: ignore
from lang_utils import (
    get_re_word_relaxed, get_re_split,
    add_tagger_arg_group, tagger_from_args,
    OPT_BASE_LEMMA_READING_POS, POSTagger,
    sub_smart_apos, repl_smart_apos,
    NORMALIZE_FULLWIDTH_TILDE
    )
from freq_utils import Storage, WordCounterGroup
from vtt import VTTCleaner, sub_space
from unicodedata import normalize as unicode_normalize
import pysbd

# We use the smaller model from
# https://fasttext.cc/docs/en/language-identification.html
FT_LID_MODEL_PATH = 'lid.176.ftz'
FT_LID_MODEL_URL = (
    'https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz'
    )


def make_lang_label(lang: str) -> list[str]:
    return [f'__label__{lang}']


def nfkc_lower(s: str) -> str:
    return unicode_normalize('NFKC', s).lower()


MIN_LANG_FREQ = 0.95
SIMILARITY_LIMIT = 0.95
MIN_LANG_CHARS_FREQ = 0.7
MIN_NONEMPTY_LINES = 3
SUB_SUFFIX = '.vtt'     # input
DATA_SUFFIX = '.txt'    # intermediate data
CLEAN_PATH_FMT = 'corpus/clean-%s'
UNIQUE_PATH_FMT = 'corpus/unique-%s'

# Defaults for CLI arguments:
DATA_PATH_FMT = 'jtubespeech-subtitles/video/%s/vtt'
SUBLIST_PATH_FMT = 'jtubespeech-subtitles/sub/%s/%s_sample.csv'
DEFAULT_FREQ_PATH_FMT = 'tubelex-%s%%.tsv'
DEFAULT_TOK_PATH_FMT = 'corpus/tokenized-%s.txt'
DEFAULT_CHANNEL_STATS_PATH_FMT = 'tubelex-%s-channels.tsv'
DEFAULT_REM_ADDR_PATH_FMT = 'tubelex-%s-removed-addresses.tsv'
DEFAULT_MIN_VIDEOS = 0
DEFAULT_MIN_CHANNELS = 0

DEFAULT_STANZA_FORCE_SPLIT_LEN = 1024
LINEAR_KERNEL_CHUNK_SIZE = 10000
Tokenizer = Callable[[str], list[str]]
TokenizerTagger = Callable[[str], list[tuple[str, str]]]
WordFilter = Callable[[str], bool]

# The functions subn_yt_censored, subn_cc_desc and substitute patterns including
# any surrounding spaces. They are meant to be used like this:
# subn_*(' ', s)

YT_CENSORED             = '[ __ ]'
YT_CENSORED_POS         = 'CENSORED'
NA_POS                  = 'N/A'
CC_PLACEHOLDER  = 'CCTUBELEXPLACEHOLDER'
CC_PLACEHOLDER_SPC  = ' CCTUBELEXPLACEHOLDER '
subn_yt_censored        = re.compile(rf'\s+{re.escape(YT_CENSORED)}\s+').subn
subn_cc_placeholder     = re.compile(rf'\b{CC_PLACEHOLDER}\b').subn

RE_CC_DESC      = r'\[(?=[^\[\]]*[\w♪～])[^\[\].?!．。？！]+\]'
RE_CC_DESC_FW   = r'【(?=[^【】]*[\w♪～])[^【】.?!．。？！]+】'
PAT_CC_DESC     = re.compile(
    # May be either:
    # - "[X]" or "【X】", where X doesn't contain the respective type of brackets,
    #   and contains at least one word-forming character, "♪" or "～".
    rf'\s*({re.escape(YT_CENSORED)}|{RE_CC_DESC}|{RE_CC_DESC_FW})\s*'
    )
subn_cc_desc = PAT_CC_DESC.subn

NORMALIZE_CC: dict[int, int] = {
    0x20: 0x005F,  # space to underscore
    0x09: 0x005F,  # tab to underscore
    # 【】 to []:
    0x3010: 0x005B,
    0x3011: 0x005D
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


# Assume that [.?!] followed by a whitespace (or end of string) or two newlines
# split a sentence, as they typically do in languages using Western punctuation
# in Stanza:

PAT_STANZA_SENT_SPLIT = re.compile(r'[.?!](\s|$)|\n\n|$')


def force_split_iter(s: str, max_len: int) -> Iterator[str]:
    r'''
    Force split substrings longer than `max_len` not containing PAT_STANZA_SENT_SPLIT.

    >>> list(force_split_iter('XXXyyyZZZ', 3))          # no spaces and linebreaks
    ['XXX', 'yyy', 'ZZZ']
    >>> list(force_split_iter('XXX yyy\nZ', 3))         # spaces and linebreaks
    ['XXX', 'yyy', 'Z']
    >>> list(force_split_iter('XX\nyy zz\nUUUw', 3))    # mix and shorter sequences
    ['XX', 'yy', 'zz', 'UUU', 'w']
    >>> list(force_split_iter('XX\n\nyy\n\nzz\n\nUUU. w', 3))   # has sentence splits
    ['XX\n\nyy\n\nzz\n\nUUU. w']
    >>> list(force_split_iter('UUU.', 3))           # trailing dot (not followed by \s)
    ['UUU.']
    '''
    prev_ss = 0
    prev_force = 0

    for m in PAT_STANZA_SENT_SPLIT.finditer(s):
        ss_start, ss_end = m.span()
        while ss_start - prev_ss > max_len:
            # Span between two splits too long:
            must_split = s[prev_ss:prev_ss + max_len + 1]

            # Find the last possible place to force split (ideally a newline):
            force_pos = must_split.rfind('\n')
            force_len = 1
            if force_pos == -1:
                force_pos = must_split.rfind(' ')
                if force_pos == -1:
                    force_pos = max_len
                    force_len = 0
            force_start = prev_ss + force_pos       # index in s

            # Split:
            yield s[prev_force:force_start]
            prev_force  = force_start + force_len   # do not include the newline/space
            prev_ss     = prev_force
        prev_ss = ss_end

    yield s[prev_force:]


def force_split(s: str, max_len: int) -> str:
    return '\n\n'.join(force_split_iter(s, max_len))


def stanza_sentences_workaround(
    nlp  # : stanza.Pipeline
    ):   # -> Callable[[str], Iterable['stanza.models.common.doc.Sentence']]
    '''
    Workaround for this bug: https://github.com/stanfordnlp/stanza/issues/1401
    '''
    def nlp_sentences(s: str):
        try:
            sentences = nlp(s).sentences    # list
            # Sometimes inexplicably results in:
            #
            # RuntimeError: Length of all samples has to be greater than 0, but found
            # an element in 'lengths' that is <= 0
        except RuntimeError:
            # Workaround:
            chunks = s.split('\n\n')
            try:
                sentences = chain.from_iterable(
                    map(lambda chunk: nlp(chunk).sentences, chunks)
                    )
            except RuntimeError as e:
                sys.stderr.write(
                    f'Irrecoverable RuntimeError in NLP pipeline:\n\n'
                    f's = "{s}"\n\n'
                    f'chunks = {chunks}\n\n'
                    )
                raise e
        return sentences    # list or iterator
    return nlp_sentences


def linear_kernel_piecewise(x, max_size=LINEAR_KERNEL_CHUNK_SIZE, wrapper=None):
    # To support sparse matrices we split manually (not using np.array_split).
    # Note: Row slicing is efficient for CSR (Compressed Sparse Row).
    pieces = [x[i:i + max_size] for i in range(0, x.shape[0], max_size)]
    if wrapper:
        pieces = wrapper(pieces)
    return np.concatenate([linear_kernel(p, x) for p in pieces])


def add_tokenizer_arg_group(
    parser: argparse.ArgumentParser,
    unique_tokenization: bool = False,
    title='Tokenization options'
    ):
    group = parser.add_argument_group(title=title)
    group.add_argument(
        '--tokenization', choices=['regex', 'treebank'], default=None, help=(
            'Use other tokenization for frequency counting than the default Stanza'
            '(StanfordNLP) for languages other than Chinese or Japanese. '
            'Treebank requires English.'
            )
        )
    if unique_tokenization:
        group.add_argument(
            '--unique-tokenization', choices=['regex', 'stanza', 'treebank'],
            default=None, help=(
                'Use other tokenization for deduplication (--uniqe) than the default '
                'treebank for English and regex for languages other than Chinese or '
                'Japanese. Treebank requires English.'
                )
            )
    group.add_argument(
        '--tokenized-files', type=str, default=None,
        help=(
            'Compute frequencies from tokenized files in a specified directory'
            '(e.g. SubIMDB; see README; requires --frequencies and --output).'
            )
        )
    group.add_argument(
        '--pos', '-P', action='store_true',
        help='Add most common POS to frequencies'
        )
    group.add_argument(
        '--extended-pos', '-X', action='store_true',
        help=(
            'Tag compound verbs, compound particles, auxiliaries, mimetic words in '
            'Japanese.'
            )
        )
    group.add_argument(
        '--form', choices=['surface', 'base', 'lemma'], default='surface',
        help=(
            'Recorded word form: surface, base form, or lemma '
            '(base form in standard orthography). Affects --unique and --frequencies.'
            )
        )
    group.add_argument(
        '--stanza-force-split', type=int,
        dest='split_len', default=DEFAULT_STANZA_FORCE_SPLIT_LEN,
        help=(
            'When using Stanza tokenization, force split sequences without sentence '
            'boundaries longer than SPLIT_LEN. '
            '(Default: {DEFAULT_STANZA_FORCE_SPLIT_LEN}.)'
            )
        )
    group.add_argument(
        '--no-smart-apostrophe', action='store_false', dest='smart_apostrophe', help=(
            'Do not heuristically convert right single quote "’" to apostrophe "\'" '
            'before English tokenization.'
            )
        )
    group.add_argument(
        '--no-filter-tokens', action='store_false', dest='filter_tokens', help=(
            'Do not filter tokens.'
            )
        )
    add_tagger_arg_group(group)


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(
        'Process subtitles from jtubespeech-subtitles or jtubespeech and '
        'compute word frequencies'
        ))

    parser.add_argument(
        '--tf-idf-ngram-min', dest='nmin', type=int, default=1,
        help='Consider n-grams where n>=NMIN for TF-IDF.'
        )
    parser.add_argument(
        '--tf-idf-ngram-max', dest='nmax', type=int, default=1,
        help='Consider n-grams where n<=NMAX for TF-IDF.'
        )

    parser.add_argument(
        '--language', type=str, default='ja',
        help='Language (2-letter code, default: ja).'
        )
    parser.add_argument(
        '--identifier', type=str, default=None,
        help='Identifier (instead of language and category) for corpus files.'
        )

    add_tokenizer_arg_group(parser, unique_tokenization=True)

    parser.add_argument(
        '--laborotvspeech', action='store_true',
        help=(
            'Tokenized files are LaboroTVSpeech (see README).'
            )
        )

    parser.add_argument(
        '--no-filter-cc-descriptions', action='store_false',
        dest='filter_cc_descriptions', help=(
            'Do not filter CC descriptions in brackets (e.g. "[Music]", "[Applause]").'
            )
        )
    parser.add_argument(
        '--data', type=str, default=None,
        help='Data path (default: jtubespeech-subtitles, based on language).'
        )
    parser.add_argument(
        '--list', type=str, default=None,
        help='List path (default: jtubespeech-subtitles sample, based on language).'
        )
    parser.add_argument(
        '--limit-categories', '-C', nargs='+', choices=CAT_ID2CATEGORY, default=None,
        help='Limit the list to certain categories (default: all)'
        )
    parser.add_argument(
        '--limit', type=int, default=None,
        help='Maximum number of videos to process (for testing purposes)'
        )

    parser.add_argument('--clean', '-c', action='store_true', help='Clean up data')
    parser.add_argument('--unique', '-u', action='store_true', help='Deduplicate data')
    final_step = parser.add_mutually_exclusive_group()
    final_step.add_argument(
        '--frequencies', '-f', action='store_true',
        help='Compute frequencies'
        )
    final_step.add_argument(
        '--tokenize', '-t', action='store_true',
        help=(
            'Tokenize (create data suitable for training embeddings instead of '
            'counting frequencies).'
            )
        )
    parser.add_argument(
        '--no-categories', action='store_false', dest='categories',
        help='Do not add video categories'
        )

    parser.add_argument(
        '--min-videos', type=int, default=DEFAULT_MIN_VIDEOS, help=(
            f'Minimum videos for the word to be counted (default: {DEFAULT_MIN_VIDEOS})'
            )
        )
    parser.add_argument(
        '--min-channels', type=int, default=DEFAULT_MIN_CHANNELS, help=(
            f'Minimum channels for the word to be counted (default: '
            f'{DEFAULT_MIN_CHANNELS})'
            )
        )

    Storage.add_arg_group(parser, 'Compression options', zip_suffix=True)

    parser.add_argument(
        '--output', '-o', type=str, default=None,
        help=(
            'Output filename for frequencies. If the placeholder "%%" is present, it '
            'is replaced with a string identifying the normalization. Otherwise, '
            'output only unnormalized data.'
            )
        )
    parser.add_argument(
        '--channel-stats', type=str, default=None,
        help='Output filename for channel stats (computed together with frequencies)'
        )
    parser.add_argument(
        '--removed-addresses', type=str, default=None,
        help='Output filename for removed addresses (during cleanup)'
        )
    parser.add_argument(
        '--verbose', '-v', action='store_true', help=(
            'Print verbose per-file information for cleanup. '
            'Print tokens when computing frequencies.'
            )
        )

    return parser.parse_args()


@contextmanager
def get_write_file(path: str, storage: Storage) -> Iterator[Callable[[str, str], None]]:
    try:
        write_file: Callable[[str, str], None]
        if storage.zip_compression is not None:
            zf = ZipFile(path + '.zip', 'w', storage.zip_compression)
            write_file = zf.writestr    # type: ignore
        else:
            os.makedirs(path, exist_ok=True)

            def write_file(fname: str, contents: str) -> None:
                with open(os.path.join(path, fname), 'w') as f:
                    f.write(contents)

        yield write_file
    finally:
        if storage.zip_compression is not None:
            zf.close()


PAT_ENGLISH_SEQ = re.compile(
    r'[\u0020-\u007E]+'     # Printable ASCII
    )
PAT_LATIN_SEQ = re.compile(
    r'[\u0020-\u007E'       # Printable ASCII
    r'\u00A0-\u024F'        # Latin-1, Latin Extended-A, Latin Extended-B
    r'\u0300-\u030F]+'      # Some(!) combining diacritics
    )
PAT_CHINESE_SEQ = re.compile(
    r'[ '               # ASCII space (we replace all white space with ASCII space)
    r'\u3400-\u4DB5\u4E00-\u9FCB\uF900-\uFA6A'  # {Han}
    r'\u2E80-\u2FD5'    # Han radicals
    r'\u3000-\u303F'    # CJK Symbols and Punctuation
    r'\uFF01-\uFF5E]+'  # FW Alphanumeric and Punctuation (basically FW ASCII)
    )
PAT_JAPANESE_SEQ = re.compile(
    # Based on http://www.localizingjapan.com/blog/2012/01/20/regular-expressions-for-\
    # japanese-text/
    # We give it a little benefit of doubt by including all kanji and even radicals,
    # which still could be indicative of Japanese or Chinese text.
    r'[ '               # ASCII space (we replace all white space with ASCII space)
    r'\u3041-\u3096'    # Hiragana
    r'\u30A0-\u30FF'    # Katakana (full-width)
    r'\u3400-\u4DB5\u4E00-\u9FCB\uF900-\uFA6A'  # {Han} (Kanji incl. Chinese-only)
    r'\u2E80-\u2FD5'    # Han/Kanji radicals
    r'\uFF5F-\uFF9F'    # HW Katakana and punctutation
    r'\u3000-\u303F'    # CJK Symbols and Punctuation
    r'\u31F0-\u31FF\u3220-\u3243\u3280-\u337F'  # Misc. Japanese Symbols and Characters
    r'\uFF01-\uFF5E]+'  # FW Alphanumeric and Punctuation (basically FW ASCII)
    )
LANG2PAT_SEQ = {
    'en': PAT_ENGLISH_SEQ,
    'zh': PAT_CHINESE_SEQ,
    'ja': PAT_JAPANESE_SEQ
    }
PAT_INVALID_TAG = re.compile(
    # invalid but frequent HTML tags (would have been escaped as &lt;/&gt; in VTT)
    r'<(font|FONT|[biu])( [^>]*)?>|'
    r'</(font|FONT|[biu])>|'
    # in some Udacity videos:
    r'\[br\]'
    )
invalid_tag_subn = PAT_INVALID_TAG.subn


def invalid_tag_subn_verbose(repl: str, s: str) -> tuple[int, str]:
    t, n = invalid_tag_subn(repl, s)
    if n:
        print(f'INVALID TAGS ({n}): {s}')
    return (t, n)


# HTTP(S)URLs and INFORMAL WEB ADDRESSES
#
# Accept most IDNA domain names/internationalized URL:

# 1. Common (not all) TLDs:
# Note: We accept either all upper case or all lower case, to avoid false positives,
# e.g. "sentence ends.It starts another sentence"
RE_TLD = (
    r'(?:'
    # common longer/non-country TLDs:
    r'com|net|org|info|xyz|biz|online|club|pro|'
    r'site|shop|store|tech|dev|gov|edu|mil|org|'
    r'COM|NET|ORG|INFO|XYZ|BIZ|ONLINE|CLUB|PRO|'
    r'SITE|SHOP|STORE|TECH|DEV|GOV|EDU|MIL|ORG|'
    # common 2-letter/country TLDs:
    r'ac|ad|ae|af|ag|ai|al|am|ao|ar|as|at|au|az|ba|bd|be|bf|bg|bh|'
    r'bn|bo|br|bt|bw|by|bz|ca|cc|cd|cf|ch|ci|cl|cm|cn|co|cr|cu|cx|'
    r'cy|cz|de|dk|do|dz|ec|ee|eg|es|et|eu|fi|fj|fm|fr|ga|ge|gg|gh|'
    r'gl|gq|gr|gs|gt|hk|hn|hr|hu|id|ie|il|im|in|io|iq|ir|is|it|jm|'
    r'jo|jp|ke|kg|kh|ki|kr|kw|ky|kz|la|lb|li|lk|lt|lu|lv|ly|ma|md|'
    r'me|mg|mk|ml|mm|mn|mo|mr|ms|mt|mu|mv|mx|my|mz|na|nc|nf|ng|ni|'
    r'nl|no|np|nu|nz|om|pa|pe|pf|ph|pk|pl|pm|ps|pt|pw|py|qa|re|ro|'
    r'rs|ru|rw|sa|sc|sd|se|sg|sh|si|sk|sn|so|st|su|sv|sx|sy|tc|th|'
    r'tj|tk|tm|tn|to|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|vc|ve|vg|vn|'
    r'ws|za|zm|zw|'
    r'AC|AD|AE|AF|AG|AI|AL|AM|AO|AR|AS|AT|AU|AZ|BA|BD|BE|BF|BG|BH|'
    r'BN|BO|BR|BT|BW|BY|BZ|CA|CC|CD|CF|CH|CI|CL|CM|CN|CO|CR|CU|CX|'
    r'CY|CZ|DE|DK|DO|DZ|EC|EE|EG|ES|ET|EU|FI|FJ|FM|FR|GA|GE|GG|GH|'
    r'GL|GQ|GR|GS|GT|HK|HN|HR|HU|ID|IE|IL|IM|IN|IO|IQ|IR|IS|IT|JM|'
    r'JO|JP|KE|KG|KH|KI|KR|KW|KY|KZ|LA|LB|LI|LK|LT|LU|LV|LY|MA|MD|'
    r'ME|MG|MK|ML|MM|MN|MO|MR|MS|MT|MU|MV|MX|MY|MZ|NA|NC|NF|NG|NI|'
    r'NL|NO|NP|NU|NZ|OM|PA|PE|PF|PH|PK|PL|PM|PS|PT|PW|PY|QA|RE|RO|'
    r'RS|RU|RW|SA|SC|SD|SE|SG|SH|SI|SK|SN|SO|ST|SU|SV|SX|SY|TC|TH|'
    r'TJ|TK|TM|TN|TO|TR|TT|TV|TW|TZ|UA|UG|UK|US|UY|UZ|VC|VE|VG|VN|'
    r'WS|ZA|ZM|ZW'
    # ensure it's not followed by other \w characters:
    r')\b'
    )

# 2. Host names (without TLD):
# - can contain letter, digit, hyphen (not underscore),
# - cannot start or end with a hyphen
# \w (perfectly?) matches characters allowed by IDNA allowing different alphabets,
# accents, digits (\W is the negation of \w)

RE_HOST = r'(?!-)(?:[^\W_]|-)+(?<!-)'

# 3. Domain name + optional port (no underscores, although allowed in subdomains):
# e.g. apple.com, www.google.com, go.to, WITH PORT: én.wik-pédi4.com:8080

RE_DOMAIN = rf'(?:{RE_HOST}\.)+{RE_TLD}'
RE_DOMAIN_PORT = rf'{RE_DOMAIN}(?::[0-9]+)?'

# 4. Simplified (int'l) paths:
# - Do not allow "'()[]", although legal in URL.
# - Do not allow trailing punctuation .!?,;:, although legal in URL.

RE_PATH = (
    r'/(?:'                     # leading slash followed by any number of
    r'[\w/!*;:@&=+$,/?#.~-]|'   # - allowed characters allowing int'l (\w)
    r'(?:%[0-9a-fA-F]{2})'      # - hex escapes
    r')*'
    r'(?<![.!?,;:])'
    )

# 5. URL: HTTP(S) URL with optional port, or URL without protocol and port,
# e.g.
# https://x.com/username,
# x.com/username
# https://www.google.com:443/search?q=term

RE_URL = (
    rf'(?:https?://{RE_DOMAIN_PORT}|{RE_DOMAIN})(?:{RE_PATH})?'
    )

RE_EMAIL = rf'[a-zA-Z0-9_.+-]+@{RE_DOMAIN}'

# Now handled by RE_URL:
# RE_WWW = r'www.[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'

# Social network handles generally don't start with . and don't contain consecutive .
RE_HANDLE = r'@(?:[a-zA-Z0-9_.]+)*[a-zA-Z0-9_]+'
PAT_ADDRESS = re.compile(
    r'|'.join((RE_URL, RE_EMAIL, RE_HANDLE))
    )

PAT_NEWLINES = re.compile(r'\n+')


def repl_newlines_for_tokenization(m: re.Match) -> str:
    return (' ' if (len(m.group()) == 1) else '\n')


def convert_newlines_for_tokenization(s: str) -> str:
    return PAT_NEWLINES.sub(repl_newlines_for_tokenization, s)


class SubCleaner(VTTCleaner):
    pat_seq: re.Pattern
    n_removed_tags: int
    n_removed_addresses: int
    n_lines_empty: int
    n_lines_nonl: int
    n_chars: int
    n_chars_l: int
    removed_addresses: list[str]

    '''
    Clean and filter lines via the __call__ method. Counts (instance variables) reflect
    the last call (non-cumulative). Counts are not set until the iterator is exhausted.
    '''

    def __init__(self, lang: str, verbose: bool = False):
        # TODO: We assume (extended) Latin alphabet for languages other than en/zh/ja
        super().__init__(verbose=verbose)
        self.pat_seq = LANG2PAT_SEQ.get(lang, PAT_LATIN_SEQ)

    def _repl_address(self, m: re.Match) -> str:
        self.removed_addresses.append(m.group())
        return ''

    def _extra_cleanup(
        self, lines: Iterable[str], delays_as_empty: bool = True
        ) -> Iterator[str]:
        n_removed_tags          = 0
        n_removed_addresses     = 0
        n_lines_empty           = 0
        self.removed_addresses  = []
        tag_subn = (
            invalid_tag_subn_verbose if self.verbose else
            invalid_tag_subn
            )
        for line in lines:
            if delays_as_empty and not line:
                # Preserve empty lines representing delays
                yield line
                continue
            # Remove formatting tags:
            line, n_tags = tag_subn('', line)
            # Replace addresses with space for better tokenization:
            line, n_addresses = PAT_ADDRESS.subn(self._repl_address, line)
            # Re-normalize whitespace:
            line = sub_space(' ', line)
            n_removed_tags += n_tags
            n_removed_addresses += n_addresses
            if not line or line == ' ':
                n_lines_empty += 1
                continue            # ignore empty line
            yield line
        self.n_removed_tags = n_removed_tags
        self.n_removed_addresses = n_removed_addresses
        self.n_lines_empty = n_lines_empty

    def __call__(
        self, lines: Iterable[str], delays_as_empty: bool = True
        ) -> Iterator[str]:
        # Reset the counts, and set them only if the iterator is exhausted
        # (safety measure)
        if hasattr(self, 'n_removed_tags'):
            del self.n_removed_tags
            del self.n_removed_addresses
            del self.n_lines_empty
            del self.n_lines_nonl
            del self.n_chars
            del self.n_chars_l
            del self.removed_addresses

        n_lines_nonl = 0
        n_chars = 0
        n_chars_l = 0

        pat_seq_sub = self.pat_seq.sub    # for speed
        for line in self._extra_cleanup(
            super().__call__(lines, delays_as_empty=delays_as_empty),
            delays_as_empty=delays_as_empty
            ):
            if delays_as_empty and not line:
                # Preserve empty lines representing delays
                yield line
                continue
            line_n_chars  = len(line)
            # The fastest way to count (non-)`lang` characters in a mostly
            # `lang` text is to remove sequences of `lang` characters using
            # a (compiled) regex:
            line_n_chars_l = line_n_chars - len(pat_seq_sub('', line))
            if not line_n_chars_l:
                n_lines_nonl += 1
                continue
            n_chars += line_n_chars
            n_chars_l += line_n_chars_l
            yield line

        self.n_lines_nonl   = n_lines_nonl
        self.n_chars        = n_chars
        self.n_chars_l      = n_chars_l

    @property
    def lang_char_frequency_in_filtered_lines(self) -> float:
        n = self.n_chars
        return (self.n_chars_l / n) if n else 0


def dir_files(
    path: str,
    suffix: Optional[str] = DATA_SUFFIX
    ) -> Iterator[tuple[str, str]]:
    for root, _subdirs, files in os.walk(path):
        for file in files:
            if (
                suffix is None or
                file.endswith(suffix)
                ):
                yield root, file


def filter_dir_files(
    dfs: Iterator[tuple[str, str]], videoids: Container[str]
    ) -> Iterator[tuple[str, str]]:
    return filter(
        lambda df: splitext(df[1])[0] in videoids,
        dfs
        )


@contextmanager
def get_files_contents(
    path: str, storage: Storage, any_suffix: bool = False,
    filenames: Optional[Iterable[str]] = None
    ):
    zf = None
    try:
        if storage.zip_compression is not None:
            zf = ZipFile(path + '.zip', 'r')
            files = (
                filenames if (filenames is not None) else
                zf.namelist()
                )

            def iter_contents(max_files=None):
                for file in files[:max_files]:
                    yield zf.read(file).decode('utf-8')

        elif filenames:
            files = filenames

            def iter_contents(max_files=None):
                for file in files[:max_files]:
                    with open(os.path.join(path, file)) as f:
                        yield f.read()
        else:
            dfs = list(dir_files(
                path,
                suffix=(None if any_suffix else DATA_SUFFIX)
                ))
            files = [f for _d, f in dfs]

            def iter_contents(max_files=None):
                for directory, file in dfs[:max_files]:
                    with open(os.path.join(directory, file)) as f:
                        yield f.read()

        yield (files, iter_contents)
    finally:
        if zf is not None:
            zf.close()


def do_clean(
    lang: str,
    identifier: str,
    storage: Storage,
    sublist: pd.DataFrame,  # for filtering files
    data_path: str,
    removed_addresses_path: Optional[str],
    limit: Optional[int],
    verbose: bool
    ) -> None:

    if not os.path.exists(FT_LID_MODEL_PATH):
        urlretrieve(FT_LID_MODEL_URL, filename=FT_LID_MODEL_PATH)

    lid_model = fasttext.load_model(FT_LID_MODEL_PATH)
    lang_label = make_lang_label(lang)

    videoids = set(sublist.index)  # Faster than index
    dfs = list(
        filter_dir_files(dir_files(data_path, suffix=SUB_SUFFIX), videoids)
        )[:limit]

    n_list  = len(sublist)
    n_total = len(dfs)
    n_short = 0
    n_nonlc = 0
    n_lid   = 0
    n_valid = 0
    n_valid_removed_tags = 0
    n_valid_removed_addresses = 0
    n_valid_lines_empty = 0
    n_valid_lines_nonl = 0
    n_valid_lines_valid = 0
    vtt_n_cues = 0
    vtt_n_empty = 0
    vtt_n_repeat = 0
    vtt_n_scroll = 0
    removed_addresses = []
    removed_addresses_videoids = []

    cleaner = SubCleaner(lang, verbose=verbose)

    with get_write_file(CLEAN_PATH_FMT % identifier, storage) as write_file:
        for directory, file in tqdm(
            desc='Cleaning',
            iterable=dfs
            ):
            videoid = splitext(file)[0]
            # clean each line, and remove repeated or empty lines
            with open(os.path.join(directory, file)) as f:
                lines = list(cleaner(f))
            if verbose:
                print(file)
                print(cleaner)

            n_lines = len(lines)

            # Exclude short files based non-empty lines:
            min_or_less_lines = len(list(
                islice(filter(None, lines), 0, MIN_NONEMPTY_LINES)
                ))
            if min_or_less_lines < MIN_NONEMPTY_LINES:
                if verbose:
                    print(f'EXCLUDE FILE: lines = {min_or_less_lines}')
                n_short += 1
                continue

            # Filter by `lang` char frequency:
            if cleaner.lang_char_frequency_in_filtered_lines < MIN_LANG_CHARS_FREQ:
                n_nonlc += 1
                if verbose:
                    print(f'EXCLUDE FILE: lang char freq = '
                          f'{cleaner.lang_char_frequency_in_filtered_lines}')
                continue

            # Filter by `lang` line frequency:
            # Deciding on `lang`-labeled line frequency is actually stricter than
            # deciding based on the whole document (which could get a high `lang`
            # probability even though many lines are not `lang`.
            #
            # Additionally, we are ignoring the probabilities.
            # labels, probs = lid_model.predict(lines)
            # lang_probs = [p for l, p in zip(labels, probs) if l == lang_label]
            # lang_prob  = np.concatenate(lang_probs).mean() if lang_probs else 0

            labels, __probs = lid_model.predict(lines)
            lengths         = list(map(len, lines))
            lang_freq       = sum(compress(
                lengths,
                (l == lang_label for l in labels)
                )) / sum(lengths)                     # weighted by line length

            if lang_freq < MIN_LANG_FREQ:
                n_lid += 1
                if verbose:
                    print(f'EXCLUDE FILE: detected lang = {lang_freq}')
                continue

            n_valid_removed_tags        += cleaner.n_removed_tags
            n_valid_removed_addresses   += cleaner.n_removed_addresses
            n_valid_lines_empty         += cleaner.n_lines_empty
            n_valid_lines_nonl          += cleaner.n_lines_nonl
            n_valid_lines_valid         += n_lines
            vtt_n_cues                  += cleaner.n_cues
            vtt_n_empty                 += cleaner.n_empty
            vtt_n_repeat                += cleaner.n_repeat
            vtt_n_scroll                += cleaner.n_scroll
            removed_addresses           += cleaner.removed_addresses
            removed_addresses_videoids  += [videoid] * len(cleaner.removed_addresses)

            text = '\n'.join(chain(lines, ('',)))   # adds trailing \n
            out_file = file.removesuffix(SUB_SUFFIX) + DATA_SUFFIX
            write_file(out_file, text)
            n_valid += 1

    n_valid_lines_total = (
        n_valid_lines_empty + n_valid_lines_nonl + n_valid_lines_valid
        )

    print('Cleaning stats:')
    print('* video IDs in the list:')
    print(f'  {n_list}')
    print('* files:')
    print(f'  {n_total} total')
    print(f'  {n_short} too short (<{MIN_NONEMPTY_LINES} lines)')
    print(f'  {n_nonlc} not enough {lang} characters (<{MIN_LANG_CHARS_FREQ} '
          f'characters of the corresponding charset)')
    print(f'  {n_lid} not enough detected {lang} (<{MIN_LANG_FREQ})')
    print(f'  {n_valid} valid files after cleaning')
    print('* sequences removed from valid files:')
    print(f'  {n_valid_removed_tags} tags')
    print(f'  {n_valid_removed_addresses} addresses')
    print('* lines in valid files:')
    print(f'  {n_valid_lines_total} total lines')
    print(f'  {n_valid_lines_empty} whitespace-only lines')
    print(f'  {n_valid_lines_nonl} lines composed of non-{lang} characters')
    print(f'  {n_valid_lines_valid} valid lines after cleaning')
    print('* VTT cue cleanup:')
    print(f'  {vtt_n_cues} total cues')
    print(f'  {vtt_n_empty} empty cues')
    print(f'  {vtt_n_repeat} repeated cues')
    print(f'  {vtt_n_scroll} scrolling cues')
    print()

    pd.DataFrame({
        'video_id': removed_addresses_videoids,
        'removed_address': removed_addresses
        }).to_csv(
            removed_addresses_path or (DEFAULT_REM_ADDR_PATH_FMT % lang),
            sep='\t', index=False
            )


def do_unique(
    lang: str,
    identifier: str,
    storage: Storage,
    limit: Optional[int],
    tokenize: Tokenizer,
    ngram_range: tuple[int, int],
    max_matching: bool = False
    ) -> None:
    with \
        get_write_file(UNIQUE_PATH_FMT % identifier, storage) as write_file, \
        get_files_contents(CLEAN_PATH_FMT % identifier, storage) as files_contents:
        files, iter_contents = files_contents

        tfidf = TfidfVectorizer(
            tokenizer=tokenize,
            ngram_range=ngram_range
            ).fit_transform(
            tqdm(
                desc='Building TF-IDF',
                iterable=iter_contents(limit),
                total=len(files[:limit])
                )
            )

        # Cosine similarity:
        # tf-idf is normalized so we only need to compute the mutual dot-product
        # of the vectors, i.e. linear kernel:
        sim = linear_kernel_piecewise(
            tfidf,
            wrapper=lambda pcs: tqdm(desc='Computing similarity', iterable=pcs)
            )
        sim = np.tril(sim, -1)  # under diagonal only

        # print('Similarity histogram:')
        # print(np.histogram(sim, bins=20))
        # print()

        dup = np.tril(sim, -1) >= SIMILARITY_LIMIT
        dup_is, dup_lower_js = np.where(dup)
        dup_indices = set()

        # In practice, it is rarely the case, but is generally not necessary to
        # remove all the duplicates.
        # E.g. assume the pairs (1, 2) and (2, 3), but not (1, 3) are duplicates.
        # After removing 2, it is no longer necessary to remove 3.)
        # We have such "removed" duplicates in `dup_indices`.
        for i, ijs in groupby(
            zip(dup_is, dup_lower_js),
            lambda ij: ij[0]
            ):
            # Note: all(j<i for j in js) because of np.tril()
            if any((j not in dup_indices) for _, j in ijs):
                # Similar to a j<i that hasn't been already removed
                dup_indices.add(i)
            # else:
            # We have already removed all j<i to which i is similar.
            # Thus we can keep i.

        # Creating a minimum list of duplicates = Minimum Vortex Cover (NP hard).
        # https://en.wikipedia.org/wiki/Vertex_cover
        #
        # In theory going vertex-by-vertex as we do, could result in a larger cover than
        # the well-known maximum matching 2-approximation, but for our data the cover is
        # actually smaller. More importantly vertex-by-vertex approach guarantees that
        # we keep (i.e. do not add into cover/`dup_indices`) at least one node from
        # each conneted subgraph.
        #
        # For our data, the maximmum matching 2-approximation (code below) removes
        # 2318 documents instead of just 1686.
        #
        # TODO use something that combines the virtues of both?
        #
        # for i, ijs in groupby(
        #     zip(dup_is, dup_lower_js),
        #     lambda ij: ij[0]
        #     ):
        #         for _, j in ijs:
        #             if j not in dup_indices:
        #                 dup_indices.add(i)
        #                 dup_indices.add(j)
        #                 break

        print(f'Duplicate (similarity >= {SIMILARITY_LIMIT}) stats:')
        print(f'  {len(files[:limit])} total')
        print(f'  {len(dup_indices)} duplicates removed')
        print(f'  {len(files[:limit])-len(dup_indices)} valid files')
        print()

        sorted_dup_indices = sorted(dup_indices)

        # To list duplicate pairs and their similarities:
        # for i, j in zip(dup_is, dup_lower_js):
        #     print(files[i], files[j], sim[i,j])

        # To check that there are no duplicates now:
        # no_dups = np.delete(
        #     np.delete(dup, sorted_dup_indices, axis=0),
        #     sorted_dup_indices, axis=1
        #     )
        # test_dup_i = np.where(no_dups)[0]
        # assert len(test_dup_i)==0, test_dup_i

        iter_dup = iter(sorted_dup_indices)
        next_dup = next(iter_dup, None)
        for i, (file,  text) in tqdm(
            desc='Copying valid files',
            iterable=enumerate(zip(files[:limit], iter_contents(limit))),
            total=len(files[:limit])
            ):
            if i == next_dup:
                next_dup = next(iter_dup, None)
            else:
                write_file(file, text)


LABOROTV_FILES = [
    'LaboroTVSpeech_v1.0b/data/train/text.csv',
    'LaboroTVSpeech_v1.0b/data/dev/text.csv',
    'LaboroTVSpeech_v2.0b/data/train/text.csv',
    'LaboroTVSpeech_v2.0b/data/dev/text.csv'
    ]
LABOROTV_SEG_PAT = re.compile(r'\+\w+\s*')


def do_frequencies(
    lang: str,
    identifier: str,
    storage: Storage,
    sublist: Optional[pd.DataFrame],   # for channel ids or categories
    tokenized_files: Optional[str],
    tokenize: Optional[Tokenizer],
    categories: bool,
    pos_tag: Optional[TokenizerTagger],
    filter_cc_descriptions: bool,
    limit: Optional[int],
    path: Optional[str],
    channel_stats_path: Optional[str],
    min_videos: int,
    min_channels: int,
    verbose: bool,
    laborotv: bool = False
    ) -> None:

    assert (tokenize is not None) != (pos_tag is not None), (tokenize, pos_tag)

    cat_ids = None
    if sublist is not None:
        channel_ids = sublist[
            # backward compatibility with the original JTubeSpeech
            'channelid' if 'channelid' in sublist else
            'channel_id'
            ]

        ch2n = Counter(channel_ids)
        n_no_channel = ch2n.pop('', 0)
        n2chn = Counter(ch2n.values())
        n_channels_and_no_channels = len(ch2n) + n_no_channel

        with open(
            channel_stats_path or (DEFAULT_CHANNEL_STATS_PATH_FMT % lang), 'wt'
            ) as f:
            f.write('videos_in_channel\tchannels\n')
            for n, chn in sorted(n2chn.items()):
                f.write(f'{n}\t{chn}\n')
            f.write(f'NO_CHANNEL_ID\t{n_no_channel}\n')

        if categories:
            cat2id  = {cat: cat_id for cat_id, cat in CAT_ID2CATEGORY.items()}
            cat_ids = sublist['categories'].apply(cat2id.__getitem__)
    else:
        # Only warn and fall back to not outputting categories:
        sys.stderr.write('Cannot count frequencies by category, missing sublist.\n')
        categories = False

        channel_ids = None
        n_channels_and_no_channels = None

    freq_path: str  = path or ((DEFAULT_FREQ_PATH_FMT % identifier) + storage.suffix)
    normalize       = '%' in freq_path

    counters = WordCounterGroup(
        normalize=normalize,
        channels=(channel_ids is not None),
        pos=(pos_tag is not None),
        categories=categories
        )
    n_cc_descriptions = 0

    with get_files_contents(
        tokenized_files or (UNIQUE_PATH_FMT % identifier),
        storage,
        any_suffix=(tokenized_files is not None),
        filenames=(LABOROTV_FILES if laborotv else None)
        ) as files_contents:
        files, iter_contents = files_contents
        n_videos = len(files[:limit])
        assert n_videos, 'Something went wrong, no subtitles found.'
        for video_no, (file, text) in tqdm(
            desc='Computing frequencies',
            iterable=enumerate(zip(files[:limit], iter_contents(limit))),
            total=n_videos
            ):
            video_id = file.removesuffix(DATA_SUFFIX)

            # Videos without a channel id are counted as unique 1-video channels:
            channel_id = (
                (channel_ids.loc[video_id] or video_no)
                if (channel_ids is not None) else None
                )
            # Category ID:
            cat_id = (
                cat_ids.loc[video_id]
                if (cat_ids is not None) else None
                )

            if tokenized_files is None:
                # Normalize tilde: always AND before tokenization:
                text = text.translate(NORMALIZE_FULLWIDTH_TILDE)

                # Remove and count censored words (included in frequencies)
                text, n = subn_yt_censored(' ', text)
                if tokenize is not None:
                    counters.add([YT_CENSORED] * n, channel_id, cat_id)
                else:
                    counters.add_pos([(YT_CENSORED, YT_CENSORED_POS)] * n,
                                     channel_id, cat_id)

                if filter_cc_descriptions:
                    # Remove and count CC descriptions
                    text, n = subn_cc_desc(' ', text)
                    n_cc_descriptions += n
            elif laborotv:
                for line in text.split('\n'):
                    if not line:
                        continue
                    words = LABOROTV_SEG_PAT.split(line.split(',', 1)[1])
                    assert not words[-1], (video_id, words)
                    counters.add(words[:-1], channel_id, cat_id)
                # In case of LaboroTV we consider v1 (train+dev) and v2 (train+dev)
                # two "videos":
                if video_no%2:
                    counters.close_doc()
                continue  # Bypass the usual tokenization process

            if tokenize is not None:
                words = tokenize(text)
                counters.add(words, channel_id, cat_id)
                if verbose:
                    print(f'{file}:')
                    for w in words:
                        print(w)
                    print()
            else:
                words_pos = pos_tag(text)
                counters.add_pos(words_pos, channel_id, cat_id)
                if verbose:
                    print(f'{file}:')
                    for w, p in words_pos:
                        print(f'{w}\t{p}')
                    print()

            counters.close_doc()

    if min_videos:
        counters.remove_less_than_min_docs(min_videos)
    if min_channels and (channel_ids is not None):
        counters.remove_less_than_min_channels(min_channels)

    cols = ['word', 'count', 'videos']
    if channel_ids is not None:
        cols.append('channels')
    if pos_tag is not None:
        cols.append('pos')
    # Categories are appended automatically if we used them.
    counters.dump(
        freq_path,
        storage,
        cols=cols,
        n_docs=n_videos,
        n_channels=n_channels_and_no_channels
        )

    print('Frequency counting stats:')
    if filter_cc_descriptions and not tokenized_files:
        print(f' {n_cc_descriptions} CC descriptions filtered')
    print(f' {counters.n_words} tokens counted')
    print()

    counters.warnings_for_markup()


def do_tokenize(
    lang: str,
    identifier: str,
    storage: Storage,
    # sublist: Optional[pd.DataFrame],   # for channel ids or categories
    tokenized_files: Optional[str],
    tokenize: Tokenizer,
    # categories: bool,
    # pos_tag: Optional[TokenizerTagger],
    # filter_cc_descriptions: bool,
    limit: Optional[int],
    path: Optional[str],
    # channel_stats_path: Optional[str],
    # min_videos: int,
    # min_channels: int,
    # verbose: bool
    ) -> None:

    # TODO: no sentence segmentation rules for Indonesian:
    sseg = pysbd.Segmenter('en' if (lang == 'id') else lang).segment

    tok_path: str  = path or (DEFAULT_TOK_PATH_FMT % identifier)
    assert '%' not in tok_path

    assert tokenize is not None

    ccs = []
    cci = 0
    ccn = 0

    def repl_cc_placeholder(m: re.Match) -> str:
        nonlocal ccs
        ccs.append(m.group(1).translate(NORMALIZE_CC))
        return CC_PLACEHOLDER_SPC

    def repl_placeholder_cc(m: re.Match) -> str:
        nonlocal ccs, cci, ccn
        assert cci < ccn
        cc = ccs[cci]
        cci += 1
        return cc

    with open(tok_path, 'w') as fo:
        with get_files_contents(
            tokenized_files or (UNIQUE_PATH_FMT % identifier),
            storage,
            any_suffix=(tokenized_files is not None)
            ) as files_contents:
            files, iter_contents = files_contents
            n_videos = len(files[:limit])
            assert n_videos, 'Something went wrong, no subtitles found.'
            for text in tqdm(
                desc='Tokenizing',
                iterable=iter_contents(limit),
                total=n_videos
                ):
                ccs = []
                cci = 0
                ccn = 0
                if not tokenized_files:
                    # Normalize tilde: always AND before tokenization:
                    text = text.translate(NORMALIZE_FULLWIDTH_TILDE)

                    # Normalize CC and censored words
                    text, ccn = subn_cc_desc(repl_cc_placeholder, text)

                sentences = sseg(convert_newlines_for_tokenization(text))

                tokenized = '\n'.join(' '.join(tokenize(s)) for s in sentences) + '\n'

                if ccn:
                    # TODO DEBUGME print('CC', ccs)
                    tokenized, phn = subn_cc_placeholder(repl_placeholder_cc, tokenized)
                    # TODO DEBUGME assert phn == ccn

                fo.write(nfkc_lower(tokenized))


def get_stanza_tokenizers(
    lang: str, full: bool, args: argparse.Namespace
    ) -> tuple[Tokenizer, Optional[Tokenizer], Optional[TokenizerTagger]]:
    '''
    See get_tokenizers()
    '''
    tokenize: Optional[Tokenizer] = None
    pos_tag: Optional[TokenizerTagger] = None

    import stanza

    surface_processors = 'tokenize,mwt'
    nlp_sentences = stanza_sentences_workaround(
        stanza.Pipeline(lang=lang, processors=surface_processors, logging_level='WARN')
        )
    split_len = args.split_len

    def surface_tokenize(s):
        return [word.text for sent in nlp_sentences(force_split(s, split_len))
                for word in sent.words]

    if full:
        full_processors = surface_processors + ',pos,lemma'
        full_nlp_sentences = stanza_sentences_workaround(
            stanza.Pipeline(lang=lang, processors=full_processors, logging_level='WARN')
            )
        if args.pos:
            if args.form == 'surface':
                def pos_tag(s):
                    return [(word.text, word.upos or NA_POS)
                            for sent in full_nlp_sentences(force_split(s, split_len))
                            for word in sent.words]
            else:
                # Return lemma both for lemma and base:
                # Stanza sometimes returns None as lemma for words it cannot lemmatize
                # (e.g. "What's" in Indonesian), in which case we return the surface
                # form (word.text) or empty string as a last resort. Empty string would
                # be later filtered out as a non-word.
                def pos_tag(s):
                    return [(word.lemma or word.text or '', word.upos or NA_POS)
                            for sent in full_nlp_sentences(force_split(s, split_len))
                            for word in sent.words]

        else:
            if args.form == 'surface':
                tokenize = surface_tokenize
            else:
                # both for lemma and base:
                def tokenize(s):
                    return [word.lemma
                            for sent in full_nlp_sentences(force_split(s, split_len))
                            for word in sent.words]

    assert not ((tokenize is not None) and (pos_tag is not None))
    return (surface_tokenize, tokenize, pos_tag)


def get_tokenizers(
    lang: str, tokenization: Optional[str], full: bool, args: argparse.Namespace
    ) -> tuple[Tokenizer, Optional[TokenizerTagger], Optional[TokenizerTagger]]:
    '''
    If `tokenization` is None, it's interpreted as 'stanza' for non-CJ languages.

    Returns a triple of:
    (1) surface tokenizer
    (2) tokenizer   if (full and not args.pos) else None
    (3) tagger      if (full and     args.pos) else None

    (1) is used only for TF-IDF and always outputs surface form
    (2) or (3) are used for frequency counting.


    The actual form (lemma, surface, or base) returned by (2) and (3) is determined by
    args.form, and it's the same for both.
    The forms base and surface are equivalent for all languages except Japanese.
    All forms are equivalent for Chinese.
    Likewise whether (2) and (3) filter non-words is determined by args.filter_tokens.

    Argument names from `args` used:
- tokenized_files
    - pos
    - form
    - extended_pos
    - smart_apostrophe
    - filter_tokens
    (tagger_from_args:)
    - dictionary, dicdir
    (via get_stanza_tokenizers:)
    - split_len
    '''
    tokenize: Optional[Tokenizer] = None
    pos_tag: Optional[TokenizerTagger] = None

    if args.tokenized_files:
        def surface_tokenize(s):
            return s.split()

        if full:
            assert (not args.pos) and (args.form == 'surface')
            tokenize = surface_tokenize
    elif lang == 'ja':
        if tokenization is not None:
            raise Exception('Cannot use non-default tokenization with Japanese.')
        with_pos = args.pos
        form = args.form
        extended_pos = args.extended_pos
        assert not extended_pos or with_pos, '--extended-pos requires --pos'

        wakati_parse = tagger_from_args(args).parse

        def surface_tokenize(s: str) -> list[str]:
            return wakati_parse(s).split(' ')

        if full:
            pos_tagger = POSTagger(
                tagger_from_args(args, OPT_BASE_LEMMA_READING_POS),
                extended=extended_pos,
                token_form=form,
                word_only=False
                )
            if with_pos:
                def pos_tag(s: str) -> list[tuple[str, str]]:
                    return list(pos_tagger(s))
            elif args.form == 'surface':
                tokenize = surface_tokenize
            else:
                def tokenize(s: str) -> list[tuple[str, str]]:
                    return [token for token, _pos in pos_tagger(s)]
    elif lang == 'zh':
        if tokenization is not None:
            raise Exception('Cannot use non-default tokenization with Chinese.')
        from jieba import cut as surface_tokenize  # type: ignore
        if full:
            if args.pos:
                from jieba.posseg import cut as jieba_posseg_cut

                def pos_tag(s: str) -> list[tuple[str, str]]:
                    # Jieba returns a generator of `pair` objects => convert
                    return list(map(tuple, jieba_posseg_cut(s)))
            else:
                tokenize = surface_tokenize  # lemma = base = surface
    elif lang == 'en':
        en_tokenize: Optional[Tokenizer] = None
        en_pos_tag: Optional[TokenizerTagger] = None
        if tokenization == 'regex':
            en_surface_tokenize = get_re_split().split
            if full:
                if args.pos:
                    raise Exception('Cannot tag POS with regex tokenizer.')
                en_tokenize = en_surface_tokenize
        elif tokenization == 'treebank':
            from nltk.tokenize import word_tokenize  # type: ignore
            try:
                word_tokenize('Split this!')
            except LookupError:
                sys.stderr.write(
                    'It seems that "punkt" package for NLTK hasn\'t been '
                    'downloaded yet. Will try to download.\n'
                    )
                import nltk  # type: ignore
                nltk.download('punkt')
            en_surface_tokenize = word_tokenize
            if full:
                if args.pos:
                    raise Exception('Cannot tag POS with treebank tokenizer.')
                en_tokenize = en_surface_tokenize
        else:
            assert (tokenization is None) or (tokenization == 'stanza')
            (
                en_surface_tokenize,
                en_tokenize,
                en_pos_tag
                ) = get_stanza_tokenizers(lang, full, args)

        if args.smart_apostrophe:
            def surface_tokenize(s):
                return en_surface_tokenize(sub_smart_apos(repl_smart_apos, s))
            if full:
                if args.pos:
                    assert en_pos_tag is not None

                    def pos_tag(s):
                        return en_pos_tag(sub_smart_apos(repl_smart_apos, s))
                else:
                    assert en_tokenize is not None

                    def tokenize(s):
                        return en_tokenize(sub_smart_apos(repl_smart_apos, s))
        else:
            surface_tokenize = en_surface_tokenize
            tokenize = en_tokenize
            pos_tag = en_pos_tag

        assert not ((tokenize is not None) and (pos_tag is not None))
        assert not full or ((tokenize is not None) or (pos_tag is not None))
    else:
        if tokenization == 'treebank':
            raise Exception('Cannot use treebank tokenization with language {lang}.')
        if tokenization == 'regex':
            surface_tokenize = get_re_split().split
            if full:
                if args.pos:
                    raise Exception('Cannot tag POS with regex tokenizer.')
                tokenize = surface_tokenize
        else:
            assert (tokenization is None) or (tokenization == 'stanza')
            surface_tokenize, tokenize, pos_tag = get_stanza_tokenizers(
                lang, full, args
                )

    if not args.filter_tokens:
        assert not ((tokenize is not None) and (pos_tag is not None))
        return (surface_tokenize, tokenize, pos_tag)

    re_word = get_re_word_relaxed()
    is_word = re_word.match if (re_word is not None) else None

    f_tokenize: Optional[Tokenizer] = None
    f_pos_tag: Optional[TokenizerTagger] = None

    def f_surface_tokenize(s: str) -> list[str]:
        return list(filter(is_word, surface_tokenize(s)))

    if full:
        if args.pos:
            def f_pos_tag(s: str) -> list[str]:
                return list(filter(lambda token_pos: is_word(token_pos[0]), pos_tag(s)))
        else:
            def f_tokenize(s: str) -> list[str]:
                return list(filter(is_word, tokenize(s)))

    return (f_surface_tokenize, f_tokenize, f_pos_tag)


def main() -> None:
    args = parse()
    storage = Storage.from_args(args)
    limit = args.limit
    clean = args.clean
    unique = args.unique
    frequencies = args.frequencies
    # args.tokenize
    with_pos = args.pos
    categories = args.categories
    tokenized_files = args.tokenized_files
    limit_categories = args.limit_categories

    assert (
        (not tokenized_files) or
        ((frequencies or args.tokenize) and args.output and not args.list)
        ), (
        '--tokenized-files requires --frequencies, --output, no --list option.'
        )

    if not (clean or unique or frequencies or args.tokenize):
        clean = True
        unique = True
        frequencies = True

    lang = args.language
    identifier = args.identifier
    if not identifier:
        cat_suf     = (
            ('+' + '+'.join(sorted(limit_categories))) if limit_categories else
            ''
            )
        identifier  = f'{lang}{cat_suf}'

    if limit_categories:
        if not clean:
            raise Exception(
                '--limit-categories can only be applied at the --clean stage.'
                )
        if tokenized_files:
            raise Exception(
                '--limit--categories cannot be applied with --tokenized-files.'
                )
    if args.laborotv and not tokenized_files:
        raise Exception('--laborotv cannot be applied without --tokenized-files.')

    # sublist: for file filtering (clean), and channel ids (frequencies)

    if (clean or frequencies) and not tokenized_files:
        list_path = args.list or (SUBLIST_PATH_FMT % (lang, lang))
        all_subtitles = pd.read_csv(
            list_path,
            index_col='videoid',
            na_filter=False  # keep empty channelids as empty strings
            )

        # Filtering for original JTubeSpeech:
        # Keep manual only:
        manual_subtitles = all_subtitles[all_subtitles['sub']]
        # Remove duplicates
        # (pairs where ['auto']==True, ['auto']==False -- we don't care):
        sublist = manual_subtitles[~manual_subtitles.index.duplicated()]
        if limit_categories and clean:
            if 'categories' not in sublist.columns:
                raise Exception(
                    f'Subtitle list (--list) at "{list_path}" does not have a '
                    f'"categories" column required to apply --limit-categories.'
                    )
            cat_set = set(map(CAT_ID2CATEGORY.get, limit_categories))
            cat_cond = sublist['categories'].apply(lambda c: c in cat_set)
            sublist = sublist.loc[cat_cond]
    else:
        sublist = None

    if clean:
        data_path = args.data or (DATA_PATH_FMT % lang)
        do_clean(
            lang,
            identifier,
            storage,
            sublist,
            data_path,
            removed_addresses_path=args.removed_addresses,
            limit=limit,
            verbose=args.verbose
            )

    if unique or frequencies or args.tokenize:
        tokenization           = args.tokenization
        unique_tokenization    = args.unique_tokenization
        if unique_tokenization is None and (lang not in ('ja', 'zh')):
            unique_tokenization = (
                'treebank' if lang == 'en' else
                'regex'
                )
        surface_tokenize, tokenize, pos_tag = get_tokenizers(
            lang=lang, tokenization=tokenization, full=(frequencies or args.tokenize),
            args=args
            )
        if unique_tokenization != tokenization:
            surface_tokenize, _, _ = get_tokenizers(
                lang=lang, tokenization=unique_tokenization, full=False, args=args
                )

        if unique:
            do_unique(
                lang,
                identifier,
                storage,
                tokenize=surface_tokenize,
                limit=limit,
                ngram_range=(args.nmin, args.nmax)
                )
        if frequencies:
            assert (tokenize is not None) != with_pos, (with_pos, tokenize, pos_tag)
            assert (pos_tag is not None) == with_pos
            do_frequencies(
                lang,
                identifier,
                storage,
                sublist,
                tokenized_files=tokenized_files,
                laborotv=args.laborotv,
                tokenize=tokenize,
                categories=categories,
                pos_tag=pos_tag,
                filter_cc_descriptions=args.filter_cc_descriptions,
                limit=limit,
                path=args.output,
                channel_stats_path=args.channel_stats,
                min_videos=args.min_videos,
                min_channels=args.min_channels,
                verbose=args.verbose
                )
        elif args.tokenize:
            assert (tokenize is not None)
            assert (pos_tag is None)
            assert not with_pos
            do_tokenize(
                lang,
                identifier,
                storage,
                tokenized_files=tokenized_files,
                tokenize=tokenize,
                limit=limit,
                path=args.output
                )


if __name__ == '__main__':
    main()
