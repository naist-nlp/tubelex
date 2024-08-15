from typing import NamedTuple, Optional
from contextlib import contextmanager
from zipfile import ZipFile
from typing import TextIO, TypedDict, Union, ContextManager
from collections import Counter
from collections.abc import Sequence, Iterator
from urllib.request import urlretrieve
import lzma
import io
import sys
import os

CHECK_CASE = False


def _mkdir_parent(path: str) -> None:
    parent, __ = os.path.split(path)
    if parent:  # may be ''
        os.makedirs(parent, exist_ok=True)


def download_if_necessary(
    url: str,
    path: Optional[str] = None     # None -> return random (temporary) filename
    ) -> Optional[str]:
    if path is not None:
        if os.path.exists(path):
            return None
        _mkdir_parent(path)
    sys.stderr.write(f'Downloading data from "{url}"...\n')
    urlretrieve(url, filename=path)
    sys.stderr.write(f'Finished download to "{path}".\n')
    return path


class FrequencyDataSpecBase(TypedDict, total=True):
    # required (total=True):
    filename: str


class FrequencyDataSpec(FrequencyDataSpecBase, total=False):
    # optional (total=False):
    url: str
    zip_args: Union[tuple[str], tuple[str, str]]
    total_row: bool
    total_header: bool
    header: bool
    cols: Sequence[str]
    cased: bool
    to_lower: bool


# All of SUBTLEX files are linked from here:
# http://crr.ugent.be/programs-data/subtitle-frequencies

LANG2SUBT: dict[str, FrequencyDataSpec] = {
    # https://www.ugent.be/pp/experimentele-psychologie/
    # en/research/documents/subtlexus/overview.htm
    'en': FrequencyDataSpec(
        filename='data/downloads/subtlex-en.zip',
        url=(
            'https://www.ugent.be/pp/experimentele-psychologie/en/research/documents/'
            'subtlexus/subtlexus2.zip'
            ),
        cols=('Word', 'FREQcount', 'CDcount'),
        to_lower=True,
        ),
    # http://crr.ugent.be/programs-data/subtitle-frequencies/subtlex-ch
    'zh': FrequencyDataSpec(
        filename='data/downloads/subtlex-zh.zip',
        url=('http://www.ugent.be/pp/experimentele-psychologie/'
             'en/research/documents/subtlexch/subtlexchwf.zip'),
        zip_args=('SUBTLEX-CH-WF', 'gb18030'),
        total_header=True,
        cols=('Word', 'WCount', 'W-CD'),
        to_lower=True,
        ),
    # http://crr.ugent.be/archives/679
    #
    # Only available as a hand-formatted XLS file without docs/context diversity.
    # We redistribute a plain text version under the CC license (also see comments
    # inside the file.)
    #
    # Citation/Paper:
    #   Cuetos, F., Glez-Nosti, M., Barbon, A., & Brysbaert, M. (2011). SUBTLEX-ESP:
    #   Spanish word frequencies based on film subtitles. Psicologica, 32, 133-143.
    #   http://crr.ugent.be/papers/CUETOS%20et%20al%202011.pdf
    # License:
    #   Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License
    #   http://creativecommons.org/licenses/by-nc-nd/3.0/deed.en_US
    'es': FrequencyDataSpec(
        filename='data/subtlex-es.tsv.xz',
        cols=('word', 'count'),
        ),
    # https://psico.fcep.urv.cat/projectes/gip/papers/SUBTLEX-CAT.xlsx:
    #
    # Only available as a hand-formatted XLS file
    #
    # Citation/Paper:
    #   Roger Boada, Marc Guasch, Juan Haro, Josep Demestre, and Pilar Ferré. 2020.
    #   SUBTLEX-CAT: Subtitle word frequencies and contextual diversity for Catalan.
    #   Behavior Research Methods, 52(1):360–375.
    #   https://link.springer.com/article/10.3758/s13428-019-01233-1
    'ca': FrequencyDataSpec(
        filename='data/subtlex-cat.tsv.xz',
        cols=('word', 'count'),
        ),
    # http://crr.ugent.be/programs-data/subtitle-frequencies/subtlex-pt-br
    'pt': FrequencyDataSpec(
        filename='data/downloads/subtlex-pt.zip',
        url='http://crr.ugent.be/subtlex-pt-br/csv/SUBTLEX-BRPOR.zip',
        cols=('Word', 'FREQcount', 'CDcount'),
        to_lower=True
        ),
    }
FREQ_DATA_LANGS     = list(LANG2SUBT.keys())
FREQ_DATA_CORPORA   = ['subtitles', 'wiki']

TOTAL_KEY = '[TOTAL]'
COLS_DEFAULT    = 3  # (word, frequency, contextual diversity)
COLS_RANGE      = range(2, 4)   # at least (word, frequency)


@contextmanager
def single_text_file_zip(
    path: str,
    filename: Optional[str] = None,
    encoding: str = 'utf-8'
    ) -> Iterator[TextIO]:
    zf = None
    bf = None
    tf = None
    try:
        zf = ZipFile(path)
        if filename is not None:
            bf = zf.open(filename)
        else:
            infos = zf.infolist()
            assert len(infos) == 1, infos
            bf = zf.open(infos[0])
        tf = io.TextIOWrapper(bf, encoding=encoding)
        yield tf
    finally:
        if zf is not None:
            zf.close()
            if bf is not None:
                bf.close()
                if tf is not None:
                    tf.close()


def _total_from_header(line: str, label: Optional[str] = None) -> int:
    fields: list[str] = line.rstrip().strip('"').split(': ')
    assert len(fields) == 2
    if label is not None and fields[0] != label:
        raise ValueError(f'Expected {label}, found {fields[0]}.')
    return int(fields[1].replace(',', ''))


_fd_cache: dict[tuple[str, str], 'FrequencyData'] = {}


class FrequencyData(NamedTuple):
    f: Counter[str]               # frequency
    cd: Optional[Counter[str]]    # contextual diversity (document frequency)
    f_total: int
    cd_total: Optional[int] = None

    @staticmethod
    def load(
        file: TextIO,
        total_row: bool = False,
        total_header: bool = False,
        header: bool = True,
        cols: Optional[Sequence[str]] = None,
        cased: bool = False,
        to_lower: bool = False,
        verbose: bool = False,
        filename: Optional[str] = None  # for exceptions
        ) -> 'FrequencyData':
        f: Counter[str] = Counter()
        cd: Optional[Counter[str]]
        f_total: int
        cd_total: Optional[int]

        exc_fn = f'"{filename}": ' if (filename is not None) else ''

        if total_header:
            if total_row:
                raise ValueError(f'{exc_fn}Both total_header and total_row are True.')
            f_total = _total_from_header(next(file), 'Total word count')
            cd_total = _total_from_header(next(file), 'Context number')

        if to_lower and cased:
            raise ValueError(
                f'{exc_fn}Both to_lower and cased are True.'
                )

        if cols is not None:
            if not header:
                raise ValueError(f'cols={cols}, but header=False.')
            if len(cols) not in COLS_RANGE:
                raise ValueError(
                    f'{exc_fn}Number of columns to read must be {COLS_RANGE}, '
                    f'but cols={cols}.'
                    )

        indices: Sequence[int] = range(COLS_DEFAULT)
        if header:
            for line in file:
                # ignore any number of opening comments that start with a '#'
                if line.startswith('#'):
                    continue
                if cols is not None:
                    header_cols = line.rstrip('\n').split('\t')
                    indices = [header_cols.index(c) for c in cols]
                # else just ignore
                break

        cd = Counter() if (len(indices) == COLS_DEFAULT) else None

        for line in file:
            fields = line.rstrip('\n').split('\t')
            word, freq, *opt_docs = (fields[i] for i in indices)

            if to_lower:
                word = word.lower()

            # Use += to allow for possible lowercasing via `to_lower`:
            f[word]         += int(freq)
            if cd is not None:
                wcd = int(*opt_docs)
                if to_lower:
                    cd[word] = max(cd[word], wcd)
                cd[word]    += wcd

        if total_row:
            f_total         = f.pop(TOTAL_KEY)
            cd_total        = cd.pop(TOTAL_KEY) if (cd is not None) else None
        elif not total_header:
            f_total         = sum(f.values())
            cd_total        = None

        if CHECK_CASE:
            # w.islower() is False for CJK, so we use `w.lower()==w`:
            if not to_lower and (all(w.lower() == w for w in f) == cased):
                message = (
                    'Expected cased data, but all words are lowercase.' if cased else
                    'Expected uncased data, but some words contain uppercase.'
                    )
                raise Exception(f'{exc_fn}{message}')

        if verbose:
            n_total = len(f)
            total_source = (
                'total_row' if total_row else
                'total_header' if total_header else
                'computed'
                )
            lcase_msg = ' after lowercasing' if to_lower else ''
            sys.stderr.write(
                f'- {n_total} words in file{lcase_msg}\n'
                f'- totals ({total_source}): f={f_total}, cd={cd_total}\n'
                )

        fd = FrequencyData(f, cd, f_total, cd_total)
        return fd


    def smooth_frequency_missing(self, word: str) -> tuple[float, bool]:
        '''
        Return a pair of values:
        - non-zero float: frequency smoothed out for missing values,
        - bool: whether the word is missing.

                                count(w) + 1
        smooth_frequency(w) = ----------------
                              #tokens + #types
        '''
        f  = self.f
        count_w = f.get(word, 0)
        return (
            (count_w + 1) / (self.f_total + len(f)),    # smooth_frequency
            not count_w                                 # missing
            )


    @staticmethod
    def _open(
        filename: str,
        zip_args=()
        ) -> ContextManager[TextIO]:
        if filename.endswith('.xz'):
            return lzma.open(filename, 'rt')
        if filename.endswith('.zip'):
            return single_text_file_zip(filename, *zip_args)
        return open(filename)

    @staticmethod
    def from_file(
        filename: str,
        zip_args=(),
        total_row: bool = False,
        total_header: bool = False,
        header: bool = True,
        cols: Optional[Sequence[str]] = None,
        cased: bool = False,
        to_lower: bool = False,
        verbose: bool = False
        ) -> 'FrequencyData':
        with FrequencyData._open(filename, zip_args) as file:
            return FrequencyData.load(
                file, total_row, total_header, header, cols, cased, to_lower,
                verbose=verbose, filename=filename
                )

    @staticmethod
    def from_file_url(
        filename: str,
        url: Optional[str] = None,
        zip_args=(),
        total_row: bool = False,
        total_header: bool = False,
        header: bool = True,
        cols: Optional[Sequence[str]] = None,
        cased: bool = False,
        to_lower: bool = False,
        force_verbose: bool = False
        ) -> 'FrequencyData':

        verbose: bool = force_verbose

        if (url is not None) and (download_if_necessary(url, filename) is not None):
            verbose = True  # be verbose if freshly downloaded (already logged info)
        elif force_verbose:
            sys.stderr.write(f'Local file "{filename}".\n')

        return FrequencyData.from_file(
            filename, zip_args, total_row, total_header, header, cols, cased, to_lower,
            verbose=verbose
            )

    @staticmethod
    def from_wiki(
        lang: str,
        force_verbose: bool = False
        ) -> 'FrequencyData':
        '''
        Get uncased Wikipedia frequencies for any of the supported languages.
        '''
        # From wikipedia-word-frequency-clean v0.2
        if lang == 'id':
            filename = f'{lang}wiki-frequency-20240801-nfkc-lower.tsv.xz'
        else:
            assert lang in {'en', 'zh', 'es', 'pt', 'ja'}
            filename = f'{lang}wiki-frequency-20221020-nfkc-lower.tsv.xz'
        return FrequencyData.from_file_url(
            filename=f'data/downloads/{filename}',
            url=(
                f'https://github.com/adno/wikipedia-word-frequency-clean/raw/v0.2/'
                f'results/{filename}'
                ),
            total_row=True
            )

    @staticmethod
    def from_subtitles(
        lang: str,
        force_verbose: bool = False
        ) -> 'FrequencyData':
        '''
        Get uncased SUBTLEX frequencies for any of the supported languages.
        '''
        spec = LANG2SUBT[lang]
        return FrequencyData.from_file_url(
            **spec,
            force_verbose=force_verbose
            )

    @staticmethod
    def from_corpus(
        corpus: str,
        lang: str,
        force_verbose: bool = False,
        cache: bool = True
        ) -> 'FrequencyData':
        if (corpus not in FREQ_DATA_CORPORA) or (lang not in FREQ_DATA_LANGS):
            raise Exception(
                f'FrequencyData not available for corpus={corpus}, lang={lang}.'
                )
        if cache:
            cfd = _fd_cache.get((corpus, lang))
            if cfd is not None:
                return cfd

        if corpus == 'wiki':
            fd = FrequencyData.from_wiki(lang, force_verbose=force_verbose)
        else:
            assert corpus == 'subtitles'
            fd = FrequencyData.from_subtitles(lang, force_verbose=force_verbose)

        if cache:
            _fd_cache[(corpus, lang)] = fd

        return fd


if __name__ == '__main__':
    import doctest
    doctest.testmod()
