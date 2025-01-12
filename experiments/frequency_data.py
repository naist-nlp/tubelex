from typing import NamedTuple, Optional
from contextlib import contextmanager
from zipfile import ZipFile
from typing import TextIO, TypedDict, Union, ContextManager
from collections import Counter
from collections.abc import Sequence, Iterator
from urllib.request import urlretrieve
import pickle
import lzma
import gzip
import io
import sys
import os
import numpy as np
import pandas as pd
import scipy as sp

CHECK_CASE = False

NP_EPS = np.finfo(float).eps


class CounterDict(dict):
    '''
    Like defaultdict, but doesn't insert missing values, behaving similar to Counter.
    '''
    def __init__(self, default_factory, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_factory = default_factory

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        return self.default_factory()


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
    total_row: bool | str  # pass str for a key different from TOTAL_KEY
    total_header: bool
    header: bool | Sequence[str]
    cols: Sequence[str]
    sub_lemma: Optional[str]  # subLemma for CSJ (joined using '-')
    delimiter: str
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
    # https://www.psychology.nottingham.ac.uk/subtlex-uk/
    'en-uk': FrequencyDataSpec(
        filename='data/downloads/subtlex-uk.zip',
        url=(
            'https://www.psychology.nottingham.ac.uk/subtlex-uk/SUBTLEX-UK.txt.zip'
            ),
        cols=('Spelling', 'FreqCount', 'CD_count'),
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
CAT_PREFIX = 'count:'
COLS_DEFAULT    = 3  # (word, frequency, contextual diversity)
COLS_RANGE      = range(2, 4)   # at least (word, frequency)
DEFAULT_DELIMITER = '\t'


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


class NoCountsType:
    _singleton = None

    def __new__(cls):
        if cls._singleton is None:
            cls._singleton = super().__new__(cls)
        return cls._singleton


NoCounts = NoCountsType()


class CountArrays:
    # Data has units as columns, words as rows:
    data: pd.DataFrame      # columns are pd.arrays.SparseArray
    cache: dict[str, Optional[pd.arrays.SparseArray]]
    cache_updated: bool
    cache_filename: Optional[str]
    totals: pd.arrays.SparseArray
    units: int
    zeros: pd.arrays.SparseArray

    def __init__(
        self,
        filename: str,
        cache: Optional[str]
        ):
        with open(filename, 'rb') as f:
            self.data = pickle.load(f)
        self.units = len(self.data.columns)
        self.zeros = pd.arrays.SparseArray(np.zeros(self.units, dtype='int32'))
        self.cache_filename = cache
        if cache is not None and os.path.exists(cache):
            with open(cache, 'rb') as cf:
                self.cache = pickle.load(cf)
            self.totals = self.cache.get(TOTAL_KEY)
        else:
            self.cache = {}
            self.totals = None
        assert self.cache is not None
        self.cache_updated = False
        if self.totals is None:
            self.totals = self.data.sum().array
            self.cache[TOTAL_KEY] = self.totals
            self.cache_updated = True

    def get(self, word, default=None):
        # Return `default` (None) if word is not found.
        # We cache missing words too (NoCounts)
        counts = self.cache.get(word)

        if counts is NoCounts:
            return None
        if counts is None:
            try:
                counts = self.data.loc[word].array
            except KeyError:
                self.cache[word] = NoCounts
                self.cache_updated = True
                return None
            self.cache[word] = counts
            self.cache_updated = True
        return counts

    def __getitem__(self, word: str) -> pd.arrays.SparseArray:
        # Return zeros if word is not found.
        if (counts := self.get(word)) is None:
            return self.zeros
        return counts

    def __contains__(self, word: str) -> bool:
        return self.get(word) is not None

    def save_cache(self) -> None:
        if self.cache_filename is not None and self.cache_updated:
            with open(self.cache_filename, 'wb') as cf:
                pickle.dump(self.cache, cf)
            self.cache_updated = False


class FrequencyData(NamedTuple):
    f: Counter[str]               # frequency
    cd: Optional[Counter[str]]    # contextual diversity (document frequency)
    f_total: int
    cd_total: Optional[int] = None
    # Categories/counts:
    cnt_f: CounterDict | CountArrays | None = None      # CountArrays are sparse
    cnt_f_totals: list[int] | np.ndarray | None = None  # cnt_f_totals is always dense
    # Cumulative sum of cnt_f_totals sorted descending:
    cs_cnt_f_totals: np.ndarray | None = None

    @staticmethod
    def load(
        file: TextIO,
        total_row: bool | str = False,  # pass str for a key different from TOTAL_KEY
        total_header: bool = False,
        header: bool | Sequence[str] = True,
        cols: Optional[Sequence[str]] = None,
        sub_lemma: Optional[str] = None,  # subLemma for CSJ (joined using '-')
        delimiter: str = DEFAULT_DELIMITER,
        cased: bool = False,
        to_lower: bool = False,
        counts: Optional[CountArrays] = None,
        categories: bool = False,
        categories_as_cd: bool = False,
        verbose: bool = False,
        filename: Optional[str] = None,  # for exceptions
        ignore_errors: bool = False,
        ) -> 'FrequencyData':
        f: Counter[str] = Counter()
        cd: Optional[Counter[str]]
        f_total: int
        cd_total: Optional[int]
        n_ignored_errors: int = 0

        exc_fn = f'"{filename}": ' if (filename is not None) else ''

        if total_header:
            if total_row:
                raise ValueError(f'{exc_fn}Both total_header and total_row are True.')
            if categories or categories_as_cd:
                raise ValueError(
                    f'{exc_fn}Both total_header categories(_as_cd) are True.'
                    )
            f_total = _total_from_header(next(file), 'Total word count')
            cd_total = _total_from_header(next(file), 'Context number')
            cnt_f_totals = None

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
        if sub_lemma is not None:
            if not cols:
                raise ValueError(f'sub_lemma={sub_lemma}, but cols is None.')

        if (categories or categories_as_cd) and not header:
            raise ValueError(
                f'{exc_fn}Categories(_as_cd) are True, but header is False.'
                )
        if categories and (counts is not None):
            raise ValueError(
                f'{exc_fn}Categories are True and counts are not None.'
                )

        indices: Sequence[int] = range(COLS_DEFAULT)
        cat_indices: Optional[Sequence[int]] = None
        if header:
            # get `header_cols`: for cols, categories or categories_as_cd
            # (not used directly for anything else)
            if isinstance(header, Sequence):
                header_cols = header
            else:
                assert isinstance(header, bool)
                for line in file:
                    # ignore any number of opening comments that start with a '#'
                    if line.startswith('#'):
                        continue
                    header_cols = line.rstrip('\n').split(delimiter)
                    break
            if cols:
                indices = [header_cols.index(c) for c in cols]
                if sub_lemma is not None:
                    sub_lemma_index = header_cols.index(sub_lemma)
            if categories or categories_as_cd:
                cat_indices = np.array([
                    i for i, c in enumerate(header_cols)
                    if c.startswith(CAT_PREFIX)
                    ], dtype=int)

        cd = Counter() if (
            (len(indices) == COLS_DEFAULT) or
            categories_as_cd
            ) else None
        cnt_f = (CounterDict(lambda: np.zeros_like(cat_indices)) if categories else
                 None)

        freq = None
        for line in file:
            line = line.rstrip(f'\n{delimiter}')
            if not line:
                continue    # Ignore lines only consisting of tabs/empty (SUBTLEX-UK)
            fields = line.split(delimiter)
            try:
                word, freq, *opt_docs = (fields[i] for i in indices)
                # Only add non-empty sub_lemma:
                if (sub_lemma is not None) and (word_sl := fields[sub_lemma_index]):
                    word = f'{word}-{word_sl}'
                if to_lower:
                    word = word.lower()

                # Use += to allow for possible lowercasing via `to_lower`:
                f[word]         += int(freq)

                if cat_indices is not None:
                    wcnt_f = np.array(fields)[cat_indices].astype(int)
                    if categories:
                        cnt_f[word] += wcnt_f

                if cd is not None:
                    wcd = (
                        (wcnt_f != 0).sum() if categories_as_cd else
                        int(*opt_docs)
                        )
                    if to_lower:
                        cd[word] = max(cd[word], wcd)
                    cd[word]    += wcd
            except Exception:
                if ignore_errors:
                    n_ignored_errors += 1
                else:
                    raise Exception(
                        f'{exc_fn}Error parsing freq={freq!r} on line "{line}" '
                        f'using indices={indices}.'
                        )

        if total_row:
            total_key = total_row if isinstance(total_row, str) else TOTAL_KEY
            f_total         = f.pop(total_key)
            # Works for categories_as_cd too (assuming all categories are non-empty):
            cd_total        = cd.pop(total_key) if (cd is not None) else None
            cnt_f_totals    = cnt_f.pop(total_key) if categories else None
        elif not total_header:
            f_total         = sum(f.values())
            cd_total        = None
            cnt_f_totals    = None

        if counts is not None:
            cnt_f = counts
            cnt_f_totals = np.array(counts.totals)

        if cnt_f_totals is not None:
            # `cnt_f_totals` => cumulative sum s (`cs_cnt_f_totals`):
            # x = (x[1], x[2], ..., x[n]): sorted descending x[1] >= x[2] >= ... >= x[n]
            # s = (s[0], s[1], ..., s[n]): s[i] = sum(x[<=i]), note: s[0] = 0
            cs_cnt_f_totals = np.concatenate((
                np.zeros(1, dtype=cnt_f_totals.dtype),  # prepend with [0]
                -np.sort(-cnt_f_totals).cumsum()        # sort descending; cumsum
                ))
        else:
            cs_cnt_f_totals = None

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
                f'- totals ({total_source}): f={f_total}, cd={cd_total}, '
                f'cnt_f={cnt_f_totals}\n'
                )
            if n_ignored_errors:
                sys.stderr.write(f'- {n_ignored_errors} ignored errors\n')

        fd = FrequencyData(f, cd, f_total, cd_total, cnt_f,
                           cnt_f_totals, cs_cnt_f_totals)
        return fd

    def smooth_frequency_missing(self, word: str) -> tuple[float, bool]:
        '''
        Return a pair of values:
        - non-zero float: frequency smoothed out for missing values,
        - bool: whether the word is missing.

                                count(w) + 1
        smooth_frequency(w) = ----------------
                              #tokens + #types
        (i.e. Laplace smoothing)
        '''
        f  = self.f
        count_w = f[word]
        return (
            (count_w + 1) / (self.f_total + len(f)),    # smooth_frequency
            not count_w                                 # missing
            )

    def simple_smooth_frequency(self, word: str) -> float:
        '''
        Return a non-zero float: frequency smoothed out for missing values:

                              count(w) + 1
        smooth_frequency(w) = ------------
                              #tokens + 1
        Note: This is NOT Laplace smoothing (uncorrected Laplace smoothing)
        '''
        return (self.f[word] + 1) / (self.f_total + 1)    # smooth_frequency

    # TODO UNUSED:
    # def smooth_cnt_frequencies_missing(self, word: str) -> tuple[np.array, np.array]:
    #     '''
    #     Return a pair of vectors (size=#categories):
    #     - non-zero floats: frequencies smoothed out for missing values,
    #     - bools: whether the word is missing.
    #
    #                             count(w) + 1
    #     smooth_frequency(w) = ----------------
    #                           #tokens + #types
    #
    #     Note: #types counted in the whole corpus, not per category!
    #     '''
    #     f  = self.cnt_f
    #     count_w = f[word]
    #     return (
    #         (count_w + 1) / (self.cnt_f_totals + len(f)),   # smooth_frequency
    #         count_w == 0                                    # missing
    #         )
    # All of the following are defined so that:
    # - Values fall in [0, 1]
    # - Values = 1 or close to 1 correspond to maximum dispersion (distribution
    #   even across parts). If the original definition did not achieve this, we redefine
    #   the metric X as X_dispersion = 1 - X.
    # - For a word outside corpus, we always return 0 (minimum dispersion), regardless
    #   of what would be computed by the formula.
    # - The dispersions are computed in a way appropriate for corpus parts of different
    #   size. If necessary, we adjusted the original formulas for that.
    #
    # Available measures:
    # - range (normalized contextual diversity)
    # - gini_dispersion
    # - maxmin_dispersion
    # - juilland_d
    # - gries_dp_dispersion
    # - rosengren_s
    # - carrol_d2

    def cd_range(self, word: str, smooth: bool = False) -> float:
        '''
        Normalized range (in 0..1) based on `cd`, (categories only if categories_as_cd).
        '''
        return (
            (self.cd[word] + 1) / (self.cd_total + 1) if smooth else
            self.cd[word] / self.cd_total               # may be 0 if word not in cd
            )

    def range_nofreq(self, word: str, smooth: bool = False) -> float:
        '''
        Normalized range (in 0..1) only for categories/counts (not based on cd),
        "controlled" for frequency - improves (Gries, 2021).
        '''
        cnt_f_totals = self.cnt_f_totals
        cs_cnt_f_totals = self.cs_cnt_f_totals

        wc      = self.f[word]                   # word count
        if not wc:
            return 1                             # decoupled from freq: max. dispersion
        n       = len(cnt_f_totals)              # part count
        r       = (self.cnt_f[word] != 0).sum()  # range (#parts w occurs in)
        # Theoretical min and max for `r` fiven the `wc`:
        r_hi     = min(wc, n)
        r_lo     = np.searchsorted(cs_cnt_f_totals, wc)  # 0..n

        assert r_lo <= r <= r_hi, (
            (r_lo, r, r_hi),
            (cs_cnt_f_totals[0], cs_cnt_f_totals[r_lo], cs_cnt_f_totals[-1]),
            cs_cnt_f_totals
            )

        if r_hi == r_lo:
            print('r_hi == r_lo', word, r_hi, file=sys.stderr)  # TODO
            return 1

        return (
            (r - r_lo + 1) / (r_hi - r_lo + 1) if smooth else
            (r - r_lo) / (r_hi - r_lo)
            )

    def range_nofreq_gries(self, word: str, smooth: bool = False) -> float:
        '''
        Normalized range (in 0..1) only for categories/counts (not based on cd),
        "controlled" for frequency (Gries, 2021).
        '''
        cnt_f_totals = self.cnt_f_totals

        wc      = self.f[word]                   # word count
        if not wc:
            return 1                             # decoupled from freq: max. dispersion
        n       = len(cnt_f_totals)              # part count
        r       = (self.cnt_f[word] != 0).sum()  # range (#parts w occurs in)
        # Theoretical min and max for `r` fiven the `wc`:
        r_hi    = min(wc, n)
        r_lo    = int(wc > 0)  # either 0 or 1 (Gries says always 1)

        if r_hi == r_lo:
            print('gries: r_hi == r_lo', word, r_hi, file=sys.stderr)  # TODO
            return 1

        return (
            (r - r_lo + 1) / (r_hi - r_lo + 1) if smooth else
            (r - r_lo) / (r_hi - r_lo)
            )

    def weighted_range(self, word: str, smooth: bool = False) -> float:
        '''
        Normalized weighted range, only for categories/counts (not based on cd).
        '''

        cnt_f_totals = self.cnt_f_totals
        return (
            (
                (((self.cnt_f[word] != 0) * cnt_f_totals).sum() + 1) /
                (cnt_f_totals.sum() + 1)
                ) if smooth else
            ((self.cnt_f[word] != 0) * cnt_f_totals).sum() / cnt_f_totals.sum()
            )

    def sparse_gini_dispersion(
        self, word: str, smooth: bool = False, weight: bool = False
        ) -> float:
        return self.gini_dispersion(word, smooth=smooth, weight=weight, sparse=True)

    def sort_gini_dispersion(
        self, word: str, smooth: bool = False, weight: bool = False
        ) -> float:
        return self.gini_dispersion(word, smooth=smooth, weight=weight, sort=True)

    def gini_dispersion(
        self, word: str, smooth: bool = False, weight: bool = False,
        sparse: bool = False, sort: bool = False
        ) -> float:
        '''
        This is Gini *dispersion* (i.e. equality), i.e. the complement of
        the Gini inequality index.

        As described by Murayama et al. (2018), except for finally not applying -log.
        Equivalent: DA (Egbert et al., 2020).

        Smoothing is added by smoothing the individual frequencies.
        '''

        if sort:
            f = self.cnt_f
            if word not in f:
                return 1 / (len(self.cnt_f_totals) + 1) if smooth else 0.0  # no dispersion
            f_w = f[word]
            if not isinstance(f_w, pd.arrays.SparseArray):
                f_w = pd.arrays.SparseArray(f_w)
            assert f_w.fill_value == 0
            nz_indices  = f_w.sp_index.indices
            nz_values   = f_w.sp_values
            nz_totals   = self.cnt_f_totals[nz_indices]
            n = len(f_w)
            # The following computations are non-sparse (using only non-zero values)
            f_w = nz_values / nz_totals                 # normalize by unit
            f_w /= f_w.sum()                            # normalize by word
            f_w.sort()
            nnz = len(f_w)

            coef = np.arange(n - 2 * nnz + 1, n, 2)

            if smooth:
                n += 1
            g = 1 - (coef * f_w).sum() / n

        elif sparse:
            raise Exception('Do not use, TODO DELETEME.')
            assert not smooth
            f = self.cnt_f
            if word not in f:
                return 0.0                              # no dispersion
            # TODO this basically makes the array non-sparse:
            f_w = f[word] / self.cnt_f_totals           # normalize by unit

            # We normalize by word before the final computation, as this will make the
            # numbers larger, resulting in better precision:
            f_w /= f_w.sum()                            # normalize by word

            n = len(f_w)

            # TOOD optimize the above?
            # TOOD we ignore the sparsity we have, create CSR from scratch:
            sp_f_w = sp.sparse.csr_matrix(f_w)
            as_rows = sp.sparse.vstack([sp_f_w] * sp_f_w.shape[1], 'csr')
            as_cols = as_rows.T
            assert not weight
            g = 1 - abs(as_rows - as_cols).sum() / (2 * n)
        else:
            raise Exception('Do not use, TODO DELETEME.')
            if smooth:
                f_w = self.simple_smooth_cnt_frequencies(word)
            else:
                f = self.cnt_f
                if word not in f:
                    return 0.0                              # no dispersion
                # TODO this basically makes the array non-sparse:
                f_w = f[word] / self.cnt_f_totals           # normalize by unit

            # We normalize by word before the final computation, as this will make the
            # numbers larger, resulting in better precision:
            f_w /= f_w.sum()                            # normalize by word

            n = len(f_w)

            as_rows = np.tile(f_w, (n, 1))
            as_cols = as_rows.T

            # No need to divide by f_w.sum() == f_w.mean() * n,
            # which is == 1 after the normalization by word.
            # Note that we are returning complement (1 - Gini inequality), i.e. index
            # of dispersion
            if weight:
                w = as_rows * as_cols
                d = np.abs(as_rows - as_cols)
                return 1 - (w * d).sum() / (2 * n)

            g = 1 - np.abs(as_rows - as_cols).sum() / (2 * n)

        return g

    def maxmin_dispersion(self, word: str, smooth: bool = False) -> float:
        raise Exception('Do not use, TODO DELETEME.')
        if smooth:
            f_w = self.simple_smooth_cnt_frequencies(word)
        else:
            f = self.cnt_f
            if word not in f:
                return 0.0                              # no dispersion
            f_w = f[word] / self.cnt_f_totals           # normalize by unit

        # Do not normalize by word, max - min is already in 0..1:
        return 1 - (f_w.max() - f_w.min())  # 1 - maxmin

    def juilland_d(self, word: str, smooth: bool = False) -> float:
        # ~ variation coefficient

        f = self.cnt_f
        if word not in f:
            return 1 / (len(self.cnt_f_totals) + 1) if smooth else 0.0   # no dispersion
        f_w = f[word] / self.cnt_f_totals           # normalize by unit

        # Do not normalize by word: no effect on VC = SD / mean:
        vc = np.std(f_w) / np.mean(f_w)
        n = len(f_w)
        d = 1 - vc / np.sqrt(n - 1)
        if not smooth:
            return d
        return (d * n + 1) / (n + 1)

    def vmr_dispersion(self, word: str, smooth: bool = False) -> float:
        # https://en.wikipedia.org/wiki/Index_of_dispersion
        # variance-to-mean ratio (VMR)

        f = self.cnt_f
        if word not in f:
            return 1 / (len(self.cnt_f_totals) + 1) if smooth else 0.0   # no dispersion
        f_w = f[word] / self.cnt_f_totals           # normalize by unit

        f_w /= f_w.sum()                        # Normalize by word => 0..1 range

        vmr = np.var(f_w) / np.mean(f_w)
        d = 1 - vmr
        if not smooth:
            return d
        n = len(f_w)
        return (d * n + 1) / (n + 1)

    def gries_dp_dispersion(self, word: str, smooth: bool = False) -> float:
        f = self.cnt_f
        if not smooth and word not in f:
            return 1 / (len(self.cnt_f_totals) + 1) if smooth else 0.0    # no dispersion
        f_w = f[word]

        f_totals    = self.cnt_f_totals

        if smooth:
            # Smoothing as if using simple_smooth_cnt_frequencies().
            # Note: Avoid += so that we do not overwrite original values in dict.
            f_w         = f_w + 1
            f_totals    = f_totals + 1
        cat_prop    = f_totals / f_totals.sum()
        word_prop   = f_w / f_w.sum()

        # Return 1 - DP (= D_P in Egbert et al.(2020))
        d = 1 - np.sum(np.abs(word_prop - cat_prop)) / 2
        if not smooth:
            return d
        n = len(f_w)
        return (d * n + 1) / (n + 1)

    def lyne_d3(self, word: str, smooth: bool = False) -> float:
        f = self.cnt_f
        if not smooth and word not in f:
            return 1 / (len(self.cnt_f_totals) + 1) if smooth else 0.0    # no dispersion
        f_w = f[word]

        f_totals    = self.cnt_f_totals

        if smooth:
            # Smoothing as if using simple_smooth_cnt_frequencies().
            # Note: Avoid += so that we do not overwrite original values in dict.
            f_w         = f_w + 1
            f_totals    = f_totals + 1
        cat_prop    = f_totals / f_totals.sum()
        word_prop   = f_w / f_w.sum()

        d = 1 - np.sum((word_prop - cat_prop)**2) / 4
        if not smooth:
            return d
        n = len(f_w)
        return (d * n + 1) / (n + 1)

    def rosengren_s(self, word: str, smooth: bool = False) -> float:
        f = self.cnt_f
        if word not in f:
            return 1 / (len(self.cnt_f_totals) + 1) if smooth else 0.0    # no dispersion
        f_w = f[word] / self.cnt_f_totals           # normalize by unit

        # We normalize by word before the final computation, as this will make the
        # numbers larger, resulting in better precision:
        f_w /= f_w.sum()                            # normalize by word
        n = len(f_w)
        if smooth:
            return (np.sqrt(f_w).sum() ** 2 + 1) / (n + 1)
        return np.sqrt(f_w).sum() ** 2 / n

    # def rosengren_like_sqrt(self, word: str, smooth: bool = False) -> float:
    #     f = self.cnt_f
    #     if word not in f:
    #         return 1 / (len(self.cnt_f_totals) + 1) if smooth else 0.0    # no dispersion
    #     f_w = f[word] / self.cnt_f_totals           # normalize by unit
    #
    #     # We normalize by word before the final computation, as this will make the
    #     # numbers larger, resulting in better precision:
    #     f_w /= f_w.sum()                            # normalize by word
    #     n = len(f_w)
    #     if smooth:
    #         return (np.sqrt(f_w).sum() + 1) / (n + 1)
    #     return np.sqrt(f_w).sum() / n

    def carrol_d2(self, word: str, smooth: bool = False) -> float:
        f = self.cnt_f
        if word not in f:
            return 1 / (len(self.cnt_f_totals) + 1) if smooth else 0.0    # no dispersion
        f_w = f[word] / self.cnt_f_totals           # normalize by unit

        # We normalize by word before the final computation, as this will make the
        # numbers larger, resulting in better precision:
        f_w /= f_w.sum()                            # normalize by word

        nz_f_w = f_w[f_w != 0]                      # ignore zeros (avoid -Inf->NaNs)
        log_f_w = np.log(nz_f_w)
        entropy = - (nz_f_w * log_f_w).sum()

        # 1 ~ max entropy ~ max dispersion
        # Same as \eta (efficiency) in information theory
        n = len(f_w)
        d = entropy / np.log(n)           # do NOT ignore zeros here
        if not smooth:
            return d
        return (d * n + 1) / (n + 1)

    @staticmethod
    def _open(
        filename: str,
        zip_args=()
        ) -> ContextManager[TextIO]:
        if filename.endswith('.xz'):
            return lzma.open(filename, 'rt')
        if filename.endswith('.gz'):
            return gzip.open(filename, 'rt')
        if filename.endswith('.zip'):
            return single_text_file_zip(filename, *zip_args)
        return open(filename)

    @staticmethod
    def from_file(
        filename: str,
        zip_args=(),
        total_row: bool | str = False,  # pass str for a key different from TOTAL_KEY
        total_header: bool = False,
        header: bool | Sequence[str] = True,
        cols: Optional[Sequence[str]] = None,
        sub_lemma: Optional[str] = None,  # subLemma for CSJ (joined using '-')
        delimiter: str = DEFAULT_DELIMITER,
        cased: bool = False,
        to_lower: bool = False,
        counts: Optional[CountArrays] = None,
        categories: bool = False,
        categories_as_cd: bool = False,
        verbose: bool = False,
        ignore_errors: bool = False
        ) -> 'FrequencyData':
        with FrequencyData._open(filename, zip_args) as file:
            return FrequencyData.load(
                file, total_row, total_header, header, cols, sub_lemma, delimiter,
                cased, to_lower, counts, categories, categories_as_cd,
                verbose=verbose, filename=filename, ignore_errors=ignore_errors
                )

    @staticmethod
    def from_file_url(
        filename: str,
        url: Optional[str] = None,
        zip_args=(),
        total_row: bool | str = False,  # pass str for a key different from TOTAL_KEY
        total_header: bool = False,
        header: bool | Sequence[str] = True,
        cols: Optional[Sequence[str]] = None,
        sub_lemma: Optional[str] = None,  # subLemma for CSJ (joined using '-')
        delimiter: str = DEFAULT_DELIMITER,
        cased: bool = False,
        to_lower: bool = False,
        counts: Optional[CountArrays] = None,
        categories: bool = False,
        categories_as_cd: bool = False,
        force_verbose: bool = False,
        ignore_errors: bool = False
        ) -> 'FrequencyData':

        verbose: bool = force_verbose

        if (url is not None) and (download_if_necessary(url, filename) is not None):
            verbose = True  # be verbose if freshly downloaded (already logged info)
        elif force_verbose:
            sys.stderr.write(f'Local file "{filename}".\n')

        return FrequencyData.from_file(
            filename, zip_args, total_row, total_header, header, cols, sub_lemma,
            delimiter, cased, to_lower, counts, categories, categories_as_cd,
            verbose=verbose, ignore_errors=ignore_errors
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

    def difference(self, other: 'FrequencyData') -> 'FrequencyData':
        f = self.f.copy()
        f.subtract(other.f)
        negw = [w for w, c in f.items() if c < 0]
        if negw:
            raise ValueError(f'Words have negative frequency after subtraction: {negw}')
        f_total = self.f_total - other.f_total
        if f_total < 0:
            raise ValueError(f'Negative total frequency after subtraction: {f_total}')

        cd = None
        cd_total = None
        if (self.cd is None) != (other.cd is None):
            raise ValueError('One of the objects has CD, while the other does not')
        if self.cd is not None:
            cd = self.cd.copy()
            cd.subtract(other.cd)
            negw = [w for w, c in cd.items() if c < 0]
            if negw:
                raise ValueError(f'Words have negative CD after subtraction: {negw}')
            cd_total = self.cd_total - other.cd_total
            if cd_total < 0:
                raise ValueError(f'Negative total CD after subtraction: {cd_total}')

        return FrequencyData(f, cd, f_total, cd_total)  # ignores categories/counts


if __name__ == '__main__':
    import doctest
    doctest.testmod()
