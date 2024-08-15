'''
Language processing for `tubelex` and `wikipedia-word-frequency-clean`.
'''

import fugashi  # type: ignore
import os
from typing import Optional
from collections.abc import Callable, Iterator
import argparse
import re
from extended_pos import (
    X_PARTICLE_POS, X_VERB_POS, X_AUX_POS, X_MIMETIC_POS,
    VV_POS, VV_READING_SET, PAT_PARTICLE_AUX, AUX_GROUP, aux2base,
    MIMETIC_POS, MIMETIC_SET
    )


# Word matching (not just) for Japanese


def _assert_safe_for_re_range(s: str) -> None:
    '''
    Sanity checks before we insert `s` it at the end of a regex range [...s].
    '''
    assert len(s) == len(set(s))
    assert ']' not in s
    assert '\\' not in s
    assert ('-' not in s) or s.endswith('-')


def get_re_word(
    allow_start_end: str = '',
    allow_end: str = ''
    ) -> re.Pattern:
    '''
    Match words of len>=1. No decimal digits (\\d) at any position.
    First and last character must be word-forming (\\w), i.e. alphabet, CJK, etc.

    Note: \\w includes accented chars, CJK, etc.
    \\d are decimals in many scripts, but not CJK.

    Use `allow_start_end` to allow characters other than \\w, such as hyphen,
    apostrophe (English) or wave dash (Japanese) to appear as the first or last
    characters. (Note: does not work for adding digits.)

    Use `allow_end` to allow characters to appear as last characters of a word longer
    than a single character.

    Useful both for space-separated languages (segmented with regex) and languages
    requiring more complex segmentation (Chinese, Japanese).
    '''

    _assert_safe_for_re_range(allow_start_end)
    _assert_safe_for_re_range(allow_end)
    assert '-' not in allow_end

    return re.compile(
        rf'^(?!\d)[\w{allow_start_end}]'
        rf'([^\d]*[\w{allow_end}{allow_start_end}])?(?<!\d)$'
        )


def get_re_word_relaxed() -> re.Pattern:
    '''
    All non-digit ([^\\d]), at least one word-forming ([\\w]) character.
    '''
    return re.compile(
        r'^([^\d]*(?!\d)[\w][^\d]*)$'
        )


def get_re_split(no_split: str = '') -> re.Pattern:
    '''
    Match non-word sequences to split words. Such sequences may consist of:
    - characters not in \\w or in `no_split`
    - characters in \\d

    For languages that can be segmented with a regex (not Chinese or Japanase).
    Also see `get_re_word()`.
    '''
    _assert_safe_for_re_range(no_split)

    # We need a non-capturing group '(?:...)' for split() to use the whole regex
    return re.compile(rf'(?:[^\w{no_split}]|\d)+')


WAVE_DASH   = '\u301C'  # 〜 may look like fullwidth tilde ～
EN_DASH     = '\u2013'  # – may look like hyphen -


# Examples (test):
_re_word = get_re_word()
_re_split = get_re_split()
assert all(_re_word.fullmatch(w) for w in ['a', '亀', 'コアラ', 'Pú', 'A/B', 'bla-bla'])
assert not any(
    _re_word.match(w) for w in ['', '1', 'a1', '1a', 'C3PIO', '/', '-', 'あ〜']
    )
assert get_re_word(allow_start_end=WAVE_DASH).match('あ〜')
assert (
    _re_split.split('a.b  cč5dď-eé\'ff1+2*3.5koala') ==
    ['a', 'b', 'cč', 'dď', 'eé', 'ff', 'koala']
    )


NORMALIZE_FULLWIDTH_TILDE: dict[int, int] = {
    0xFF5E: 0x301C  # fullwidth tilde '～' (common typo) => wave dash '〜'
    }

OPT_WAKATI = '-O wakati'
OPT_UNIDIC = '-O unidic'
OPT_BASE_LEMMA_READING_POS = (
    r'-O "" '
    r'-F "%m\\t%f[10]\\t%f[7]\\t%f[6]\\t%F-[0,1,2,3]\\n" '
    # surface, "orthBase", "lemma", "lForm" (語彙素読み), POS
    r'-U "%m\\t%m\\t%m\\t%m\\tUNK\n"'  # for UNK
    )


def fugashi_tagger(
    dicdir: Optional[str],
    option: str = OPT_WAKATI
    ) -> fugashi.GenericTagger:
    if dicdir is None:
        return fugashi.Tagger(option)  # -d/-r supplied automatically
    # GenericTagger: we do not supply wrapper (not needed for -O wakati)
    mecabrc = os.path.join(dicdir, 'mecabrc')
    return fugashi.GenericTagger(f'{option} -d {dicdir} -r {mecabrc}')


def add_tagger_arg_group(
    parser: argparse.ArgumentParser,
    title: Optional[str] = None
    ):
    titled_group = parser.add_argument_group(title=title)
    dic_group = titled_group.add_mutually_exclusive_group()
    dic_group.add_argument(
        '--dicdir', type=str, default=None,
        help='Dictionary directory for fugashi/MeCab.'
        )
    dic_group.add_argument(
        '--dictionary', '-D', choices=('unidic', 'unidic-lite'), default=None,
        help=(
            'Dictionary (installed as a Python package) for fugashi/MeCab.'
            'Default: unidic-lite.'
            )
        )


def tagger_from_args(
    args: argparse.Namespace,
    option: str = OPT_WAKATI
    ) -> fugashi.GenericTagger:

    # We always specify dicdir EXPLICITLY
    if args.dicdir is not None:
        dicdir = args.dicdir
    else:
        if args.dictionary == 'unidic':
            import unidic  # type: ignore
            dicdir = unidic.DICDIR
        else:
            assert args.dictionary is None or args.dictionary == 'unidic-lite'
            import unidic_lite  # type: ignore
            dicdir = unidic_lite.DICDIR
    return fugashi_tagger(dicdir, option)


RE_WORD = get_re_word(allow_start_end=WAVE_DASH)

_SAHEN_NOUN_POS  = '名詞-普通名詞-サ変可能'
_SAHEN_VERB_POS  = '動詞-非自立可能'
_SAHEN_MARKER    = 'サ'
_SAHEN_VERB_LEMMAS = {
    '為る',
    '出来る'
    '致す',
    '為さる',
    '頂く',
    '下さる'
    }
SAHEN_VERB_NOUN_POS = '動詞-サ変'   # used if sahen_verbs=True (not used by MeCab)


class POSTagger:
    __slots__ = ('tagger_parse', 'extended', 'sahen_verbs', 'ret_index', 'word_only')
    tagger_parse: Callable[[str], str]
    extended: bool
    sahen_verbs: bool
    ret_index: int
    word_only: bool

    def __init__(
        self,
        tagger: Callable[[str], str],  # OPT_BASE_LEMMA_READING_POS tagger
        extended: bool = False,
        sahen_verbs: bool = False,
        token_form: str = 'surface',  # One of ['surface', 'base', 'lemma']
        word_only: bool = True
        ):
        assert not sahen_verbs or extended, 'POSTagger: sahen_verbs requires extended'
        self.tagger_parse   = tagger.parse
        self.extended       = extended
        self.sahen_verbs    = sahen_verbs
        self.ret_index      = ['surface', 'base', 'lemma'].index(token_form)
        self.word_only      = word_only

    def _pos_tag(self, s: str) -> Iterator[tuple[str, str]]:
        tagger_parse = self.tagger_parse
        extended = self.extended
        sahen_verbs = self.sahen_verbs
        ret_index = self.ret_index
        word_only = self.word_only

        lines = tagger_parse(s).split('\n')
        token_buffer = ''
        nonword_token = False
        prev_pos = None
        for line in lines:
            if line == 'EOS':
                nonword_token = True
            else:
                fields = line.split('\t')
                token, base, lemma, lemma_reading, pos = fields
                if word_only and not RE_WORD.match(token):
                    nonword_token = True
            if nonword_token:
                if extended:
                    # will yield both compound in addition to single tokens:
                    for m in PAT_PARTICLE_AUX.finditer(token_buffer):
                        if aux_tokens := m.group(AUX_GROUP):
                            yield (
                                aux2base(aux_tokens) if self.ret_index!=0 else
                                aux_tokens.replace(' ', ''),    # surface
                                X_AUX_POS
                                )
                        else:
                            yield (m.group(1).replace(' ', ''), X_PARTICLE_POS)
                    token_buffer = ''
                nonword_token = False
                prev_pos = None
                continue
            token_buffer += f' {token}'
            if extended:
                # will yield only the compound verb (single token):
                if pos == VV_POS:
                    if lemma_reading in VV_READING_SET:
                        pos = X_VERB_POS
                elif pos == MIMETIC_POS and token in MIMETIC_SET:
                    pos = X_MIMETIC_POS
                elif (
                    sahen_verbs and
                    (pos == _SAHEN_VERB_POS) and
                    (prev_pos == _SAHEN_NOUN_POS) and
                    (lemma in _SAHEN_VERB_LEMMAS)
                    ):
                    pos = _SAHEN_MARKER
            yield (fields[ret_index], pos)
            prev_pos = pos

    def __call__(self, s: str) -> Iterator[tuple[str, str]]:
        iter_pos_tag = self._pos_tag(s)
        if self.sahen_verbs:
            # if pos is _SAHEN_MARKER:
            #   yield SAHEN_VERB_NOUN_POS instead of previous pos
            #   yield _SAHEN_VERB_POS (the original 動詞-非自立可能) for this pos
            for ret0, pos0 in iter_pos_tag:
                break
            else:
                return
            for ret, pos in iter_pos_tag:
                if pos is _SAHEN_MARKER:
                    pos     = _SAHEN_VERB_POS
                    pos0    = SAHEN_VERB_NOUN_POS
                yield (ret0, pos0)
                ret0 = ret
                pos0 = pos
            yield (ret0, pos0)
            return
        else:
            yield from iter_pos_tag

# The right single quote '’' (but not the left single quote) is allowed to occur
# inside ‘...’ as long as it is surrounded by \w from both sides (\b’\b in the RE).
# E.g. ‘It’s an apostrophe.’ => “It’s an apostrophe.”
#
# The following RE and replaceement function replaces:
# 1. legit single quotes by double quotes
# 2. primes that look like apostrophe
RSQUOTE2APOS: dict[int, int] = {ord('’'): ord('\'')}

RE_SMART_APOS     = re.compile(
    # Preserve -- has group(1)
    r'‘(([^‘’]*\b’\b)*[^‘’]*)’(?!s)|'   # paired single quotes, see `in_quotes` below
    # Replace by apostrophe:
    r'’|'                               # right single quote except pairs like above
    r'(?<=[A-Za-z]{2})′|'               # prime following at least two alphabet letters
    r'′(?=s)'                           # prime before 's'
    )
sub_smart_apos   = RE_SMART_APOS.sub  # optimization


def repl_smart_apos(m: re.Match) -> str:
    '''
    Translates "smart" apostrophe (right single quote ’ or prime ′) to apostrophe '.
    Keeps legit "‘...’" or "a′" as is.

    Basic quotes/apostrophies:

    >>> sub_smart_apos(repl_smart_apos, 'It’s me. It’s ‘you and me’.')
    "It's me. It's ‘you and me’."

    Trickier case (resolved using r'(?!s)'):

    >>> sub_smart_apos(repl_smart_apos, '‘It’s A’ ‘and’ it’s B.')
    "‘It's A’ ‘and’ it's B."

    Imperfect matching:

    >>> sub_smart_apos(repl_smart_apos,
    ...     'This isn‘t an apostrophe. ‘It’s an apostrophe.’ ‘This isn’t an apostrophe.'
    ...     )
    "This isn‘t an apostrophe. ‘It's an apostrophe.’ ‘This isn’t an apostrophe."

    Primes:

    >>> sub_smart_apos(repl_smart_apos, 'It′s an a′, it can′t be b′. Countries′ names.')
    "It's an a′, it can't be b′. Countries' names."
    '''

    in_quotes = m.group(1)  # inside paired single quotes
    return (
        # Keep outer quotes, replace inner right single quotes using translate():
        f'‘{in_quotes.translate(RSQUOTE2APOS)}’' if in_quotes is not None
        # Replace other right single quotes or primes found by regex:
        else '\''
        )
