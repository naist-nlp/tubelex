'''
WebVTT subtitle/caption parsing and conversion to plain text,
language independent, with support for ruby.
'''

import re
import sys
from typing import Optional
from collections.abc import Iterable, Iterator

TIMING_SEP = '-->'


def _make_ts(hs: Optional[str], ms: str, ss: str, ts: str) -> Optional[float]:
    h = int(hs) if (hs is not None) else 0
    m = int(ms)
    s = int(ss)
    t = int(ts)
    if m >= 60 or s >= 60:
        return None
    return (h * 3600 +
            m * 60 +
            s + t / 1000)


RE_TS       = r'(?:(\d{2,}):)?(\d{2}):(\d{2})\.(\d{3})'
PAT_TIMING  = re.compile(
    fr'{RE_TS}[ \t]+-->[ \t]+'
    fr'{RE_TS}(?:[ \t\n]|$)'
    )

# VTT subtitles contain only the following entities:
ENTITY2STR = {
    'lt': '<', 'gt': '>', 'amp': '&',
    'nbsp': ' ',                        # replace NBSP with normal space
    'lrm': '\u200E', 'rlm': '\u200F'
    }
PAT_ENTITY  = re.compile(r'&(%s);' % '|'.join(ENTITY2STR))


def repl_entities(m: re.Match) -> str:
    # Used as: text = PAT_ENTITY.sub(repl_entities, text)
    return ENTITY2STR[m.group(1)]


sub_entity  = PAT_ENTITY.sub

PAT_TAGS    = re.compile(
    fr'<[cibu](\.[^\s&<>.]*)*>|'                # c, i, b, u: no annotation
    # Note: <c.> sometimes occur too
    fr'<{RE_TS}>|'                              # timestamp tag
    fr'<(v|lang)(\.[^\s&<>.]*)*[\t ][^<>]+>|'   # v, lang:  has annotation
    fr'</([cibuv]|lang)>'
    )
sub_tags    = PAT_TAGS.sub

# Within <ruby>:
# Keep untagged content or content tagged <rb> (the base text),
# (and ignore <rp>/<rt> outside <ruby>):
PAT_RUBY     = re.compile(
    r'<ruby(?: [^>]+)?>(.*?)</ruby>|'
    r'<rp(?: [^>]+)?>.*?</rp>|'
    r'<rt(?: [^>]+)?>.*?</rt>'
    )  # .*? non-greedy, group(1) is None in case of rp/rt
# Do not keep content in <rp> (ruby parents) or <rt> (ruby text, i.e. furigana)
PAT_RUBY_DEL = re.compile(r'<rp>[^<]*</rp>|<rt( [^>]+)?>[^<]*</rt>|</?rb>')
sub_ruby_del        = PAT_RUBY_DEL.sub
sub_ruby            = PAT_RUBY.sub


def repl_ruby(m: re.Match) -> str:
    in_ruby = m.group(1)
    if in_ruby is None:
        return ''       # <rp> or <rt> outside ruby => replace with ''
    return sub_ruby_del('', in_ruby)


def markup2text(s: str) -> str:
    t = sub_ruby(repl_ruby, sub_tags('', s))
    if '<' in t:
        sys.stderr.write(
            f'Unrecognized tag delimiter "<" in WebVTT cue: "{t}"\n'
            f'Original string: "{s}"\n'
            )
    return sub_entity(repl_entities, t)


def parse_timing(s: str) -> Optional[tuple[float, float]]:
    '''
    Accepted format returns the corresponding number of seconds:
    - hh[...]:mm:ss:ttt:
    >>> parse_timing('01:02:03.001 --> 9999:01:59.001')
    (3723.001, 35996519.001)

    - mm:ss:ttt:
    >>> parse_ts('00:01:59.999 --> 59:34.567')
    (119.999, 3574.567)

    Wrong format returns None:
    - wrong numbers of digits:
    >>> parse_ts('01:02:03.0010')
    >>> parse_ts('01:02:030.001')
    >>> parse_ts('2:03.001')

    - wrong ms separator:
    >>> parse_ts('01:02:03,004')

    - too many seconds or minutes:
    >>> parse_ts('60:03.001')
    >>> parse_ts('01:60.000')

    - leading characters:
    >>> parse_ts(' 01:02:03.001')
    - trailing characters:
    >>> parse_ts('01:02:03.001\n')
    '''
    m = PAT_TIMING.match(s)
    if m is None:
        return None
    digit_fields = m.groups()
    start = _make_ts(*digit_fields[:4])
    if start is None:
        return None
    end = _make_ts(*digit_fields[4:])
    if end is None:
        return None
    return (start, end)


Cue = tuple[float, float, str]


def vtt2cues(vtt: Iterable[str]) -> Iterator[Cue]:
    '''
    Iteratively transform lines of WebVTT text to `Cue`s, while removing markup.
    May output WebVTT format errors to stderr, but never fails.
    '''
    start       = None
    end         = None
    cue_lines   = None
    is_timing   = False

    # Cue must start with a timing
    # Cue may end with a timing for the next cue, empty line or end of file
    # The string '-->' must not occur except for a timing.

    for line in vtt:
        line = line.rstrip('\n')
        if TIMING_SEP in line:
            is_timing = True

        if cue_lines is not None:
            if is_timing or not line:
                # May write to stderr:
                yield (start, end, markup2text('\n'.join(cue_lines)))
                cue_lines = None
            else:
                cue_lines.append(line)

        if is_timing:
            timing = parse_timing(line)
            if timing is None:
                sys.stderr.write(f'Unrecognized WebVTT timing line: "{line.strip()}"\n')
                assert cue_lines is None  # keep reading lines until we find a timing
                continue
            start, end = timing
            cue_lines = []
            is_timing = False
    if cue_lines is not None:
        yield (start, end, markup2text('\n'.join(cue_lines)))


PAT_SPACE   = re.compile('\s+')
sub_space   = PAT_SPACE.sub


class VTTCleaner:
    '''
    Clean and filter lines via the __call__ method. Counts (instance variables) reflect
    the last call (non-cumulative). Counts are not set until the iterator is exhausted.
    '''
    verbose: bool
    n_cues: int
    n_empty: int
    n_repeat: int
    n_scroll: int
    n_delay: int

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def __str__(self) -> str:
        n_cues      = self.n_cues
        n_other     = n_cues - self.n_empty - self.n_repeat - self.n_scroll
        p_empty     = (self.n_empty / n_cues) if n_cues else 0
        p_repeat    = (self.n_repeat / n_cues) if n_cues else 0
        p_scroll    = (self.n_scroll / n_cues) if n_cues else 0
        p_other     = (n_other / n_cues) if n_cues else 0
        return (
            f'empty:      {self.n_empty:6d} ({p_empty:.2%})\n'
            f'repeated:   {self.n_repeat:6d} ({p_repeat:.2%})\n'
            f'scrolling:  {self.n_scroll:6d} ({p_scroll:.2%})\n'
            f'other:      {n_other:6d} ({p_other:.2%})\n'
            f'TOTAL CUES: {n_cues:6d}\n'
            f'delays:     {self.n_delay:6d}'
            )

    def cues2clean_text(
        self,
        cues: Iterable[Cue],
        delays_as_empty: bool = True
        ) -> Iterator[str]:
        '''
        Cleans up repetition/subtitle scrolling, generting cue text without timings.
        Lines within a cue are still '\\n' separated (without trailing '\\n's).

        By default (delays_as_empty=True), delays between cues result into empty strings
        in output.
        '''
        # Reset the counts, and set them only if the iterator is exhausted
        # (safety measure)
        if hasattr(self, 'n_cues'):
            del self.n_cues
            del self.n_empty
            del self.n_repeat
            del self.n_scroll
            del self.n_delay

        n_cues = 0
        n_empty = 0
        n_repeat = 0
        n_scroll = 0
        n_delay = 0

        prev_end    = None
        prev_text   = None
        for start, end, text in cues:
            n_cues += 1
            text = text.strip()     # often starts with ' \n' or ends with '\n '
            if not text:
                n_empty += 1
                continue    # Totally ignore empty text
            if prev_text is None or start > prev_end:
                # Note: If subtitles overlap (start < prev_end), we treat it the same as
                # no delay (start == prev_end)
                # If no prev_text, or there is a delay, yield:
                if prev_text is not None:
                    n_delay += 1
                    if delays_as_empty:
                        yield ''
                yield text
            elif text == prev_text:
                # Same cue repeated without delay -- yield nothing:
                prev_end = end
                n_repeat += 1
                continue
            else:
                # Scrolling -- yield only new text:
                # No delay, but text is different check scrolling and yield
                # only new text:
                prev_tail = prev_text.split('\n',
                                            1)[-1]  # lines 1...(n-1) if multi-line
                cur_split = text.rsplit('\n', 1)    # lines 0...(n-2), line n-1
                if (
                    cur_split[0] == prev_tail or    # scroll 1 line (top and bottom)
                    cur_split[0] == prev_text       # add 1 line (bottom)
                    ):
                    n_scroll += 1
                    if len(cur_split) == 2:
                        yield cur_split[1]
                    # else yield nothing
                else:
                    yield text
            prev_end = end
            prev_text = text

        self.n_cues = n_cues
        self.n_empty = n_empty
        self.n_repeat = n_repeat
        self.n_scroll = n_scroll
        self.n_delay = n_delay

    def __call__(
        self, vtt_lines: Iterable[str],
        delays_as_empty: bool=True,
        normalize_whitespace: bool = True,
        add_new_lines: bool = False,
        ) -> Iterator[str]:
        '''
        Normalizes whitespace by default to single ' ' (ASCII space), affecting
        most importantly sequences of new lines, tabs, and full-width (Japanese) spaces.
        '''
        clean = self.cues2clean_text(
            vtt2cues(vtt_lines),
            delays_as_empty=delays_as_empty
            )

        if normalize_whitespace:
            if add_new_lines:
                def postprocess(s: str) -> str:
                    return sub_space(' ', s) + '\n'
            else:
                def postprocess(s: str) -> str:
                    return sub_space(' ', s)
        elif add_new_lines:
            def postprocess(s: str) -> str:
                return s + '\n'
        else:
            return clean

        return map(postprocess, clean)


if __name__ == '__main__':
    for filename in sys.argv[1:]:
        cleaner = VTTCleaner()
        with (
            open(filename) as fin,
            open(filename.removesuffix('.vtt') + '.txt', 'w') as fout
            ):
            print(filename)
            fout.writelines(cleaner(fin, add_new_lines=True))
            print(cleaner)
