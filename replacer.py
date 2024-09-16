from collections import Counter, defaultdict
from collections.abc import Iterator, Callable, Sequence
from typing import Optional, T
import re

# CC descriptions and YouTube censored words:

YT_CENSORED             = '[ __ ]'

RE_CC_DESC_HW   = r'\[(?=[^\[\]]*[\w♪～])[^\[\].?!．。？！]+\]'
RE_CC_DESC_FW   = r'【(?=[^【】]*[\w♪～])[^【】.?!．。？！]+】'
RE_CC_DESC = (
    # May be either:
    # - "[X]" or "【X】", where X doesn't contain the respective type of brackets,
    #   and contains at least one word-forming character, "♪" or "～".
    rf'{RE_CC_DESC_HW}|{RE_CC_DESC_FW}'
    )
RE_CENSORED     = re.escape(YT_CENSORED)

NORMALIZE_CC: dict[int, int] = {
    0x09: 0x005F,  # tab to underscore
    0x0A: 0x005F,  # LF to underscore
    0x0D: 0x005F,  # CR to underscore
    0x20: 0x005F,  # space to underscore
    # 【】 to []:
    0x3010: 0x005B,
    0x3011: 0x005D
    }

# HTTP(S)URLs and INFORMAL WEB ADDRESSES
#
# Accept most IDNA domain names/internationalized URL:

CJ_RANGES = (
    # Based on http://www.localizingjapan.com/blog/2012/01/20/regular-expressions-for-\
    # japanese-text/
    # We give it a little benefit of doubt by including all kanji and even radicals,
    # which still could be indicative of Japanese or Chinese text.
    r'\u3041-\u3096'    # Hiragana
    r'\u30A0-\u30FF'    # Katakana (full-width)
    r'\u3400-\u4DB5\u4E00-\u9FCB\uF900-\uFA6A'  # {Han} (Kanji incl. Chinese-only)
    r'\u2E80-\u2FD5'    # Han/Kanji radicals
    r'\u31F0-\u31FF\u3220-\u3243\u3280-\u337F'  # Misc. Japanese Symbols and Characters
    )


# 1. Common (not all) TLDs:
# Note: We accept either all upper case or all lower case, to avoid false positives,
# e.g. "sentence ends.It starts another sentence"
RE_TLD = (
    r'(?:(?:'
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
    # ensure it's not followed by another alpha character:
    r')(?![A-Za-z])|'
    # Some East Asian internationalized TLDs
    r'(?:中国|中國|香港|澳门|澳門|新加坡|台湾|台灣|网络|公司|日本|コム)'
    # ensure it's not followed by another CJ character:
    rf'(?![{CJ_RANGES}]))'
    )

# 2. Host names (without TLD):
# - can contain letter, digit, hyphen (not underscore),
# - cannot start or end with a hyphen
# \w (perfectly?) matches characters allowed by IDNA allowing different alphabets,
# accents, digits (\W is the negation of \w)
# We allow either a name consisting entirely of CJ characters,
# or not containing any CJ characters (e.g. '日本', 'nihon', but not '日本nihon':

RE_WITH_OR_WO_CJ = rf'(?:[^\W_{CJ_RANGES}]|-)+|[{CJ_RANGES}]+'
RE_HOST = rf'(?!-)(?:{RE_WITH_OR_WO_CJ})(?<!-)'

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
# PAT_ADDRESS = re.compile(
#     r'|'.join((RE_URL, RE_EMAIL, RE_HANDLE))
#     )


NAME2RE = {
    'censored': RE_CENSORED,
    'cc_desc': RE_CC_DESC,
    'web': RE_URL,
    'email': RE_EMAIL,
    'handle': RE_HANDLE
    }
NAME2POS = {
    'censored': 'CENSORED',
    'cc_desc':  'CCDESC',
    'web':      'WEB',
    'email':    'EMAIL',
    'handle':   'HANDLE'
    }
NAMES_CENSORED_CC  = ('censored', 'cc_desc')

RE_IN           = r'|'.join(rf'(?P<{name}>{regex})' for name, regex in NAME2RE.items())
sub_in          = re.compile(RE_IN).sub

# If the placeholder is longer than this, MeCab (always?) breaks it into more tokens,
# which makes replacing it difficult (see retry_if_broken).
# Sometimes tokenizers keep placeholders as one token with the neighbor word, e.g.
# '{word} tubelexplchldr', we resolve that as a third try via retry_if_broken too.
# Also: Stanza (Indonesian lemmatization?) lowercases, so the placeholder needs
# to be lowercase.
PLACEHOLDER     = 'tubelexplchldr'
PLACEHOLDER_SPC = ' tubelexplchldr '
PLACEHOLDER_LSPC = ' tubelexplchldr'
PLACEHOLDER_RSPC = 'tubelexplchldr '
RE_BROKEN_PLACEHOLDER = r' ?'.join(PLACEHOLDER)
sub_placeholder = re.compile(rf'\b{PLACEHOLDER}\b').sub
sub_broken_placeholder = re.compile(rf'\b{RE_BROKEN_PLACEHOLDER}\b').sub


def _find_substr_in_list(
    xs: list[T], y: str, start: int = 0, f: Callable[[T], str] = lambda x: x
    ) -> Optional[slice]:

    for j in range(max(0, start), len(xs)):
        wjk = ''
        for k in range(j, len(xs)):
            wjk += f(xs[k])
            if wjk == y:
                return slice(j, k + 1)
    return None


def _it_split_placeholders_in_tokens(tokens: list[str]) -> Iterator[str]:
    i = 0
    for j, t in enumerate(tokens):
        if PLACEHOLDER not in t:
            continue
        if t.startswith(PLACEHOLDER_RSPC):
            yield from tokens[i:j]
            i = j + 1
            yield PLACEHOLDER
            yield t.removeprefix(PLACEHOLDER_RSPC)
        elif t.endswith(PLACEHOLDER_LSPC):
            yield from tokens[i:j]
            i = j + 1
            yield t.removesuffix(PLACEHOLDER_LSPC)
            yield PLACEHOLDER
    yield from tokens[i:]


def _it_split_placeholders_in_tagged(
    tagged: list[tuple[str, str]]
    ) -> Iterator[list[tuple[str, str]]]:
    i = 0
    for j, t_pos in enumerate(tagged):
        t, pos = t_pos
        if PLACEHOLDER not in t:
            continue
        if t.startswith(PLACEHOLDER_RSPC):
            yield from tagged[i:j]
            i = j + 1
            yield (PLACEHOLDER, pos)
            yield (t.removeprefix(PLACEHOLDER_RSPC), pos)
        elif t.endswith(PLACEHOLDER_LSPC):
            yield from tagged[i:j]
            i = j + 1
            yield (t.removesuffix(PLACEHOLDER_LSPC), pos)
            yield (PLACEHOLDER, pos)
    yield from tagged[i:]


class Replacer:
    '''
    Replace special tokens by placeholders before tokenization, and then by
    final tokens (`out_tokens`) after tokenization.

    Usage on continuous text:

    >>> counter = Counter()
    >>> addresses = defaultdict(list)
    >>> r = Replacer(counter, addresses)
    >>> t = '[ __ ] Email foo@bar.com and @foo_bar. www.foo.com[engine starts]'
    >>> p = r.replace_with_placeholders(t)
    >>> p.replace(PLACEHOLDER, '*')
    ' *  Email  *  and  * .  *  * '
    >>> r.replace_placeholders_with_tokens(p)
    ' [____]  Email  <email>  and  <handle> .  <web>  [engine_starts] '
    >>> r.all_placeholders_replaced()
    True
    >>> counter.total()
    5
    >>> addresses['handle']
    ['@foo_bar']
    >>> addresses['web']
    ['www.foo.com']

    Usage on individual tokens:

    >>> r = Replacer()
    >>> p = r.replace_with_placeholders(t)
    >>> ' '.join(map(r.replace_placeholder_with_token, p.split()))
    '[____] Email <email> and <handle> . <web> [engine_starts]'

    Alphabetical addresses and IDNs in Japanese text:

    >>> t = 'ウェブサイトはfoobar.co.jpまたは フーバー.コム です。このフーバー.コムは無視する。'
    >>> r.replace_placeholders_with_tokens(r.replace_with_placeholders(t))
    'ウェブサイトは <web> または  <web>  です。このフーバー.コムは無視する。'

    Combined alpha/IDN (without surrounding text):

    >>> r.replace_placeholders_with_tokens(r.replace_with_placeholders('foo.日本'))
    ' <web> '

    Several methods have a retry_if_broken parameter: If we cannot replace all
    placeholders, it tries again, searching for placeholders broken by the tokenization
    (MeCab sometimes does that).
    '''

    counter:    Counter[str]
    addresses:  dict[str, list]
    out_tokens: list[str]
    out_pos:    list[str]
    out_idx:    int

    def __init__(self,
                 counter: Optional[Counter] = None,
                 addresses: Optional[dict[str, list]] = None
                 ):
        self.counter        = counter if (counter is not None) else Counter()
        self.addresses      = (addresses if (addresses is not None) else
                               defaultdict(list))
        self.out_tokens     = []
        self.out_pos        = []
        self.out_idx        = 0

    def _repl_in_placeholder(self, m: re.Match) -> str:
        for name, group in m.groupdict().items():
            if group is None:
                continue
            if (name in NAMES_CENSORED_CC):
                out = group.translate(NORMALIZE_CC)
            else:
                self.addresses[name].append(group)
                out = f'<{name}>'
            self.out_tokens.append(out)
            self.out_pos.append(NAME2POS[name])
            self.counter[name] += 1
            break   # Only one non-None group
        else:
            raise Exception(
                f'repl_in_placeholder called without any matched group: {m.groupdict()}'
                )
        return PLACEHOLDER_SPC

    def _repl_placeholder_out(self, _m: re.Match) -> str:
        out = self.out_tokens[self.out_idx]
        self.out_idx += 1
        return out

    def _it_replace_broken_in_tokens(self, tokens: Sequence[str]) -> Iterator[str]:
        i   = 0
        while (s := _find_substr_in_list(tokens, PLACEHOLDER, i)):
            yield from tokens[i:s.start]
            yield self.out_tokens[self.out_idx]
            self.out_idx += 1
            i = s.stop
        yield from tokens[i:]

    def _it_replace_broken_in_tagged(
        self, tagged: Sequence[tuple[str, str]]
        ) -> list[tuple[str, str]]:
        i   = 0
        while (s := _find_substr_in_list(tagged, PLACEHOLDER, i, f=lambda tp: tp[0])):
            yield from tagged[i:s.start]
            idx = self.out_idx
            yield (self.out_tokens[idx], self.out_pos[idx])
            self.out_idx += 1
            i = s.stop
        yield from tagged[i:]

    def replace_with_placeholders(self, s: str) -> str:
        return sub_in(self._repl_in_placeholder, s)

    def replace_placeholders_with_tokens(
        self, s: str, retry_if_broken: bool = False
        ) -> str:
        '''
        In-text replacement.
        retry_if_broken: If we cannot replace all placeholders, try again, searching
        for placeholders broken by the tokenization (MeCab sometimes does that).
        '''
        if not self.out_tokens:
            return s
        assert not (retry_if_broken and self.out_idx)
        repl_s = sub_placeholder(self._repl_placeholder_out, s)
        if retry_if_broken and not self.all_placeholders_replaced():
            self.out_idx = 0
            repl_s = sub_broken_placeholder(self._repl_placeholder_out, s)
        return repl_s

    def replace_placeholder_with_token(self, t: str) -> str:
        '''
        Single token replacement.
        '''
        if t != PLACEHOLDER:
            return t
        out = self.out_tokens[self.out_idx]
        self.out_idx += 1
        return out

    def replace_tagged_placeholder_with_token(
        self, t_pos: tuple[str, str]
        ) -> tuple[str, str]:
        '''
        Single tagged token replacement.
        '''
        t, _ = t_pos
        if t != PLACEHOLDER:
            return t_pos
        idx = self.out_idx
        out = self.out_tokens[idx]
        pos = self.out_pos[idx]
        self.out_idx += 1
        return (out, pos)

    def replace_in_tokens(
        self, tokens: Sequence[str], retry_if_broken: bool = False
        ) -> list[str]:
        if not self.out_tokens:
            return tokens
        assert not (retry_if_broken and self.out_idx)
        repl_tokens = [self.replace_placeholder_with_token(t) for t in tokens]
        if retry_if_broken and not self.all_placeholders_replaced():
            assert isinstance(tokens, Sequence)
            self.out_idx = 0
            repl_tokens = list(self._it_replace_broken_in_tokens(tokens))
            if not self.all_placeholders_replaced():
                self.out_idx = 0
                tokens = list(_it_split_placeholders_in_tokens(tokens))
                repl_tokens = list(self._it_replace_broken_in_tokens(tokens))
        return repl_tokens

    def replace_in_tagged(self,
                          tagged: Sequence[tuple[str, str]],
                          retry_if_broken: bool = False) -> list[tuple[str, str]]:
        if not self.out_tokens:
            return tagged
        assert not (retry_if_broken and self.out_idx)
        repl_tagged = [self.replace_tagged_placeholder_with_token(tp) for tp in tagged]
        if retry_if_broken and not self.all_placeholders_replaced():
            assert isinstance(tagged, Sequence)
            self.out_idx = 0
            repl_tagged = list(self._it_replace_broken_in_tagged(tagged))
            if not self.all_placeholders_replaced():
                self.out_idx = 0
                tagged = list(_it_split_placeholders_in_tagged(tagged))
                repl_tagged = list(self._it_replace_broken_in_tagged(tagged))
        return repl_tagged

    def all_placeholders_replaced(self) -> bool:
        return self.out_idx == len(self.out_tokens)


if __name__ == '__main__':
    import doctest
    doctest.testmod()

# TODO iterations work the best
#     from timeit import timeit
#     s0 = ' '.join((['adasd']*10 + ['www.apple.com'])*1000)
#     r = Replacer()
#     sp = r.replace_with_placeholders(s0)
#     spb = sp.replace('tubelexplchldr', 't u b e lexplchldr')
# .split(' ')
#     print(timeit('r.out_idx=0; r.replace_in_tokens(spb, retry_if_broken=True)',
# globals=globals(), number=1))
#     assert r.all_placeholders_replaced(), r.out_idx
#     print(timeit('r.out_idx=0; r.test_replace_in_tokens(spb, retry_if_broken=True)',
# globals=globals(), number=1))
#     assert r.all_placeholders_replaced(), r.out_idx
