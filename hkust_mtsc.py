import re

SAMPLE_PATH = (
    'LDC2005T32/hkust_mtsc_p1tr/data/trans/dev/20040617_122939_A003110_B003109.txt'
    )

'''
The documentation or paper (LDC2005T32/hkust_mtsc_p1tr/index.html) does not provide much
guidance about the markup unfortunately

File example:

430.059000 435.460500 B: 这有这个嘛, 什么台民众欲公- 公投否决 ((军购))
11.948500 114.041750 B: {cough}
111.957500 113.844875 A: <noise>
'''

PAT_REMOVE = re.compile(
    r'^[0-9.]+ [0-9.]+ [A-Z]+: |'   # Timestamp
    r'<[^<]+>|'                 # Noise
    r'\{[^{]+\}|'               # Speaker noise
    r'<([A-Za-z]+)>.*</\1>|'    # Foreign language
    r'%'                        # Precedes a filled pause: %呵 %呃 %唔
    # Keep "((", "))" and anything inside, keep  "--", spaces etc.
    )

ENCODING = 'gbk'


def process(text: str) -> str:
    return '\n'.join(
        PAT_REMOVE.sub('', line)
        for line in text.split('\n')
        if line and not line.startswith('#')
        )


if __name__ == '__main__':
    with open(SAMPLE_PATH, 'rt', encoding=ENCODING) as f:
        print(process(f.read()))
