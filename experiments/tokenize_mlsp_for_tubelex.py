import argparse
from datasets import load_dataset, Dataset
from typing import Optional
from collections import defaultdict, Counter
import re
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

from tubelex import add_tokenizer_arg_group, get_tokenizers, nfkc_lower

'''
HF_DATASETS_OFFLINE=1  python experiments/tokenize_mlsp_for_tubelex.py
HF_DATASETS_OFFLINE=1  python experiments/tokenize_mlsp_for_tubelex.py --form lemma
HF_DATASETS_OFFLINE=1  python experiments/tokenize_mlsp_for_tubelex.py --form base --lang ja
'''

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
    'es': 'Spanish'
    }

LANG2DATASET_ID: dict[str, str] = {
    lang: f'{name.lower()}_lcp_labels'
    for lang, name in LANG2FULL_NAME.items()
    }

DATASET_NAME = 'MLSP2024/MLSP2024'


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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--languages', nargs='+', choices=['en', 'es', 'ja'],
        default=['en', 'es', 'ja']
        )
    parser.add_argument('--token', default=None,
                        help='Hugging Face access token')
    add_tokenizer_arg_group(
        parser,
        default_no_filter_tokens=True # Do not replace numbers
        )  # calls add_tagger_arg_group(parser)
    return parser.parse_args()

def main(args: argparse.Namespace):
    dfs = []
    for lang in args.languages:
        tokenize = get_tokenizers(
            lang=lang, tokenization=args.tokenization, full=True, args=args
            )[1]

        def join_tokenize(s):
            return '|'.join(tokenize(s))

        for train in (True, False):
            try:
                dataset = get_mlsp_dataset(lang, train=train)
            except Exception:
                raise Exception(
                    f'Cannot retrieve dataset. Check the above exception, that you '
                    f'have requested access to {DATASET_NAME}, and that you are logged '
                    f'in with the correct token using `huggingface-cli login`. '
                    f'Alternatively you can supply the access token directly using '
                    f'the --token option.'
                    )

            df = dataset.to_pandas()
            df['lang'] = lang
            df['train'] = train
            df['context_tokenized'] = df['context'].apply(join_tokenize)
            df['target_tokenized']  = df['target'].apply(join_tokenize)
            dfs.append(df)
            print('...')

    df = pd.concat(dfs)
    form_suffix = '' if (args.form == 'surface') else f'_{args.form}'
    df.to_csv(f'experiments/mlsp_tokenized{form_suffix}.tsv', sep='\t', index=False)


if __name__ == '__main__':
    main(parse_args())
