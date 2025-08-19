# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# This file is based on https://github.com/cl-tohoku/bert-japanese/ #
# but contains modifications/additions by Adam Nohejl.              #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Copyright 2023 Masatoshi Suzuki (@singletongue)
# Copyright 2023 Adam Nohejl (@adno) -- modifications of the original code
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import gzip
import json
import re
from itertools import islice
from tqdm import tqdm
from tubelex import get_write_file
from freq_utils import Storage


# def filter_text(text):
#     # filter out text containing equations
#     return (r'\displaystyle' not in text)


def preprocess_text(text, title=None):
    # NFKC is sensible, but not necessary for this task,
    # we also prefer to keep wide punctuation, e.g. "（）"
    # text = unicodedata.normalize("NFKC", text)

    # TODO Some of this is for Japanese only, but the most imprtant parts do not depend
    # on language (it's not clear to me where the bracketed tags appear, e.g. [要出典]
    # as opposed to {{要出典}}.

    # remove invisible characters except '\n'
    text = "".join(c for c in text if c.isprintable() or c == '\n')

    # remove templates
    text = re.sub(r"\[\d+?\]|\[要.+?\]|\{\{+[^{}]+?\}\}+", "", text)

    # remove navigation
    if title is not None:
        text = re.sub(r"^.+? \> " + re.escape(title), "", text)

    # remove footnotes
    text = re.sub(r" \^ .+", "", text)
    # remove annotations
    text = re.sub(r"\[(要出典|リンク切れ|.+?\?)\]", "", text)

    # Convert whitespace except '\n' to ' ',
    # and remove empty lines/coalesce mutliple \n:
    text = '\n'.join(
        filter(None, (re.sub(r"\s+", " ", line).strip() for line in text.split('\n')))
        )
    return text


def main(args):
    language = args.language
    min_length = args.min_length
    storage = Storage.from_args(args)

    n = 0
    with (
        gzip.open(args.input_file, "rt") as f,
        get_write_file(f'corpus-wiki/wiki-{language}', storage) as write_file
        ):
        for line in tqdm(islice(f, args.limit)):
            item = json.loads(line)
            if "index" in item:
                continue
            title = item["title"]
            text = item["text"]
            text = preprocess_text(text, title=title)

            # Only output the text:

            if len(text) < min_length:
                continue

            n += 1
            write_file(f'{n:09d}.txt', text)

    print(f'Total {n} files written.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language")
    parser.add_argument('input_file')
    # Similar to the min. 3 lines limit for TUBELEX
    parser.add_argument('--min-length', type=int, default=256)
    parser.add_argument('--limit', '-n', type=int)
    Storage.add_arg_group(parser, 'Compression options', zip_suffix=True)
    args = parser.parse_args()
    main(args)
