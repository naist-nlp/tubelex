import os
# from os.path import exists, splitext
# import sys
import argparse
# from typing import Optional
import pandas as pd
from pandas.io.formats.style import Styler
from latex_utils import colapse_latex_table_header

# from vtt import vtt2cues
# from freq_utils import Storage
# from tubelex import (
#     SUBLIST_PATH_FMT,
#     UNIQUE_PATH_FMT,
#     DATA_SUFFIX,
#     SUB_SUFFIX,
#     DATA_PATH_FMT,
#     get_files_contents,
#     dir_files,
#     filter_dir_files,
#     PAT_INVALID_TAG
#     )
# from sklearn.model_selection import train_test_split
# from yt_dlp import YoutubeDL

# import numpy as np
# import matplotlib.pyplot as plt
#
#
# def bar_plot(
#     data: pd.DataFrame | dict,
#     title=None,
#     format_index=str,
#     width: Optional[float] = None,
#     stacked=False,
#     fmt='%.02f',
#     show_zero=False,
#     horizontal=False,
#     bar_label_color='white',
#     savefig: Optional[str] = None,
#     savefig_kwargs: dict = dict(bbox_inches='tight'),
#     legend_loc: str = 'best',
#     ) -> None:
#     n = len(data) if isinstance(data, pd.DataFrame) else len(data[data.keys()[0]])
#     if stacked:
#         fig, ax = plt.subplots()
#         bottom = np.zeros(n)
#         if width is None:
#             width = 0.5
#     else:
#         m = len(data.columns) if isinstance(data, pd.DataFrame) else len(data.keys())
#         if width is None:
#             width = 1 / (m + 1)
#         else:
#             width /= m
#         fig, ax = plt.subplots(layout='constrained')
#         x = np.arange(n)  # the label locations
#         left = (m - 1) if horizontal else 0
#
#     bottom_labels = list(map(format_index, data.index))
#     if horizontal:
#         # For horizontal, we are drawing from bottom to top => reverse the order to
#         # preserve it top down:
#         bottom_labels = list(reversed(bottom_labels))
#     ax_bar = ax.barh if horizontal else ax.bar
#
#     if show_zero:
#         def fmt_label(d: float) -> str:
#             return fmt%d
#     else:
#         def fmt_label(d: float) -> str:
#             return (fmt%d) if d else ''
#
#     #plt.rcParams['font.size'] = 3
#     for col, col_data in data.items():
#         if horizontal:
#             col_data = (
#                 col_data.iloc[::-1] if isinstance(col_data, pd.Series) else
#                 list(reversed(col_data))
#                 )
#         if stacked:
#             rects = ax_bar(
#                 bottom_labels,
#                 col_data, width, bottom, label=col
#                 )
#             bottom += col_data
#         else:
#             offset = width * left
#             rects = ax_bar(x + offset, col_data, width, label=col)
#             # ax.bar_label(rects, fmt=fmt, label_type='center')
#             left += (-1 if horizontal else 1)
#         ax.bar_label(
#             rects, [fmt_label(d) for d in col_data], label_type='center',
#             color=bar_label_color
#             )
#     #plt.rcParams['font.size'] = 12
#     if not stacked:
#         ax_set_ticks = ax.set_yticks if horizontal else ax.set_xticks
#         ax_set_ticks(x + width, bottom_labels)
# #     if title is not None:
# #         ax.set_title(title)
#     ax.legend(loc=legend_loc)   # can set fontsize=
#     if savefig:
#         # TODO plt.tight_layout()
#         plt.savefig(savefig, **savefig_kwargs)
#     else:
#         plt.show()


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(
        'Analyze human evaluation'
        ))
    parser.add_argument(
        '--data', type=str, default='eval',
        help='Directory with human evaluation CSV files (2-letter code, default: eval).'
        )
    return parser.parse_args()


DATA_FILE_FMT  = 'Tubelex sample evaluation 20240725 - %s.csv'
LANGS = ['en', 'zh', 'ja', 'es', 'id']

ANN_COLS = [
    'annotation 1',
    'annotation 2',
    'annotation 3'
    ]

TXT_COLS = [
    'subtitle 1',
    'subtitle 2',
    'subtitle 3'
    ]


ANN_LABELS = [
    # Seriosly broken
    # '',                                             # ideally zero
    # 'Video not accessible',                         # ideally zero
    # 'No subtitle',                                  # ideally zero
    # 'Subtitle not in expected language',            # ideally zero

    # OK
    'OK: Speech',
    'OK: Song',

    # Small problems
    # all are in the expected language, but:
    'Audio description',                             # text
    'DIfferent subtitle and audio language',        # => translated
    'No speech in audio',                           # text
    'Synthesised speech in audio'                  # text
    ]

# Order and descriptions for the final output
ANN2DESC = {
    'OK: Speech': 'Subtitles match speech in the target language',
    'OK: Song': 'Subtitles match song in the target language',
    'Audio description': 'Audio description',
    'No speech in audio': 'No speech or song',
    'Synthesised speech in audio': 'Synthesized speech',
    'DIfferent subtitle and audio language': 'Audio language differs',
    'Subtitle not in expected language': 'Subtitle language differs'
    }

LANG2DESC = {
    'zh': 'Chinese',
    'en': 'English',
    'id': 'Indonesian',
    'ja': 'Japanese',
    'es': 'Spanish'
    }

SAMPLE_SIZE = 100


def main() -> None:
    args = parse()
    data_dir = args.data

    props_by_lang = pd.DataFrame()

    for lang in LANGS:
        data_path = os.path.join(
            data_dir,
            DATA_FILE_FMT % lang
            )

        df = pd.read_csv(data_path, na_filter=False)    # read empty strings as ''

        # Checks:
        if len(df) != SAMPLE_SIZE:
            raise Exception(
                f'Unexpected number of rows in {data_path}: {len(df)}'
                )
        for col in ANN_COLS:
            ann = df[col]
            unexpected_values = set(ann.unique()).difference(ANN_LABELS)
            if unexpected_values:
                raise Exception(
                    f'Unexpected values in {data_path}:{col}: {unexpected_values}'
                    )
            unexpected_values = set(ann.unique()).difference(ANN_LABELS)

        # Combine annotation cols
        combined_ann = pd.Series(map(
            lambda x: ','.join(sorted(set(x))),
            df[ANN_COLS].itertuples(index=False)
            ))

        concat_ann = pd.concat([df[c] for c in ANN_COLS])

        print()
        print('#######################')
        print()
        print(f'Language: {lang}')
        print('Combined:')
        print(combined_ann.value_counts(normalize=True))
        print()
        print('Concatenated:')
        concat_props = concat_ann.value_counts(normalize=True)
        print(concat_props)

        props_by_lang[lang] = concat_props

    props_by_lang = props_by_lang.rename(
        index=ANN2DESC, columns=LANG2DESC
        ).reindex(
        ANN2DESC.values(), axis=0
        ).reindex(
        LANG2DESC.values(), axis=1
        )

    props_by_lang.index.name = 'Evaluation Label'

    # props_by_lang.fillna(0.0, inplace=True)
    # props_by_lang = props_by_lang.transpose()

    print(props_by_lang)

    out_path = os.path.join(
        args.data,
        'aggregate.tsv'
        )
    props_by_lang.to_csv(out_path, sep='\t')

    # Small quantities are unreadable
    # bar_plot(
    #     props_by_lang,
    #     title='Human evaluation',
    #     horizontal=False, stacked=True,
    #     savefig='figures/human_eval_by_lang.pdf'
    #     )

    tex_path = os.path.join(
        args.data,
        'human_eval.tex'
        )
    with open(tex_path, 'w') as f:

        s = Styler(
            props_by_lang,
            formatter=lambda x: rf'{x*100:.2f}\%',
            na_rep='---'
            )
        tex = colapse_latex_table_header(s.to_latex(
            hrules=True
            ), props_by_lang)
        print()
        print(tex)
        f.write(tex)


if __name__ == '__main__':
    main()
