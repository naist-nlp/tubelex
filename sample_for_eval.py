import os
from os.path import exists, splitext
import sys
import argparse
import re
from typing import Optional
import pandas as pd
from vtt import vtt2cues
from freq_utils import Storage
from tubelex import (
    SUBLIST_PATH_FMT,
    UNIQUE_PATH_FMT,
    DATA_SUFFIX,
    SUB_SUFFIX,
    DATA_PATH_FMT,
    get_files_contents,
    dir_files,
    filter_dir_files,
    PAT_INVALID_TAG
    )
import numpy as np
from sklearn.model_selection import train_test_split
from yt_dlp import YoutubeDL


N_TIMES = 3
TIME_SHIFT = 0
INFO_PATH_FMT = 'corpus/info-%s.csv'
EVAL_PATH_FMT = 'corpus/eval-%s.csv'
EVAL_SAMPLE_DIR_PATH_FMT = 'corpus/eval-%s-sample'


# YouTube video URL
def make_video_url(videoid: str) -> str:
    return f'https://www.youtube.com/watch?v={videoid}'


def make_embed_url(videoid: str, start: int, lang: str) -> str:
    start = int(max(0, start + TIME_SHIFT))
    return (
        f'https://www.youtube-nocookie.com/embed/{videoid}?'
        f'cc_load_policy=1&cc_lang_pref={lang}&'
        f'start={start}'
        )


def videoid_ok(videoid: str, lang: str) -> bool:
    ydl_opts = {
        'extract_flat': 'discard_in_playlist',
        'extractor_args': {'youtube': {'player_client': ['web']}},
        'fragment_retries': 10,
        'ignoreerrors': 'only_download',
        # 'listsubtitles': True, --- not necessary, only affects stdout
        'postprocessors': [{'key': 'FFmpegConcat',
                            'only_multi_video': True,
                            'when': 'playlist'}],
        'retries': 10,
        'skip_download': True,
        'subtitleslangs': [lang]
        }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(make_video_url(videoid), download=False)

    return info is not None


def duration2duration_class(d):
    # 0: [0, 3min)
    # 1: [3min, 10min)
    # 2: [10min, inf)
    return 0 + (d >= 180) + (d >= 600)


def remove_invalid_tags(s: str) -> str:
    return PAT_INVALID_TAG.sub('', s)


def cue_text_prefix(text: str) -> Optional[str]:
    # Take first one or two words/first <=16 chars:
    m = re.search(
        r'\w{1,6}\W{1,2}\w{1,6}|\w+',
        text
        )
    return (
        m.group()[:16] if (m is not None) else
        None
        )


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(
        'Sample videos'
        ))
    parser.add_argument(
        '--language', type=str, default='ja',
        help='Language (2-letter code, default: ja).'
        )
    parser.add_argument(
        '--identifier', type=str, default=None,
        help='Identifier (instead of language) for corpus files.'
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
        '--size', '-n', type=int, default=100,
        help='Number of videos to sample'
        )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
        )
    parser.add_argument(
        '--cached-info', action='store_true',
        help='Used cached info file'
        )
    Storage.add_arg_group(parser, 'Compression options', zip_suffix=True)
    return parser.parse_args()


def main() -> None:
    args = parse()
    lang = args.language
    identifier = args.identifier or lang

    data_path = args.data or (DATA_PATH_FMT % lang)
    info_path = INFO_PATH_FMT % identifier

    if args.cached_info and exists(info_path):
        sys.stderr.write('Reading cached subtitles info.\n')
        subtitles = pd.read_csv(
            info_path,
            index_col='videoid',
            na_filter=False  # keep empty channelids as empty strings
            )
    else:
        sys.stderr.write('Writing subtitles info.\n')
        storage = Storage.from_args(args)
        list_path = args.list or (SUBLIST_PATH_FMT % (lang, lang))
        all_subtitles = pd.read_csv(
            list_path,
            index_col='videoid',
            na_filter=False  # keep empty channelids as empty strings
            )

        with get_files_contents(
            UNIQUE_PATH_FMT % identifier, storage
            ) as files_contents:

            files, __ = files_contents
            unique_videoids = [file.removesuffix(DATA_SUFFIX) for file in files]

        print(len(all_subtitles))

        subtitles = all_subtitles.loc[unique_videoids]

        subtitles.to_csv(INFO_PATH_FMT % identifier)

    print(f'Total subtitles: {len(subtitles)}')

    subtitles['duration_class'] = duration2duration_class(subtitles['duration'])

    np.random.seed(args.seed)

    all_available = False

    while not all_available:
        print()
        print('Checking if videos online...')
        _, sample = train_test_split(
            subtitles, test_size=args.size,
            stratify=(
                subtitles['duration_class'] if (args.size <= 10) else
                subtitles[['categories', 'duration_class']]
                )
            )

        videoids = set(sample.index)  # Faster than index

        all_available = all(videoid_ok(vid, lang) for vid in videoids)

    print()
    print('All videos available.')

    # Now, peek into the subtitles and pick random times:
    print('Getting VTT files...')
    dfs = list(
        filter_dir_files(dir_files(data_path, suffix=SUB_SUFFIX), videoids)
        )

    print('Getting cleaned files...')
    with get_files_contents(
        UNIQUE_PATH_FMT % identifier, storage,
        filenames=(sample.index + DATA_SUFFIX)
        ) as files_contents:

        files, iter_contents = files_contents
        videoid2clean_text = {
            file.removesuffix(DATA_SUFFIX): text
            for file, text in zip(files, iter_contents())
            }

    sample_dir = EVAL_SAMPLE_DIR_PATH_FMT % identifier
    os.makedirs(sample_dir, exist_ok=True)

    print('Reading time info from VTT...')
    for directory, file in dfs:
        videoid = splitext(file)[0]
        clean_text = videoid2clean_text[videoid]
        with open(os.path.join(sample_dir, videoid + DATA_SUFFIX), 'w') as f:
            f.write(clean_text)

        with open(os.path.join(directory, file)) as f:
            # Filter cues that seem to correspond to something in the clean text:
            cue_time_texts = [
                (time, cue_text)
                for time, _, raw_text in vtt2cues(f)
                for cue_text in (remove_invalid_tags(raw_text),)
                for prefix in (cue_text_prefix(cue_text),)
                if prefix is not None and (prefix in clean_text)
                ]
            assert len(cue_time_texts) >= N_TIMES, (videoid, len(cue_time_texts))
            # Make a random sample:
            indices = np.random.choice(
                np.arange(len(cue_time_texts)), size=N_TIMES, replace=False
                )
            choice_cues = sorted([cue_time_texts[i] for i in indices])
        sample.loc[videoid, 'url'] = make_video_url(videoid)
        for i, (t, text) in enumerate(choice_cues):
            sample.loc[videoid, f'time_{i+1}'] = t
            sample.loc[videoid, f'url_{i+1}'] = make_embed_url(videoid, t, lang)
            sample.loc[videoid, f'prefix_{i+1}'] = cue_text_prefix(text)

    sample.to_csv(EVAL_PATH_FMT % identifier)

    print('Done.')


if __name__ == '__main__':
    main()
