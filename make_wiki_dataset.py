import argparse
import json
import re
from html import unescape

from logzero import logger
from tqdm import tqdm

from tokenization import (NltkSentenceSplitter,
                          SpacySentenceSplitter,
                          BlingfireSentenceSplitter)


def normalize_entity_token(entity):
    return unescape(entity.strip()).replace(' ', '_')


def process_wiki_dataset(dataset_path,
                         text_unit='sentence',
                         sent_splitter=None,
                         title_set=None,
                         max_paragraphs=None,
                         min_text_length=20):
    with open(dataset_path, 'rt') as f:
        matched_title_set = set()
        for line in tqdm(f):
            page = json.loads(line)
            title = normalize_entity_token(page['title'])
            if title_set is not None and title not in title_set:
                continue
            else:
                matched_title_set.add(title)

            paragraphs = re.split(r'\n\n+', page['text'])[1:]
            paragraphs = [p.strip() for p in paragraphs if p.strip()]
            if max_paragraphs is not None:
                paragraphs = paragraphs[:max_paragraphs]

            if text_unit == 'paragraph':
                paragraph_texts = [[p] for p in paragraphs]
            elif text_unit == 'sentence':
                assert sent_splitter is not None
                paragraph_texts = []
                for p in paragraphs:
                    sentences = sent_splitter(p)
                    sentences = [s.strip() for s in sentences if s.strip()]
                    paragraph_texts.append(sentences)
            else:
                raise KeyError(f'Invalid text_unit is specified: {text_unit}')

            for pi, texts in enumerate(paragraph_texts):
                for si, text in enumerate(texts):
                    if len(text) < min_text_length:
                        # logger.warning(f'Skipped (too short text): {text}')
                        continue

                    if '|' in text:  # possibly parsing error by WikiExtractor
                        continue

                    item = {
                        'text': text,
                        'entity': title,
                        'text_unit': text_unit,
                        'paragraph_index': pi
                    }
                    if text_unit == 'sentence':
                        item['sentence_index'] = si

                    yield item

        if title_set is not None:
            logger.info(f'{len(matched_title_set)}/{len(title_set)} titles were matched')
            unmatched_title_set = title_set - matched_title_set
            if len(unmatched_title_set):
                logger.info(f'unmatched titles: ' + ', '.join(unmatched_title_set))


def main(args):
    if args.sent_splitter is not None:
        logger.info('Initializing a sentence splitter')
        if args.sent_splitter == 'nltk':
            sent_splitter = NltkSentenceSplitter()
        elif args.sent_splitter == 'blingfire':
            sent_splitter = BlingfireSentenceSplitter()
        elif args.sent_splitter == 'spacy':
            sent_splitter = SpacySentenceSplitter()
        else:
            raise KeyError(f'Invalid sent_splitter is specified: {sent_splitter}')
    else:
        sent_splitter = None

    if args.title_list_file is not None:
        logger.info('Loading the title list')
        title_set = set(normalize_entity_token(line) for line in tqdm(open(args.title_list_file)))
        logger.info('Loaded {} titles'.format(len(title_set)))
    else:
        title_set = None

    logger.info('Processing the Wiki dataset')
    with open(args.output_file, 'w') as fo:
        for item in process_wiki_dataset(args.dataset_file,
                                         text_unit=args.text_unit,
                                         sent_splitter=sent_splitter,
                                         title_set=title_set,
                                         max_paragraphs=args.max_paragraphs,
                                         min_text_length=args.min_text_length):
            print(json.dumps(item, ensure_ascii=False), file=fo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_file', type=str, required=True)
    parser.add_argument('--title_list_file', type=str)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--text_unit', type=str, default='sentence')
    parser.add_argument('--sent_splitter', type=str)
    parser.add_argument('--max_paragraphs', type=int, default=None)
    parser.add_argument('--min_text_length', type=int, default=20)
    args = parser.parse_args()
    main(args)
