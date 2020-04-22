import argparse
import json
import re

from logzero import logger
from tqdm import tqdm

from tokenization import (NltkSentenceSplitter,
                          SpacySentenceSplitter,
                          BlingfireSentenceSplitter)


def process_wiki_dataset(dataset_path,
                         text_unit='sentence',
                         sent_splitter=None,
                         max_paragraphs=None,
                         min_text_length=20):
    loaded_dataset = json.load(open(dataset_path))

    for page in tqdm(loaded_dataset.values()):
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
                    logger.warning(f'Skipped (too short text): {text}')
                    continue

                item = {
                    'text': text,
                    'entity': page['title'],
                    'text_unit': text_unit,
                    'paragraph_index': pi
                }
                if text_unit == 'sentence':
                    item['sentence_index'] = si

                yield item


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

    logger.info('Processing the Wiki dataset')
    with open(args.output_file, 'w') as fo:
        for item in process_wiki_dataset(args.dataset_file,
                                         text_unit=args.text_unit,
                                         sent_splitter=sent_splitter,
                                         max_paragraphs=args.max_paragraphs,
                                         min_text_length=args.min_text_length):
            print(json.dumps(item, ensure_ascii=False), file=fo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--text_unit', type=str, default='sentence')
    parser.add_argument('--sent_splitter', type=str)
    parser.add_argument('--max_paragraphs', type=int, default=None)
    parser.add_argument('--min_text_length', type=int, default=20)
    args = parser.parse_args()
    main(args)
