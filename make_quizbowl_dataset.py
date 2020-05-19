import argparse
import json
import re
from html import unescape

from logzero import logger
from tqdm import tqdm

from tokenization import (NltkSentenceSplitter,
                          SpacySentenceSplitter,
                          BlingfireSentenceSplitter)


regex_ftp = re.compile(r'(, |--)?[Ff]or (10|[Tt]en) [Pp]oints(--|,)')


def process_text(text):
    text = regex_ftp.sub('', text)
    text = text.replace('(*)', '')
    text = ' '.join(text.strip().split())
    text = text.strip()
    return text


def normalize_entity_token(entity):
    return unescape(entity.strip()).replace(' ', '_')


def process_quiz_dataset(dataset_path,
                         text_unit='sentence',
                         sent_splitter=None,
                         do_process_text=False,
                         min_text_length=0):
    loaded_dataset = json.load(open(dataset_path))

    for question in tqdm(loaded_dataset['questions']):
        if text_unit == 'question':
            texts = [question['text'].strip()]
        elif text_unit in ('sentence', 'sequence'):
            if sent_splitter is not None:
                sentences = sent_splitter(text)
            else:
                sentences = [question['text'][start:end]
                             for start, end in question['tokenizations']]

            texts = [s.strip() for s in sentences if s.strip()]
            if text_unit == 'sequence':
                texts = [' '.join(texts[:i + 1]) for i in range(len(texts))]
        else:
            raise KeyError(f'Invalid text_unit is specified: {text_unit}')

        if do_process_text:
            texts = [process_text(t) for t in texts if process_text(t)]

        entity = normalize_entity_token(question['page'])

        for i, text in enumerate(texts):
            if len(text) < min_text_length:
                logger.warning(f'Skipped (too short text): {text}')
                continue

            item = {
                'qanta_id': question['qanta_id'],
                'text': text,
                'entity': entity,
                'text_unit': text_unit,
            }
            if text_unit in ('sentence', 'sequence'):
                item['sentence_index'] = i

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

    logger.info('Processing the Quiz dataset')
    with open(args.output_file, 'w') as fo:
        for item in process_quiz_dataset(args.dataset_file,
                                         text_unit=args.text_unit,
                                         sent_splitter=sent_splitter,
                                         do_process_text=args.do_process_text,
                                         min_text_length=args.min_text_length):
            print(json.dumps(item, ensure_ascii=False), file=fo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--text_unit', type=str, default='sentence')
    parser.add_argument('--sent_splitter', type=str)
    parser.add_argument('--do_process_text', action='store_true')
    parser.add_argument('--min_text_length', type=int, default=1)
    args = parser.parse_args()
    main(args)
