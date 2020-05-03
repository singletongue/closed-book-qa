import argparse
import json
from urllib.parse import unquote

from logzero import logger
from tqdm import tqdm


def clean_text(text):
    text = unquote(text)
    text = ' '.join(text.strip().split())
    text = text.strip()
    return text


def process_triviaqa_dataset(dataset_path,
                             min_question_length=0):
    loaded_dataset = json.load(open(dataset_path))

    for item in tqdm(loaded_dataset['Data']):
        if 'Answer' in item:
            answer = item['Answer']
            if answer['Type'] != 'WikipediaEntity':
                continue

            answer_entity = clean_text(answer['MatchedWikiEntityName'])
            assert answer_entity
        else:
            answer_entity = None

        question = clean_text(item['Question'])
        assert question

        if len(question) < min_question_length:
            logger.warning(f'Skipped (too short question): {question}')
            continue

        item = {
            'question_id': item['QuestionId'],
            'text': question,
            'entity': answer_entity,
        }
        yield item


def main(args):
    logger.info('Processing the TriviaQA dataset')
    with open(args.output_file, 'w') as fo:
        for item in process_triviaqa_dataset(
                args.dataset_file,
                min_question_length=args.min_question_length):
            print(json.dumps(item, ensure_ascii=False), file=fo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--min_question_length', type=int, default=1)
    args = parser.parse_args()
    main(args)
