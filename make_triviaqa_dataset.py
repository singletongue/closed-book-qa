import argparse
import json
from urllib.parse import unquote

from logzero import logger
from tqdm import tqdm


UNKNOWN_TOKEN = '@@UNKNOWN@@'


def clean_text(text):
    text = unquote(text)
    text = ' '.join(text.strip().split())
    text = text.strip()
    return text


def normalize_entity_token(entity):
    return unquote(entity.strip()).replace(' ', '_')


def process_triviaqa_dataset(dataset_path,
                             entities_set=None,
                             skip_no_entity=False,
                             min_question_length=0):
    loaded_dataset = json.load(open(dataset_path))

    for item in tqdm(loaded_dataset['Data']):
        question_id = item['QuestionId']

        if 'Answer' in item:
            answer = item['Answer']
            if answer ['Type'] == 'WikipediaEntity':
                answer_entity = normalize_entity_token(answer['MatchedWikiEntityName'])
            else:
                if skip_no_entity:
                    logger.warning(f'Question {question_id} is skipped: no answer entity {answer_entity}')
                    continue
                else:
                    answer_entity = UNKNOWN_TOKEN

            if entities_set is not None and answer_entity not in entities_set:
                logger.warning(f'Question {question_id} is skipped: unknown answer entity {answer_entity}')
                continue
        else:
            answer_entity = None

        question = clean_text(item['Question'])
        if len(question) < min_question_length:
            logger.warning(f'Question {question} is skipped: question is too short')
            continue

        item = {
            'question_id': question_id,
            'text': question,
            'entity': answer_entity,
        }
        yield item


def main(args):
    if args.entities_file is not None:
        logger.info('loading entity file')
        entities_set = set(normalize_entity_token(line) for line in tqdm(open(args.entities_file)))
        logger.info(f'number of entities: {len(entities_set)}')
    else:
        entities_set = None

    logger.info('Processing the TriviaQA dataset')
    with open(args.output_file, 'w') as fo:
        for item in process_triviaqa_dataset(args.dataset_file,
                                             entities_set=entities_set,
                                             skip_no_entity=args.skip_no_entity,
                                             min_question_length=args.min_question_length):
            print(json.dumps(item, ensure_ascii=False), file=fo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--entities_file', type=str)
    parser.add_argument('--skip_no_entity', action='store_true')
    parser.add_argument('--min_question_length', type=int, default=1)
    args = parser.parse_args()
    main(args)
