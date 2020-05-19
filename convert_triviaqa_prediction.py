import argparse
import json
import re


regex_disamb = re.compile(r'(.+?)\s+\(.+?\)')


def main(args):
    predictions = dict()
    with open(args.input_file) as f:
        for line in f:
            item = json.loads(line)
            question_id = item['metadata']['question_id']
            predicted_answer = item['top10_labels'][0]

            if args.postprocess_answers:
                predicted_answer = regex_disamb.sub(r'\1', predicted_answer.strip())

            predictions[question_id] = predicted_answer

    with open(args.output_file, 'w') as fo:
        json.dump(predictions, fo, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--postprocess_answers', action='store_true')
    args = parser.parse_args()
    main(args)
