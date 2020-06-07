import argparse
import json
from collections import defaultdict


def main(args):
    reciprocal_ranks = defaultdict(list)
    accuracies = defaultdict(list)
    oracle_accuracies = defaultdict(list)
    with open(args.input_file) as f:
        for line in f:
            item = json.loads(line)
            rank = item['rank']
            if rank != 'None':
                assert isinstance(rank, int) and rank > 0
                reciprocal_rank = 1.0 / rank
            else:
                reciprocal_rank = 0.0

            reciprocal_ranks['All'].append(reciprocal_rank)
            accuracies['All'].append(int(rank == 1))
            oracle_accuracies['All'].append(int(rank != 'None'))

            if 'sentence_index' in item:
                s_id = item['sentence_index']
                reciprocal_ranks[s_id].append(reciprocal_rank)
                accuracies[s_id].append(int(rank == 1))
                oracle_accuracies[s_id].append(int(rank != 'None'))

    for key in reciprocal_ranks.keys():
        print(f'# {key}')
        acc = sum(accuracies[key]) / len(accuracies[key])
        sup_acc = sum(oracle_accuracies[key]) / len(oracle_accuracies[key])
        mrr = sum(reciprocal_ranks[key]) / len(reciprocal_ranks[key])

        print(f'Acc: {acc:.3f} (upper bound: {sup_acc:.3f})')
        print(f'MRR: {mrr:.3f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    args = parser.parse_args()
    main(args)
