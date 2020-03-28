import argparse
import bz2

from tqdm import tqdm


ENTITY_MARKER = 'ENTITY/'


def main(args):
    with bz2.open(args.input_file, 'rt') as f, \
         bz2.open(args.output_file, 'wt') as fo:
        vocab_size, _ = map(int, f.readline().rstrip('\n').split(' '))
        for line in tqdm(f, total=vocab_size):
            if line[:len(ENTITY_MARKER)] == ENTITY_MARKER:
                fo.write(line[len(ENTITY_MARKER):])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()
    main(args)
