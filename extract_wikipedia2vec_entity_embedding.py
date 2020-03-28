import argparse
import bz2

from tqdm import tqdm


ENTITY_MARKER = 'ENTITY/'


def main(args):
    output_lines = []
    with bz2.open(args.input_file, 'rt') as f:
        vocab_size, n_dim = map(int, f.readline().rstrip('\n').split(' '))
        for line in tqdm(f, total=vocab_size):
            if line[:len(ENTITY_MARKER)] == ENTITY_MARKER:
                output_lines.append(line[len(ENTITY_MARKER):])

    with bz2.open(args.output_file, 'wt') as fo:
        print(vocab_size, n_dim, file=fo)
        for line in output_lines:
            fo.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()
    main(args)
