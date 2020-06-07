import os
import argparse

from allennlp.models.archival import archive_model


def main(args):
    archive_file = os.path.join(args.serialization_dir, args.archive_name)
    archive_model(args.serialization_dir, args.weights_name, archive_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--serialization_dir', type=str, required=True)
    parser.add_argument('--weights_name', type=str, required=True)
    parser.add_argument('--archive_name', type=str, required=True)
    args = parser.parse_args()
    main(args)
