import argparse
import json
import random
from collections import defaultdict

import torch
from logzero import logger
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelWithLMHead


random.seed(0)


def compute_perplexity(text, tokenizer, model, device='cpu'):
    model.to(device)
    input_ids = torch.tensor(tokenizer.encode(text, add_special_tokens=True))
    input_ids = input_ids.to(device)
    try:
        outputs = model(input_ids, labels=input_ids)
    except RuntimeError as e:
        return None

    lm_loss = outputs[0]
    perplexity = torch.exp(lm_loss)
    return perplexity.item()


def main(args):
    logger.info('Initializing the tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name)
    logger.info('Initializing the model')
    model = AutoModelWithLMHead.from_pretrained(args.pretrained_model_name)

    logger.info('Loading the dataset')
    dataset = [json.loads(line) for line in tqdm(open(args.dataset_file))]

    if args.num_samples is not None:
        dataset = random.sample(dataset, min(len(dataset), args.num_samples))

    logger.info('Computing perplexities')
    with open(args.output_file, 'w') as fo:
        for item in tqdm(dataset):
            perplexity = compute_perplexity(item['text'], tokenizer, model,
                                            device=args.device)
            item['perplexity'] = perplexity
            print(json.dumps(item, ensure_ascii=False), file=fo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--pretrained_model_name', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    main(args)
