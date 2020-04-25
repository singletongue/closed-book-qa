import json
import logging
import re
from copy import deepcopy
from typing import Dict, List, Iterable, Optional, Union, Any

from overrides import overrides

import numpy
import torch
from torch import nn
from torch.nn import functional as F

from allennlp.common.file_utils import cached_path
from allennlp.common.util import pad_sequence_to_length, JsonDict
from allennlp.data import Instance, DatasetReader, TextFieldTensors, Vocabulary
from allennlp.data.fields import TextField, LabelField, MetadataField
from allennlp.data.tokenizers import Token, Tokenizer, PretrainedTransformerTokenizer
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.token_indexers.token_indexer import IndexedTokenList
from allennlp.models import Model
from allennlp.modules import (TokenEmbedder, TextFieldEmbedder,
                              Seq2SeqEncoder, Seq2VecEncoder, FeedForward)
from allennlp.modules.token_embedders import Embedding, PretrainedTransformerEmbedder
from allennlp.nn.util import get_text_field_mask, masked_mean, masked_max
from allennlp.training.optimizers import Optimizer
from allennlp.training.metrics import Metric, CategoricalAccuracy
from allennlp.predictors.predictor import Predictor
from transformers import AdamW, XLNetConfig


logger = logging.getLogger(__name__)


Optimizer.register('pretrained_transformer_adam_w')(AdamW)


@DatasetReader.register('text_entity')
class TextEntityDatasetReader(DatasetReader):
    def __init__(self,
                 tokenizer: Optional[Tokenizer],
                 token_indexers: Dict[str, TokenIndexer],
                 do_mask_entity_mentions: bool = False,
                 mask_token: Optional[str] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy=lazy)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers
        self.do_mask_entity_mentions = do_mask_entity_mentions
        self.mask_token = mask_token

    @overrides
    def text_to_instance(self,
                         text: str,
                         entity: Optional[str] = None,
                         metadata: Dict[str, Any] = None) -> Instance:
        fields = dict()

        if self.do_mask_entity_mentions:
            assert self.mask_token is not None
            assert entity is not None
            for mention in entity.replace('_', ' ').split():
                if isinstance(self.tokenizer, PretrainedTransformerTokenizer):
                    mask_length = len(
                        self.tokenizer.intra_word_tokenize([mention])[0][0])
                else:
                    mask_length = len(self.tokenizer.tokenize(mention))

                text = text.replace(mention, self.mask_token * mask_length)
                text = text.replace(mention.lower(), self.mask_token * mask_length)

        if self.tokenizer is not None:
            tokens = self.tokenizer.tokenize(text)
            fields['text'] = TextField(tokens, self.token_indexers)

        if entity is not None:
            fields['entity'] = LabelField(entity, 'entities')
        else:
            fields['entity'] = LabelField(-1, 'entities', skip_indexing=True)

        if metadata is not None:
            fields['metadata'] = MetadataField(metadata)

        return Instance(fields)

    @overrides
    def _read(self,
              file_path: str) -> Iterable[Instance]:
        file_path = cached_path(file_path)
        logger.info(f'Reading a dataset file at {file_path}')
        with open(file_path) as dataset_file:
            for i, line in enumerate(dataset_file):
                item = json.loads(line)

                instance = self.text_to_instance(
                    text=item['text'],
                    entity=item['entity'],
                    metadata=item
                )
                if i < 20:
                    logger.info(f'Example: {instance}')

                yield instance


@TokenEmbedder.register('patched_pretrained_transformer')
class PatchedPretrainedTransformerEmbedder(PretrainedTransformerEmbedder):
    def __init__(self, model_name: str, max_length: int = None) -> None:
        super().__init__(model_name, max_length=max_length)

    @overrides
    def _number_of_token_type_embeddings(self):
        config = self.transformer_model.config
        if isinstance(config, XLNetConfig):
            return 3
        elif hasattr(config, "type_vocab_size"):
            return config.type_vocab_size
        else:
            return 0


@Model.register('quizbowl_guesser')
class QuizbowlGuesser(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 seq2vec_encoder: Seq2VecEncoder,
                 seq2seq_encoder: Optional[Seq2SeqEncoder] = None,
                 feedforward: Optional[FeedForward] = None,
                 dropout: float = 0.0,
                 do_batch_norm: bool = False) -> None:
        super(QuizbowlGuesser, self).__init__(vocab)
        self.text_field_embedder = text_field_embedder
        self.seq2seq_encoder = seq2seq_encoder
        self.seq2vec_encoder = seq2vec_encoder
        self.feedforward = feedforward

        if self.feedforward is not None:
            entity_embedding_dim = self.feedforward.get_output_dim()
        else:
            entity_embedding_dim = self.seq2vec_encoder.get_output_dim()

        num_entities = vocab.get_vocab_size('entities')
        self.entity_embedder = Embedding(entity_embedding_dim, num_entities,
                                         vocab_namespace='entities')

        self.dropout = nn.Dropout(dropout)
        if do_batch_norm:
            self.batch_norm = nn.BatchNorm1d(num_entities)

        self.accuracy = CategoricalAccuracy(top_k=1)
        self.mean_reciprocal_rank = MeanReciprocalRank()

    def forward(self,
                text: TextFieldTensors,
                entity: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        output_dict = dict()

        mask = get_text_field_mask(text)
        encoded_text = self.text_field_embedder(text)

        if self.seq2seq_encoder is not None:
            encoded_text = self.seq2seq_encoder(encoded_text, mask=mask)

        encoded_text = self.seq2vec_encoder(encoded_text, mask=mask)
        encoded_text = self.dropout(encoded_text)

        if self.feedforward is not None:
            encoded_text = self.feedforward(encoded_text)

        logits = F.linear(encoded_text, self.entity_embedder.weight)
        if hasattr(self, 'batch_norm'):
            logits = self.batch_norm(logits)

        loss = F.cross_entropy(logits, entity)

        output_dict['loss'] = loss

        if not self.training:
            log_probs = F.log_softmax(logits, dim=1)
            output_dict['log_probs'] = log_probs
            self.accuracy(log_probs, entity)
            self.mean_reciprocal_rank(log_probs, entity)

        output_dict['metadata'] = metadata

        return output_dict

    @overrides
    def make_output_human_readable(
                self,
                output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        top10_labels_list = []
        rank_list = []

        predictions = output_dict['log_probs']
        if predictions.dim() == 2:
            prediction_list = [predictions[i] for i in range(predictions.shape[0])]
        else:
            prediction_list = [predictions]

        label_list = [metadata['entity'] for metadata in output_dict['metadata']]

        for prediction, label in zip(prediction_list, label_list):
            top_label_ids = prediction.argsort(dim=-1, descending=True)
            top_labels = [self.vocab.get_index_to_token_vocabulary('entities')[i]
                          for i in top_label_ids]

            if label in top_labels:
                rank = top_labels.index(label) + 1
            else:
                rank = None

            top10_labels_list.append(top_labels[:10])
            rank_list.append(rank)

        output_dict['top10_labels'] = top10_labels_list
        output_dict['rank'] = rank_list
        del output_dict['log_probs']  # delete it since it's really big

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        if self.training:
            metrics = dict()
        else:
            metrics = {
                'acc': self.accuracy.get_metric(reset),
                'mrr': self.mean_reciprocal_rank.get_metric(reset)
            }

        return metrics


@Predictor.register('quizbowl')
class QuizbowlPredictor(Predictor):
    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({'sentence': sentence})

    @overrides
    def _json_to_instance(self,
                          json_dict: JsonDict) -> Instance:
        text = json_dict['text']
        entity = json_dict.get('entity')
        metadata = json_dict.get('metadata')
        return self._dataset_reader.text_to_instance(text, entity, metadata)

    @overrides
    def predictions_to_labeled_instances(self,
            instance: Instance,
            outputs: Dict[str, numpy.ndarray]) -> List[Instance]:
        new_instance = deepcopy(instance)
        pred_entity = numpy.argmax(outputs['log_probs'])
        new_instance.add_field('entity',
            LabelField(int(pred_entity), 'entities', skip_indexing=True))

        return new_instance


@Metric.register('mean_reciprocal_rank')
class MeanReciprocalRank(Metric):
    def __init__(self) -> None:
        self.summed_reciprocal_ranks = 0.0
        self.total_count = 0

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None) -> None:
        predictions, gold_labels, mask = \
            self.detach_tensors(predictions, gold_labels, mask)

        num_classes = predictions.size(-1)

        predictions = predictions.view((-1, num_classes))
        gold_labels = gold_labels.view(-1).long()

        predicted_ids = predictions.argsort(-1, descending=True)
        correct = predicted_ids.eq(gold_labels.unsqueeze(-1)).float()
        reciprocals = torch.arange(1, num_classes + 1,
                                   device=correct.device).float().reciprocal()
        reciprocal_ranks = torch.matmul(correct, reciprocals)

        if mask is not None:
            self.summed_reciprocal_ranks += reciprocal_ranks[mask].sum().item()
            self.total_count += mask.sum().item()
        else:
            self.summed_reciprocal_ranks += reciprocal_ranks.sum().item()
            self.total_count += gold_labels.numel()

    def get_metric(self,
                   reset: bool = False) -> float:
        if self.total_count > 0.0:
            mean_reciprocal_rank = self.summed_reciprocal_ranks / self.total_count
        else:
            mean_reciprocal_rank = 0.0

        if reset:
            self.reset()

        return mean_reciprocal_rank

    @overrides
    def reset(self) -> None:
        self.summed_reciprocal_ranks = 0.0
        self.total_count = 0
