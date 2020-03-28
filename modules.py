import re
import json
import logging
from copy import deepcopy
from typing import Dict, List, Iterable, Optional, Any

from overrides import overrides

import numpy
import torch
from torch import nn
from torch.nn import functional as F

from allennlp.common.file_utils import cached_path
from allennlp.common.util import pad_sequence_to_length, JsonDict
from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, LabelField, MetadataField
from allennlp.data.tokenizers import Tokenizer, SentenceSplitter
from allennlp.data.tokenizers.word_splitter import WordSplitter
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules import TokenEmbedder, TextFieldEmbedder, Seq2VecEncoder, Embedding
from allennlp.nn.util import get_text_field_mask, masked_mean, masked_max
from allennlp.training.optimizers import Optimizer
from allennlp.training.metrics import Metric, CategoricalAccuracy
from allennlp.predictors.predictor import Predictor
from transformers import AutoTokenizer, AutoConfig, AutoModel, XLNetTokenizer, AdamW


logger = logging.getLogger(__name__)


Optimizer.register('adam_w')(AdamW)


@DatasetReader.register('quizbowl')
class QuizbowlDatasetReader(DatasetReader):
    def __init__(self,
                 tokenizer: Optional[Tokenizer],
                 token_indexers: Dict[str, TokenIndexer],
                 sentence_splitter: Optional[SentenceSplitter] = None,
                 use_quiz: bool = True,
                 use_wiki: bool = False,
                 wiki_file_path: Optional[str] = None,
                 text_unit: str = 'sentence',
                 wiki_text_unit: str = 'sentence',
                 num_wiki_paragraphs: Optional[int] = None,
                 do_mask_entity_mentions: bool = False,
                 mask_token: str = '',
                 lazy: bool = False) -> None:
        super().__init__(lazy=lazy)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers
        self.sentence_splitter = sentence_splitter
        self.use_quiz = use_quiz
        self.use_wiki = use_wiki
        self.wiki_file_path = wiki_file_path
        self.text_unit = text_unit
        self.wiki_text_unit = wiki_text_unit
        self.num_wiki_paragraphs = num_wiki_paragraphs
        self.do_mask_entity_mentions = do_mask_entity_mentions
        self.mask_token = mask_token

    @overrides
    def text_to_instance(self,
                         text: str,
                         entity: Optional[str] = None,
                         metadata: Dict[str, Any] = None) -> Instance:
        fields = dict()

        if self.do_mask_entity_mentions:
            assert entity is not None
            for word in entity.replace('_', ' ').split():
                if isinstance(self.tokenizer, TransformerTokenizer):
                    mask_length = len(self.tokenizer.tokenizer.tokenize(word))
                else:
                    mask_length = 1

                assert self.mask_token is not None
                text = text.replace(word, self.mask_token * mask_length)
                text = text.replace(word.lower(), self.mask_token * mask_length)

        if self.tokenizer is not None:
            tokens = self.tokenizer.tokenize(text)
            fields['tokens'] = TextField(tokens, self.token_indexers)

        if entity is not None:
            fields['entity'] = LabelField(entity, 'entities')
        else:
            fields['entity'] = LabelField(-1, 'entities', skip_indexing=True)

        if metadata is not None:
            fields['metadata'] = MetadataField(metadata)

        return Instance(fields)

    def _read_quiz(self, file_path) -> Iterable[Instance]:
        file_path = cached_path(file_path)

        with open(file_path) as dataset_file:
            for question in json.load(dataset_file)['questions']:
                if self.text_unit == 'sentence':
                    if self.sentence_splitter is not None:
                        sentences = \
                            self.sentence_splitter.split_sentences(question['text'])
                    else:
                        sentences = [question['text'][start:end]
                                     for start, end in question['tokenizations']]

                    for i, sentence in enumerate(sentences):
                        sentence = sentence.strip()
                        if len(sentence) > 0:
                            metadata = {
                                'qanta_id': question['qanta_id'],
                                'text': sentence,
                                'entity': question['page'],
                                'text_unit': self.text_unit,
                                'sentence_id': i
                            }
                            instance = self.text_to_instance(
                                text=sentence,
                                entity=question['page'],
                                metadata=metadata
                            )
                            yield instance
                elif self.text_unit == 'question':
                    question_text = question['text'].strip()
                    if len(question_text) > 0:
                        metadata = {
                            'qanta_id': question['qanta_id'],
                            'text': question_text,
                            'entity': question['page'],
                            'text_unit': self.text_unit
                        }
                        instance = self.text_to_instance(
                            text=question_text,
                            entity=question['page'],
                            metadata=metadata
                        )
                        yield instance
                elif self.text_unit == 'sequence':
                    if self.sentence_splitter is not None:
                        sentences = \
                            self.sentence_splitter.split_sentences(question['text'])
                    else:
                        sentences = [question['text'][start:end]
                                     for start, end in question['tokenizations']]

                    sequence = ''
                    for i, sentence in enumerate(sentences):
                        sentence = sentence.strip()
                        if len(sentence) > 0:
                            sequence = sequence + ' ' + sentence
                            metadata = {
                                'qanta_id': question['qanta_id'],
                                'text': sequence,
                                'entity': question['page'],
                                'text_unit': self.text_unit,
                                'sentence_id': i
                            }
                            instance = self.text_to_instance(
                                text=sequence,
                                entity=question['page'],
                                metadata=metadata
                            )
                            yield instance
                else:
                    raise ValueError(f'Invalid text_unit: {self.text_unit}')

    def _read_wiki(self, file_path) -> Iterable[Instance]:
        file_path = cached_path(file_path)

        dataset_json = json.load(open(file_path))
        for page in dataset_json.values():
            entity = page['title']
            paragraphs = re.split(r'\n\n+', page['text'])[1:]
            if self.num_wiki_paragraphs is not None:
                paragraphs = paragraphs[:self.num_wiki_paragraphs]

            if self.wiki_text_unit == 'paragraph':
                for i, paragraph in enumerate(paragraphs):
                    paragraph = paragraph.strip()
                    if len(paragraph) > 0:
                        metadata = {
                            'text': paragraph,
                            'entity': page['title'],
                            'text_unit': self.wiki_text_unit,
                            'paragraph_id': i
                        }
                        instance = self.text_to_instance(
                            text=paragraph,
                            entity=page['title'],
                            metadata=metadata
                        )
                        yield instance
            elif self.text_unit == 'sentence':
                for i, paragraph in enumerate(paragraphs):
                    for j, sentence in enumerate(
                            self.sentence_splitter.split_sentences(paragraph)):
                        sentence = sentence.strip()
                        if len(sentence) > 0:
                            metadata = {
                                'text': sentence,
                                'entity': page['title'],
                                'text_unit': self.wiki_text_unit,
                                'paragraph_id': i,
                                'sentence_id': j
                            }
                            instance = self.text_to_instance(
                                text=sentence,
                                entity=page['title'],
                                metadata=metadata
                            )
                            yield instance
            else:
                raise KeyError(f'Invalid wiki_text_unit: {self.wiki_text_unit}')

    @overrides
    def _read(self, file_path) -> Iterable[Instance]:
        if self.use_quiz:
            logger.info('Reading Quiz data')
            for i, instance in enumerate(self._read_quiz(file_path)):
                if i < 10:
                    logger.info(f'Example:\n{instance}')

                yield instance

        if self.use_wiki:
            logger.info('Reading Wiki data')
            for i, instance in enumerate(self._read_wiki(self.wiki_file_path)):
                if i < 10:
                    logger.info(f'Example:\n{instance}')

                yield instance


@Tokenizer.register('transformer')
class TransformerTokenizer(Tokenizer):
    def __init__(self,
                 model_name: str,
                 add_special_tokens: bool = True,
                 max_length: int = None) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.add_special_tokens = add_special_tokens
        self.max_length = max_length

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        encoded_tokens = self.tokenizer.encode_plus(
            text=text,
            add_special_tokens=self.add_special_tokens,
            max_length=self.max_length,
            return_tensors=None)

        token_ids = encoded_tokens['input_ids']
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        tokens = [Token(text=token) for token in tokens]
        return tokens


@TokenIndexer.register('transformer')
class TransformerIndexer(TokenIndexer[int]):
    def __init__(self,
                 model_name: str,
                 namespace: str = 'tags',
                 token_min_padding_length: int = 0) -> None:
        super(TransformerIndexer, self).__init__(token_min_padding_length)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if isinstance(self.tokenizer, XLNetTokenizer):
            logger.info('Token ids will be padded on left')
            self.padding_on_right = False
        else:
            logger.info('Token ids will be padded on right')
            self.padding_on_right = True

        self.namespace = namespace
        self.padding_value = self.tokenizer.pad_token_id
        logger.info(f'Using token indexer padding value of {self.padding_value}')
        self.added_to_vocabulary = False

    @overrides
    def count_vocab_items(self,
                          token: Token,
                          counter: Dict[str, Dict[str, int]]):
        pass

    @overrides
    def tokens_to_indices(self,
                          tokens: List[Token],
                          vocabulary: Vocabulary,
                          index_name: str) -> Dict[str, List[int]]:
        tokens = [token.text for token in tokens]
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        return {index_name: token_ids, 'mask': [1] * len(token_ids)}

    @overrides
    def get_padding_lengths(self, token: int) -> Dict[str, int]:
        return {}

    @overrides
    def as_padded_tensor(self,
                         tokens: Dict[str, List[int]],
                         desired_num_tokens: Dict[str, int],
                         padding_lengths: Dict[str, int]) -> Dict[str, torch.Tensor]:
        tensors = {}
        for key, val in tokens.items():
            if key == 'mask':
                tensors[key] = torch.LongTensor(
                    pad_sequence_to_length(val, desired_num_tokens[key],
                                           default_value=lambda: 0,
                                           padding_on_right=self.padding_on_right))
            else:
                tensors[key] = torch.LongTensor(
                    pad_sequence_to_length(val, desired_num_tokens[key],
                                           default_value=lambda: self.padding_value,
                                           padding_on_right=self.padding_on_right))

        return tensors


@TokenEmbedder.register('transformer')
class TransformerEmbedder(TokenEmbedder):
    def __init__(self,
                 model_name: str,
                 layer_indices: List[int] = [-1],
                 dropout_masking: Optional[float] = None,
                 init_weights: bool = False) -> None:
        super(TransformerEmbedder, self).__init__()
        config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        self.transformer_model = AutoModel.from_pretrained(model_name, config=config)

        self.mask_token_id = AutoTokenizer.from_pretrained(model_name).mask_token_id
        self.layer_indices = layer_indices
        if init_weights:
            self.transformer_model.init_weights()

        self.dropout_masking = dropout_masking
        self.output_dim = self.transformer_model.config.hidden_size

    @overrides
    def get_output_dim(self):
        return self.output_dim

    def _dropout_mask_tokens(self,
                             token_ids: torch.LongTensor,
                             mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        probs = torch.rand_like(token_ids, dtype=torch.float)
        if mask is not None:
            probs *= mask

        token_ids = torch.where(probs > self.dropout_masking,
                                token_ids,
                                torch.full_like(token_ids, self.mask_token_id))

        return token_ids


    def forward(self,
                token_ids: torch.LongTensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.dropout_masking is not None:
            token_ids = self._dropout_mask_tokens(token_ids, mask=mask)

        hidden_states = self.transformer_model(token_ids, attention_mask=mask)[-1]
        return torch.cat([hidden_states[i] for i in self.layer_indices], dim=-1)


@Seq2VecEncoder.register('transformer')
class TransformerPooler(Seq2VecEncoder):
    def __init__(self,
                 pooling_type: str,
                 input_dim: int,
                 output_dim: int,
                 do_projection: bool = False,
                 activation: Optional[str] = None,
                 first_dropout: Optional[float] = None,
                 last_dropout: Optional[float] = None) -> None:
        super(TransformerPooler, self).__init__()
        self.pooling_type = pooling_type
        self.input_dim = input_dim
        self.output_dim = output_dim

        if do_projection:
            self.projection = nn.Linear(input_dim, output_dim)
            nn.init.xavier_uniform_(self.projection.weight)

        if activation is None:
            pass
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise KeyError(f'Invalid activation: {activation}')

        if first_dropout is not None:
            self.first_dropout = nn.Dropout(first_dropout)
        if last_dropout is not None:
            self.last_dropout = nn.Dropout(last_dropout)

    @overrides
    def get_input_dim(self) -> int:
        return self.input_dim

    @overrides
    def get_output_dim(self) -> int:
        return self.output_dim

    def forward(self,
                inputs: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert inputs.size(-1) == self.input_dim

        if self.pooling_type == 'first':
            outputs = inputs[:, 0]
        elif self.pooling_type == 'last':
            outputs = inputs[:, -1]
        elif self.pooling_type == 'mean':
            if mask is not None:
                outputs = masked_mean(inputs, mask[:,:,None], dim=1)
            else:
                outputs = inputs.mean(dim=1)
        elif self.pooling_type == 'max':
            if mask is not None:
                outputs = masked_max(inputs, mask[:,:,None], dim=1)
            else:
                outputs = inputs.max(dim=1)
        else:
            raise RuntimeError('Invalid pooling_type: {pooling_type}')

        if hasattr(self, 'first_dropout'):
            outputs = self.first_dropout(outputs)

        if hasattr(self, 'projection'):
            outputs = self.projection(outputs)

        if hasattr(self, 'activation'):
            outputs = self.activation(outputs)

        if hasattr(self, 'last_dropout'):
            outputs = self.last_dropout(outputs)

        assert outputs.size(-1) == self.output_dim
        return outputs


@Model.register('quizbowl_guesser')
class QuizbowlGuesser(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 text_encoder: Seq2VecEncoder,
                 entity_embedder: TokenEmbedder,
                 dropout_embeddings: float = 0.0,
                 dropout_encodings: float = 0.0,
                 dropout_logits: float = 0.0,
                 do_batch_norm: bool = False) -> None:
        super(QuizbowlGuesser, self).__init__(vocab)
        self.text_field_embedder = text_field_embedder
        self.dropout_embeddings = nn.Dropout(dropout_embeddings)
        self.text_encoder = text_encoder
        self.dropout_encodings = nn.Dropout(dropout_encodings)

        num_entities = vocab.get_vocab_size('entities')
        self.entity_embedder = entity_embedder
        if do_batch_norm:
            self.batch_norm = nn.BatchNorm1d(num_entities)

        self.dropout_logits = nn.Dropout(dropout_logits)

        self.accuracy = CategoricalAccuracy(top_k=1)
        self.mean_reciprocal_rank = MeanReciprocalRank()

    def forward(self,
                tokens: Dict[str, torch.Tensor],
                entity: torch.Tensor,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        output = dict()

        mask = get_text_field_mask(tokens)

        embedded_tokens = self.text_field_embedder(tokens, mask=mask)
        embedded_tokens = self.dropout_embeddings(embedded_tokens)

        encoded_text = self.text_encoder(embedded_tokens, mask=mask)
        encoded_text = self.dropout_encodings(encoded_text)

        logits = F.linear(encoded_text, self.entity_embedder.weight)
        if hasattr(self, 'batch_norm'):
            logits = self.batch_norm(logits)

        logits = self.dropout_logits(logits)
        loss = F.cross_entropy(logits, entity)

        output['loss'] = loss

        if not self.training:
            logits = torch.matmul(encoded_text, self.entity_embedder.weight.t())
            log_probs = F.log_softmax(logits, dim=1)
            output['log_probs'] = log_probs
            self.accuracy(log_probs, entity)
            self.mean_reciprocal_rank(log_probs, entity)

        output['metadata'] = metadata

        return output

    @overrides
    def decode(self,
               output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        top1_entities = []
        top10_entities = []
        ranks = []

        for probs, metadata in zip(output_dict['log_probs'], output_dict['metadata']):
            _, top_ids = torch.sort(probs, descending=True)
            top_entities = [self.vocab.get_index_to_token_vocabulary('entities')[i]
                            for i in top_ids.tolist()]
            top1_entities.append(top_entities[0])
            top10_entities.append(top_entities[:10])

            if 'entity' in metadata:
                correct_entity = metadata['entity'].replace(' ', '_')
                if correct_entity in top_entities:
                    ranks.append(top_entities.index(correct_entity) + 1)
                else:
                    ranks.append(None)

        output_dict['top1_entity'] = top1_entities
        output_dict['top10_entities'] = top10_entities
        output_dict['rank'] = ranks

        del output_dict['log_probs']  # delete it since it's really big

        return output_dict

    def get_metrics(self,
                    reset: bool = False) -> Dict[str, float]:
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
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        text = json_dict['text']
        entity = json_dict.get('entity')
        metadata = json_dict.get('metadata') or json_dict
        return self._dataset_reader.text_to_instance(text, entity, metadata)

    @overrides
    def predictions_to_labeled_instances(
            self,
            instance: Instance,
            outputs: Dict[str, numpy.ndarray]) -> List[Instance]:
        new_instance = deepcopy(instance)
        pred_entity = numpy.argmax(outputs['log_probs'])
        new_instance.add_field(
            'entity', LabelField(int(pred_entity), 'entities', skip_indexing=True))

        return new_instance

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:
        return json.dumps(outputs, ensure_ascii=False) + '\n'


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
            self.unwrap_to_tensors(predictions, gold_labels, mask)

        num_classes = predictions.size(-1)

        predictions = predictions.view((-1, num_classes))
        gold_labels = gold_labels.view(-1).long()

        predicted_ids = predictions.argsort(-1, descending=True)
        correct = predicted_ids.eq(gold_labels.unsqueeze(-1)).float()
        reciprocals = torch.arange(1, num_classes + 1).float().reciprocal()
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
