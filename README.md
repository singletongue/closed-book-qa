# Quizbowl as a Testbed of Entity Knowledge

## Make datasets

### Wiki datasets

```sh
$ ~/Repos/wikiextractor/WikiExtractor.py --output work/wikiextractor --bytes 100G --json --namespaces 0 --no_templates --processes 4 ~/data/qanta/2018/enwiki-20180420-pages-articles-multistream.xml.bz2
```

### Quizbowl datasets

```sh
# Train data
$ python make_quizbowl_dataset.py --dataset_file ~/data/qanta/2018/qanta.train.2018.04.18.json --output_file work/dataset/quizbowl/train_question.json --text_unit question
$ python make_quizbowl_dataset.py --dataset_file ~/data/qanta/2018/qanta.train.2018.04.18.json --output_file work/dataset/quizbowl/train_sentence.json --text_unit sentence

# List of entities which are present in train data
$ cat work/dataset/quizbowl/train_question.json|jq -r '.entity'|sort|uniq > work/dataset/quizbowl/train_entities.txt

# Dev data -- questions with an zero-shot entity are excluded
$ python make_quizbowl_dataset.py --dataset_file ~/data/qanta/2018/qanta.dev.2018.04.18.json --output_file work/dataset/quizbowl/dev_question.json --entities_file work/dataset/quizbowl/train_entities.txt --text_unit question
$ python make_quizbowl_dataset.py --dataset_file ~/data/qanta/2018/qanta.dev.2018.04.18.json --output_file work/dataset/quizbowl/dev_sentence.json --entities_file work/dataset/quizbowl/train_entities.txt --text_unit sentence

# Eval data -- for evaluating classification accuracy on development set
$ python make_quizbowl_dataset.py --dataset_file ~/data/qanta/2018/qanta.dev.2018.04.18.json --output_file work/dataset/quizbowl/eval_question.json --text_unit question
$ python make_quizbowl_dataset.py --dataset_file ~/data/qanta/2018/qanta.dev.2018.04.18.json --output_file work/dataset/quizbowl/eval_sentence.json --text_unit sentence

# Test data -- for Quizbowl evaluation metrics
$ python make_quizbowl_dataset.py --dataset_file ~/data/qanta/2018/qanta.test.2018.04.18.json --output_file work/dataset/quizbowl/test_question.json --text_unit question
$ python make_quizbowl_dataset.py --dataset_file ~/data/qanta/2018/qanta.test.2018.04.18.json --output_file work/dataset/quizbowl/test_sentence.json --text_unit sentence

# Create Wikipedia-augmented dataset for answer entities in train data
$ python make_wiki_dataset.py --dataset_file work/wikiextractor/AA/wiki_00 --title_list_file work/dataset/quizbowl/train_entities.txt --output_file work/dataset/quizbowl/wiki_sentence_blingfire.json --text_unit sentence --sent_splitter blingfire

# Make a concatenated dataset of Quiz and Wiki datasets
$ cat work/dataset/quizbowl/train_sentence.json work/dataset/quizbowl/wiki_sentence_blingfire.json > work/dataset/quizbowl/train_sentence_wiki_sentence_blingfire.json
```

### TriviaQA datasets

```sh
# Train data -- questions without a Wikipedia entity are excluded
$ python make_triviaqa_dataset.py --dataset_file ~/data/triviaqa/triviaqa-unfiltered/unfiltered-web-train.json --output_file work/dataset/triviaqa/train_question.json --skip_no_entity

# List of entities which are present in train data
$ cat work/dataset/triviaqa/train_question.json|jq -r '.entity'|sort|uniq > work/dataset/triviaqa/train_entities.txt

# Dev data -- questions without a Wikipedia entity and with an zero-shot entity are excluded
$ python make_triviaqa_dataset.py --dataset_file ~/data/triviaqa/qa/wikipedia-dev.json --output_file work/dataset/triviaqa/dev_question.json --entities_file work/dataset/triviaqa/train_entities.txt --skip_no_entity

# Eval data -- for evaluating classification accuracy on development set
$ python make_triviaqa_dataset.py --dataset_file ~/data/triviaqa/qa/wikipedia-dev.json --output_file work/dataset/triviaqa/eval_question.json

# Test data -- for TriviaQA evaluation metrics (no answers are originally annotated)
$ python make_triviaqa_dataset.py --dataset_file ~/data/triviaqa/qa/wikipedia-test-without-answers.json --output_file work/dataset/triviaqa/test_question.json

# Create Wikipedia-augmented dataset for answer entities in train data
$ python make_wiki_dataset.py --dataset_file work/wikiextractor/AA/wiki_00 --title_list_file work/dataset/triviaqa/train_entities.txt --output_file work/dataset/triviaqa/wiki_sentence_blingfire.json --text_unit sentence --sent_splitter blingfire

# Make a concatenated dataset of Quiz and Wiki datasets
$ cat work/dataset/triviaqa/train_question.json work/dataset/triviaqa/wiki_sentence_blingfire.json > work/dataset/triviaqa/train_question_wiki_sentence_blingfire.json
```

## make vocabulary files

### Quizbowl

```sh
$ mkdir -p work/quizbowl/vocab
$ allennlp train --dry-run --serialization-dir work/quizbowl/vocab/bert-base --include-package modules configs/quizbowl/make_vocab/bert-base.json
$ allennlp train --dry-run --serialization-dir work/quizbowl/vocab/roberta-base --include-package modules configs/quizbowl/make_vocab/roberta-base.json
$ allennlp train --dry-run --serialization-dir work/quizbowl/vocab/xlnet-base --include-package modules configs/quizbowl/make_vocab/xlnet-base.json
```

### TriviaQA

```sh
$ mkdir -p work/triviaqa/vocab
$ allennlp train --dry-run --serialization-dir work/triviaqa/vocab/bert-base --include-package modules configs/triviaqa/make_vocab/bert-base.json
$ allennlp train --dry-run --serialization-dir work/triviaqa/vocab/roberta-base --include-package modules configs/triviaqa/make_vocab/roberta-base.json
$ allennlp train --dry-run --serialization-dir work/triviaqa/vocab/xlnet-base --include-package modules configs/triviaqa/make_vocab/xlnet-base.json
```

## Training

### Quizbowl

```sh
# Quiz
$ mkdir work/quizbowl/quiz
$ allennlp train --serialization-dir work/quizbowl/quiz/bert-base --include-package modules configs/quizbowl/quiz/bert-base.json
$ allennlp train --serialization-dir work/quizbowl/quiz/roberta-base --include-package modules configs/quizbowl/quiz/roberta-base.json
$ allennlp train --serialization-dir work/quizbowl/quiz/xlnet-base --include-package modules configs/quizbowl/quiz/xlnet-base.json
$ python archive_model.py --serialization_dir work/quizbowl/quiz/bert-base --weights_name model_state_epoch_9.th --archive_name model_epoch_9.tar.gz
$ python archive_model.py --serialization_dir work/quizbowl/quiz/roberta-base --weights_name model_state_epoch_9.th --archive_name model_epoch_9.tar.gz
$ python archive_model.py --serialization_dir work/quizbowl/quiz/xlnet-base --weights_name model_state_epoch_9.th --archive_name model_epoch_9.tar.gz

# Wiki
$ mkdir work/quizbowl/wiki
$ allennlp train --serialization-dir work/quizbowl/wiki/bert-base --include-package modules configs/quizbowl/wiki/bert-base.json
$ allennlp train --serialization-dir work/quizbowl/wiki/roberta-base --include-package modules configs/quizbowl/wiki/roberta-base.json
$ allennlp train --serialization-dir work/quizbowl/wiki/xlnet-base --include-package modules configs/quizbowl/wiki/xlnet-base.json
$ python archive_model.py --serialization_dir work/quizbowl/wiki/bert-base --weights_name model_state_epoch_9.th --archive_name model_epoch_9.tar.gz
$ python archive_model.py --serialization_dir work/quizbowl/wiki/roberta-base --weights_name model_state_epoch_9.th --archive_name model_epoch_9.tar.gz
$ python archive_model.py --serialization_dir work/quizbowl/wiki/xlnet-base --weights_name model_state_epoch_9.th --archive_name model_epoch_9.tar.gz

# Quiz + Wiki
$ mkdir work/quizbowl/quiz_and_wiki
$ allennlp train --serialization-dir work/quizbowl/quiz_and_wiki/bert-base --include-package modules configs/quizbowl/quiz_and_wiki/bert-base.json
$ allennlp train --serialization-dir work/quizbowl/quiz_and_wiki/roberta-base --include-package modules configs/quizbowl/quiz_and_wiki/roberta-base.json
$ allennlp train --serialization-dir work/quizbowl/quiz_and_wiki/xlnet-base --include-package modules configs/quizbowl/quiz_and_wiki/xlnet-base.json
$ python archive_model.py --serialization_dir work/quizbowl/quiz_and_wiki/bert-base --weights_name model_state_epoch_9.th --archive_name model_epoch_9.tar.gz
$ python archive_model.py --serialization_dir work/quizbowl/quiz_and_wiki/roberta-base --weights_name model_state_epoch_9.th --archive_name model_epoch_9.tar.gz
$ python archive_model.py --serialization_dir work/quizbowl/quiz_and_wiki/xlnet-base --weights_name model_state_epoch_9.th --archive_name model_epoch_9.tar.gz

# Quiz -> Wiki
$ mkdir work/quizbowl/quiz_to_wiki
$ allennlp train --serialization-dir work/quizbowl/quiz_to_wiki/bert-base --include-package modules configs/quizbowl/quiz_to_wiki/bert-base.json
$ allennlp train --serialization-dir work/quizbowl/quiz_to_wiki/roberta-base --include-package modules configs/quizbowl/quiz_to_wiki/roberta-base.json
$ allennlp train --serialization-dir work/quizbowl/quiz_to_wiki/xlnet-base --include-package modules configs/quizbowl/quiz_to_wiki/xlnet-base.json
$ python archive_model.py --serialization_dir work/quizbowl/quiz_to_wiki/bert-base --weights_name model_state_epoch_9.th --archive_name model_epoch_9.tar.gz
$ python archive_model.py --serialization_dir work/quizbowl/quiz_to_wiki/roberta-base --weights_name model_state_epoch_9.th --archive_name model_epoch_9.tar.gz
$ python archive_model.py --serialization_dir work/quizbowl/quiz_to_wiki/xlnet-base --weights_name model_state_epoch_9.th --archive_name model_epoch_9.tar.gz

# Wiki -> Quiz
$ mkdir work/quizbowl/wiki_to_quiz
$ allennlp train --serialization-dir work/quizbowl/wiki_to_quiz/bert-base --include-package modules configs/quizbowl/wiki_to_quiz/bert-base.json
$ allennlp train --serialization-dir work/quizbowl/wiki_to_quiz/roberta-base --include-package modules configs/quizbowl/wiki_to_quiz/roberta-base.json
$ allennlp train --serialization-dir work/quizbowl/wiki_to_quiz/xlnet-base --include-package modules configs/quizbowl/wiki_to_quiz/xlnet-base.json
$ python archive_model.py --serialization_dir work/quizbowl/wiki_to_quiz/bert-base --weights_name model_state_epoch_9.th --archive_name model_epoch_9.tar.gz
$ python archive_model.py --serialization_dir work/quizbowl/wiki_to_quiz/roberta-base --weights_name model_state_epoch_9.th --archive_name model_epoch_9.tar.gz
$ python archive_model.py --serialization_dir work/quizbowl/wiki_to_quiz/xlnet-base --weights_name model_state_epoch_9.th --archive_name model_epoch_9.tar.gz
```

### TriviaQA

```sh
# Quiz
$ mkdir work/triviaqa/quiz
$ allennlp train --serialization-dir work/triviaqa/quiz/bert-base --include-package modules configs/triviaqa/quiz/bert-base.json
$ allennlp train --serialization-dir work/triviaqa/quiz/bert-base_30ep --include-package modules configs/triviaqa/quiz/bert-base_30ep.json
$ allennlp train --serialization-dir work/triviaqa/quiz/roberta-base --include-package modules configs/triviaqa/quiz/roberta-base.json
$ allennlp train --serialization-dir work/triviaqa/quiz/xlnet-base --include-package modules configs/triviaqa/quiz/xlnet-base.json
$ python archive_model.py --serialization_dir work/triviaqa/quiz/bert-base --weights_name model_state_epoch_9.th --archive_name model_epoch_9.tar.gz
$ python archive_model.py --serialization_dir work/triviaqa/quiz/bert-base_30ep --weights_name model_state_epoch_29.th --archive_name model_epoch_29.tar.gz
$ python archive_model.py --serialization_dir work/triviaqa/quiz/roberta-base --weights_name model_state_epoch_9.th --archive_name model_epoch_9.tar.gz
$ python archive_model.py --serialization_dir work/triviaqa/quiz/xlnet-base --weights_name model_state_epoch_9.th --archive_name model_epoch_9.tar.gz

# Wiki
$ mkdir work/triviaqa/wiki
$ allennlp train --serialization-dir work/triviaqa/wiki/bert-base --include-package modules configs/triviaqa/wiki/bert-base.json
$ allennlp train --serialization-dir work/triviaqa/wiki/roberta-base --include-package modules configs/triviaqa/wiki/roberta-base.json
$ allennlp train --serialization-dir work/triviaqa/wiki/xlnet-base --include-package modules configs/triviaqa/wiki/xlnet-base.json
$ python archive_model.py --serialization_dir work/triviaqa/wiki/bert-base --weights_name model_state_epoch_9.th --archive_name model_epoch_9.tar.gz
$ python archive_model.py --serialization_dir work/triviaqa/wiki/roberta-base --weights_name model_state_epoch_9.th --archive_name model_epoch_9.tar.gz
$ python archive_model.py --serialization_dir work/triviaqa/wiki/xlnet-base --weights_name model_state_epoch_9.th --archive_name model_epoch_9.tar.gz

# Quiz + Wiki
$ mkdir work/triviaqa/quiz_and_wiki
$ allennlp train --serialization-dir work/triviaqa/quiz_and_wiki/bert-base --include-package modules configs/triviaqa/quiz_and_wiki/bert-base.json
$ allennlp train --serialization-dir work/triviaqa/quiz_and_wiki/roberta-base --include-package modules configs/triviaqa/quiz_and_wiki/roberta-base.json
$ allennlp train --serialization-dir work/triviaqa/quiz_and_wiki/xlnet-base --include-package modules configs/triviaqa/quiz_and_wiki/xlnet-base.json
$ python archive_model.py --serialization_dir work/triviaqa/quiz_and_wiki/bert-base --weights_name model_state_epoch_9.th --archive_name model_epoch_9.tar.gz
$ python archive_model.py --serialization_dir work/triviaqa/quiz_and_wiki/roberta-base --weights_name model_state_epoch_9.th --archive_name model_epoch_9.tar.gz
$ python archive_model.py --serialization_dir work/triviaqa/quiz_and_wiki/xlnet-base --weights_name model_state_epoch_9.th --archive_name model_epoch_9.tar.gz

# Quiz -> Wiki
$ mkdir work/triviaqa/quiz_to_wiki
$ allennlp train --serialization-dir work/triviaqa/quiz_to_wiki/bert-base --include-package modules configs/triviaqa/quiz_to_wiki/bert-base.json
$ allennlp train --serialization-dir work/triviaqa/quiz_to_wiki/roberta-base --include-package modules configs/triviaqa/quiz_to_wiki/roberta-base.json
$ allennlp train --serialization-dir work/triviaqa/quiz_to_wiki/xlnet-base --include-package modules configs/triviaqa/quiz_to_wiki/xlnet-base.json
$ python archive_model.py --serialization_dir work/triviaqa/quiz_to_wiki/bert-base --weights_name model_state_epoch_9.th --archive_name model_epoch_9.tar.gz
$ python archive_model.py --serialization_dir work/triviaqa/quiz_to_wiki/roberta-base --weights_name model_state_epoch_9.th --archive_name model_epoch_9.tar.gz
$ python archive_model.py --serialization_dir work/triviaqa/quiz_to_wiki/xlnet-base --weights_name model_state_epoch_9.th --archive_name model_epoch_9.tar.gz

# Wiki -> Quiz
$ mkdir work/triviaqa/wiki_to_quiz
$ allennlp train --serialization-dir work/triviaqa/wiki_to_quiz/bert-base --include-package modules configs/triviaqa/wiki_to_quiz/bert-base.json
$ allennlp train --serialization-dir work/triviaqa/wiki_to_quiz/roberta-base --include-package modules configs/triviaqa/wiki_to_quiz/roberta-base.json
$ allennlp train --serialization-dir work/triviaqa/wiki_to_quiz/xlnet-base --include-package modules configs/triviaqa/wiki_to_quiz/xlnet-base.json
$ python archive_model.py --serialization_dir work/triviaqa/wiki_to_quiz/bert-base --weights_name model_state_epoch_9.th --archive_name model_epoch_9.tar.gz
$ python archive_model.py --serialization_dir work/triviaqa/wiki_to_quiz/roberta-base --weights_name model_state_epoch_9.th --archive_name model_epoch_9.tar.gz
$ python archive_model.py --serialization_dir work/triviaqa/wiki_to_quiz/xlnet-base --weights_name model_state_epoch_9.th --archive_name model_epoch_9.tar.gz
```

## Prediction

### Quizbowl

```sh
# Question level
$ for dir in work/quizbowl/quiz*/* work/quizbowl/wiki*/*; do allennlp predict $dir/model_epoch_9.tar.gz work/dataset/quizbowl/train_question.json --output-file $dir/prediction_train_question.json --batch-size 64 --silent --cuda-device 0 --use-dataset-reader --predictor quiz --include-package modules; done
$ for dir in work/quizbowl/quiz*/* work/quizbowl/wiki*/*; do allennlp predict $dir/model_epoch_9.tar.gz work/dataset/quizbowl/dev_question.json --output-file $dir/prediction_dev_question.json --batch-size 64 --silent --cuda-device 0 --use-dataset-reader --predictor quiz --include-package modules; done
$ for dir in work/quizbowl/quiz*/* work/quizbowl/wiki*/*; do allennlp predict $dir/model_epoch_9.tar.gz work/dataset/quizbowl/eval_question.json --output-file $dir/prediction_eval_question.json --batch-size 1 --silent --cuda-device 0 --use-dataset-reader --predictor quiz --include-package modules; done

# Sentence level
$ for dir in work/quizbowl/quiz*/* work/quizbowl/wiki*/*; do allennlp predict $dir/model_epoch_9.tar.gz work/dataset/quizbowl/train_sentence.json --output-file $dir/prediction_train_sentence.json --batch-size 256 --silent --cuda-device 0 --use-dataset-reader --predictor quiz --include-package modules; done
$ for dir in work/quizbowl/quiz*/* work/quizbowl/wiki*/*; do allennlp predict $dir/model_epoch_9.tar.gz work/dataset/quizbowl/dev_sentence.json --output-file $dir/prediction_dev_sentence.json --batch-size 256 --silent --cuda-device 0 --use-dataset-reader --predictor quiz --include-package modules; done
$ for dir in work/quizbowl/quiz*/* work/quizbowl/wiki*/*; do allennlp predict $dir/model_epoch_9.tar.gz work/dataset/quizbowl/eval_sentence.json --output-file $dir/prediction_eval_sentence.json --batch-size 1 --silent --cuda-device 0 --use-dataset-reader --predictor quiz --include-package modules; done
$ for dir in work/quizbowl/quiz*/* work/quizbowl/wiki*/*; do allennlp predict $dir/model_epoch_9.tar.gz work/dataset/quizbowl/test_sentence.json --output-file $dir/prediction_test_sentence.json --batch-size 1 --silent --cuda-device 0 --use-dataset-reader --predictor quiz --include-package modules; done
```

### TriviaQA

```sh
$ for dir in work/triviaqa/quiz*/* work/triviaqa/wiki*/*; do allennlp predict $dir/model_epoch_9.tar.gz work/dataset/triviaqa/train_question.json --output-file $dir/prediction_train_question.json --batch-size 64 --silent --cuda-device 0 --use-dataset-reader --predictor quiz --include-package modules; done
$ for dir in work/triviaqa/quiz*/* work/triviaqa/wiki*/*; do allennlp predict $dir/model_epoch_9.tar.gz work/dataset/triviaqa/dev_question.json --output-file $dir/prediction_dev_question.json --batch-size 64 --silent --cuda-device 0 --use-dataset-reader --predictor quiz --include-package modules; done
$ for dir in work/triviaqa/quiz*/* work/triviaqa/wiki*/*; do allennlp predict $dir/model_epoch_9.tar.gz work/dataset/triviaqa/eval_question.json --output-file $dir/prediction_eval_question.json --batch-size 64 --silent --cuda-device 0 --use-dataset-reader --predictor quiz --include-package modules; done

$ allennlp predict work/triviaqa/quiz/bert-base_30ep/model_epoch_29.tar.gz work/dataset/triviaqa/train_question.json --output-file work/triviaqa/quiz/bert-base_30ep/prediction_train_question.json --batch-size 64 --silent --cuda-device 0 --use-dataset-reader --predictor quiz --include-package modules
$ allennlp predict work/triviaqa/quiz/bert-base_30ep/model_epoch_29.tar.gz work/dataset/triviaqa/dev_question.json --output-file work/triviaqa/quiz/bert-base_30ep/prediction_dev_question.json --batch-size 64 --silent --cuda-device 0 --use-dataset-reader --predictor quiz --include-package modules
$ allennlp predict work/triviaqa/quiz/bert-base_30ep/model_epoch_29.tar.gz work/dataset/triviaqa/eval_question.json --output-file work/triviaqa/quiz/bert-base_30ep/prediction_eval_question.json --batch-size 1 --silent --cuda-device 0 --use-dataset-reader --predictor quiz --include-package modules
```

## Print classification results

### Quizbowl

```sh
$ for file in work/quizbowl/*/*/prediction_train_question.json; do echo $file; python print_result.py --input_file $file; echo; done > work/quizbowl/results_train_question
$ for file in work/quizbowl/*/*/prediction_eval_question.json; do echo $file; python print_result.py --input_file $file; echo; done > work/quizbowl/results_eval_question
```

### TriviaQA

```sh
$ for file in work/triviaqa/*/*/prediction_train_question.json; do echo $file; python print_result.py --input_file $file; echo; done > work/triviaqa/results_train_question
$ for file in work/triviaqa/*/*/prediction_eval_question.json; do echo $file; python print_result.py --input_file $file; echo; done > work/triviaqa/results_eval_question
```

## Test set evaluation

### Quizbowl

```sh
$ allennlp predict work/quizbowl/quiz/bert-base/model_epoch_9.tar.gz work/dataset/quizbowl/test_question.json --output-file work/quizbowl/quiz/bert-base/prediction_test_question.json --batch-size 1 --silent --cuda-device 0 --use-dataset-reader --predictor quiz --include-package modules
$ allennlp predict work/quizbowl/wiki_to_quiz/bert-base/model_epoch_9.tar.gz work/dataset/quizbowl/test_question.json --output-file work/quizbowl/wiki_to_quiz/bert-base/prediction_test_question.json --batch-size 1 --silent --cuda-device 0 --use-dataset-reader --predictor quiz --include-package modules
```

### TriviaQA

```sh
$ allennlp predict work/triviaqa/wiki_to_quiz/bert-base/model_epoch_9.tar.gz work/dataset/triviaqa/test_question.json --output-file work/triviaqa/wiki_to_quiz/bert-base/prediction_test_question.json --batch-size 1 --silent --cuda-device 0 --use-dataset-reader --predictor quiz --include-package modules

# convert the test results to TriviaQA format
$ python convert_triviaqa_prediction.py --input_file work/triviaqa/wiki_to_quiz/bert-base/prediction_test_question.json --output_file work/triviaqa/wiki_to_quiz/bert-base/prediction_test_question_converted.json --postprocess_answers

# prepare submission files for TriviaQA leader's board
$ cd work/triviaqa/wiki_to_quiz/bert-base
$ cp prediction_test_question_converted.json predictions.json && zip -j submission.zip predictions.json && rm predictions.json
```
