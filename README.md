# Quizbowl as a Testbed of Entity Knowledge

- All the experiments were conducted on RAIDEN.

## Make datasets

```sh
$ python make_quiz_dataset.py --dataset_file ~/data/qanta/2018/qanta.train.2018.04.18.json --output_file work/dataset/train_question.json --text_unit question
$ python make_quiz_dataset.py --dataset_file ~/data/qanta/2018/qanta.train.2018.04.18.json --output_file work/dataset/train_sentence.json --text_unit sentence
$ python make_quiz_dataset.py --dataset_file ~/data/qanta/2018/qanta.dev.2018.04.18.json --output_file work/dataset/dev_question.json --text_unit question
$ python make_quiz_dataset.py --dataset_file ~/data/qanta/2018/qanta.dev.2018.04.18.json --output_file work/dataset/dev_sentence.json --text_unit sentence
$ python make_quiz_dataset.py --dataset_file ~/data/qanta/2018/qanta.test.2018.04.18.json --output_file work/dataset/test_question.json --text_unit question
$ python make_quiz_dataset.py --dataset_file ~/data/qanta/2018/qanta.test.2018.04.18.json --output_file work/dataset/test_sentence.json --text_unit sentence

$ python make_wiki_dataset.py --dataset_file ~/data/qanta/2018/wiki_lookup.json --output_file work/dataset/wiki_sentence_blingfire.json --text_unit sentence --sent_splitter blingfire
$ python make_wiki_dataset.py --dataset_file ~/data/qanta/2018/wiki_lookup.json --output_file work/dataset/wiki-fp_sentence_blingfire.json --text_unit sentence --sent_splitter blingfire --max_paragraphs 1

$ cat work/dataset/train_sentence.json work/dataset/wiki_sentence_blingfire.json > work/dataset/train_sentence_wiki_sentence_blingfire.json
$ cat work/dataset/train_sentence.json work/dataset/wiki-fp_sentence_blingfire.json > work/dataset/train_sentence_wiki-fp_sentence_blingfire.json

$ python make_triviaqa_dataset.py --dataset_file ~/data/triviaqa/triviaqa-unfiltered/unfiltered-web-train.json --output_file work/dataset/triviaqa_train.json
$ python make_triviaqa_dataset.py --dataset_file ~/data/triviaqa/triviaqa-unfiltered/unfiltered-web-dev.json --output_file work/dataset/triviaqa_dev.json
$ python make_triviaqa_dataset.py --dataset_file ~/data/triviaqa/triviaqa-unfiltered/unfiltered-web-test-without-answers.json --output_file work/dataset/triviaqa_test.json
```

## make vocabulary files

```sh
$ mkdir -p work/vocab
$ allennlp train --dry-run --serialization-dir work/vocab/bert-base --include-package modules configs/make_vocab/bert-base.json

$ mkdir -p work/triviaqa/vocab
$ allennlp train --dry-run --serialization-dir work/triviaqa/vocab/bert-base --include-package modules configs/triviaqa/make_vocab/bert-base.json
```

## Training

```sh
$ mkdir work/param_search
$ qsub -v CONFIG_FILE=configs/param_search/bert-base_b64_e6.json,SERIALIZATION_DIR=work/param_search/bert-base_b64_e6 -N b64_e6 train_raiden.sh
$ qsub -v CONFIG_FILE=configs/param_search/bert-base_b64_e8.json,SERIALIZATION_DIR=work/param_search/bert-base_b64_e8 -N b64_e8 train_raiden.sh
$ qsub -v CONFIG_FILE=configs/param_search/bert-base_b64_e10.json,SERIALIZATION_DIR=work/param_search/bert-base_b64_e10 -N b64_e10 train_raiden.sh

$ mkdir work/quiz
$ qsub -v CONFIG_FILE=configs/quiz/bert-base.json,SERIALIZATION_DIR=work/quiz/bert-base -N quiz train_raiden.sh

$ mkdir work/wiki
$ qsub -v CONFIG_FILE=configs/wiki/bert-base.json,SERIALIZATION_DIR=work/wiki/bert-base -N wiki train_raiden.sh

$ mkdir work/wiki-fp
$ qsub -v CONFIG_FILE=configs/wiki-fp/bert-base.json,SERIALIZATION_DIR=work/wiki-fp/bert-base -N wiki-fp train_raiden.sh

$ mkdir work/quiz_and_wiki
$ qsub -v CONFIG_FILE=configs/quiz_and_wiki/bert-base.json,SERIALIZATION_DIR=work/quiz_and_wiki/bert-base -N quiz_and_wiki -jc gpu-container_g1.72h train_raiden.sh

$ mkdir work/quiz_and_wiki-fp
$ qsub -v CONFIG_FILE=configs/quiz_and_wiki-fp/bert-base.json,SERIALIZATION_DIR=work/quiz_and_wiki-fp/bert-base -N quiz_and_wiki-fp train_raiden.sh

$ mkdir work/quiz_to_wiki
$ qsub -v CONFIG_FILE=configs/quiz_to_wiki/bert-base.json,SERIALIZATION_DIR=work/quiz_to_wiki/bert-base -N quiz_to_wiki train_raiden.sh

$ mkdir work/wiki_to_quiz
$ qsub -v CONFIG_FILE=configs/wiki_to_quiz/bert-base.json,SERIALIZATION_DIR=work/wiki_to_quiz/bert-base -N wiki_to_quiz train_raiden.sh

$ mkdir work/quiz_to_wiki-fp
$ qsub -v CONFIG_FILE=configs/quiz_to_wiki-fp/bert-base.json,SERIALIZATION_DIR=work/quiz_to_wiki-fp/bert-base -N quiz_to_wiki-fp train_raiden.sh

$ mkdir work/wiki-fp_to_quiz
$ qsub -v CONFIG_FILE=configs/wiki-fp_to_quiz/bert-base.json,SERIALIZATION_DIR=work/wiki-fp_to_quiz/bert-base -N wiki-fp_to_quiz train_raiden.sh
```

```sh
$ mkdir work/triviaqa/quiz
$ qsub -v CONFIG_FILE=configs/triviaqa/quiz/bert-base.json,SERIALIZATION_DIR=work/triviaqa/quiz/bert-base -N quiz train_raiden.sh
```

---

## Evaluation

```sh
$ allennlp evaluate --output-file work/quiz/bert-base/metrics_train.json --cuda-device 0 --include-package modules work/quiz/bert-base/model.tar.gz work/dataset/train_question.json
$ allennlp evaluate --output-file work/quiz/bert-base/metrics_dev.json --cuda-device 0 --include-package modules work/quiz/bert-base/model.tar.gz work/dataset/dev_question.json
$ allennlp evaluate --output-file work/quiz/bert-base/metrics_test.json --cuda-device 0 --include-package modules work/quiz/bert-base/model.tar.gz work/dataset/test_question.json

$ allennlp evaluate --output-file work/wiki/bert-base/metrics_dev.json --cuda-device 0 --include-package modules work/wiki/bert-base/model.tar.gz work/dataset/dev_question.json
$ allennlp evaluate --output-file work/wiki/bert-base/metrics_test.json --cuda-device 0 --include-package modules work/wiki/bert-base/model.tar.gz work/dataset/test_question.json

$ allennlp evaluate --output-file work/wiki-fp/bert-base/metrics_dev.json --cuda-device 0 --include-package modules work/wiki-fp/bert-base/model.tar.gz work/dataset/dev_question.json
$ allennlp evaluate --output-file work/wiki-fp/bert-base/metrics_test.json --cuda-device 0 --include-package modules work/wiki-fp/bert-base/model.tar.gz work/dataset/test_question.json

$ allennlp evaluate --output-file work/quiz_and_wiki/bert-base/metrics_dev.json --cuda-device 0 --include-package modules work/quiz_and_wiki/bert-base/model.tar.gz work/dataset/dev_question.json
$ allennlp evaluate --output-file work/quiz_and_wiki/bert-base/metrics_test.json --cuda-device 0 --include-package modules work/quiz_and_wiki/bert-base/model.tar.gz work/dataset/test_question.json

$ allennlp evaluate --output-file work/quiz_and_wiki-fp/bert-base/metrics_dev.json --cuda-device 0 --include-package modules work/quiz_and_wiki-fp/bert-base/model.tar.gz work/dataset/dev_question.json
$ allennlp evaluate --output-file work/quiz_and_wiki-fp/bert-base/metrics_test.json --cuda-device 0 --include-package modules work/quiz_and_wiki-fp/bert-base/model.tar.gz work/dataset/test_question.json

$ allennlp evaluate --output-file work/quiz_to_wiki/bert-base/metrics_dev.json --cuda-device 0 --include-package modules work/quiz_to_wiki/bert-base/model.tar.gz work/dataset/dev_question.json
$ allennlp evaluate --output-file work/quiz_to_wiki/bert-base/metrics_test.json --cuda-device 0 --include-package modules work/quiz_to_wiki/bert-base/model.tar.gz work/dataset/test_question.json

$ allennlp evaluate --output-file work/wiki_to_quiz/bert-base/metrics_dev.json --cuda-device 0 --include-package modules work/wiki_to_quiz/bert-base/model.tar.gz work/dataset/dev_question.json
$ allennlp evaluate --output-file work/wiki_to_quiz/bert-base/metrics_test.json --cuda-device 0 --include-package modules work/wiki_to_quiz/bert-base/model.tar.gz work/dataset/test_question.json

$ allennlp evaluate --output-file work/quiz_to_wiki-fp/bert-base/metrics_dev.json --cuda-device 0 --include-package modules work/quiz_to_wiki-fp/bert-base/model.tar.gz work/dataset/dev_question.json
$ allennlp evaluate --output-file work/quiz_to_wiki-fp/bert-base/metrics_test.json --cuda-device 0 --include-package modules work/quiz_to_wiki-fp/bert-base/model.tar.gz work/dataset/test_question.json

$ allennlp evaluate --output-file work/wiki-fp_to_quiz/bert-base/metrics_dev.json --cuda-device 0 --include-package modules work/wiki-fp_to_quiz/bert-base/model.tar.gz work/dataset/dev_question.json
$ allennlp evaluate --output-file work/wiki-fp_to_quiz/bert-base/metrics_test.json --cuda-device 0 --include-package modules work/wiki-fp_to_quiz/bert-base/model.tar.gz work/dataset/test_question.json
```

```sh
# dev
$ allennlp print-results work -k acc mrr -m metrics_dev.json
...
model_run, acc, mrr
work/quiz/bert-base/metrics_dev.json, 0.6935920577617328, 0.7358414085333098
work/quiz_and_wiki-fp/bert-base/metrics_dev.json, 0.6407942238267148, 0.6968914494187393
work/quiz_and_wiki/bert-base/metrics_dev.json, 0.6886281588447654, 0.7342919106948247
work/quiz_to_wiki-fp/bert-base/metrics_dev.json, 0.6380866425992779, 0.6938308516133993
work/quiz_to_wiki/bert-base/metrics_dev.json, 0.5009025270758123, 0.5773992912863997
work/wiki-fp/bert-base/metrics_dev.json, 0.0825812274368231, 0.1445978161498958
work/wiki-fp_to_quiz/bert-base/metrics_dev.json, 0.7003610108303249, 0.7443732929143665
work/wiki/bert-base/metrics_dev.json, 0.2612815884476534, 0.3416881928159872
work/wiki_to_quiz/bert-base/metrics_dev.json, 0.7247292418772563, 0.7634097776688393

# test
$ allennlp print-results work -k acc mrr -m metrics_test.json
...
model_run, acc, mrr
work/quiz/bert-base/metrics_test.json, 0.6325536062378168, 0.6835962791889034
work/quiz_and_wiki-fp/bert-base/metrics_test.json, 0.5986842105263158, 0.6561420620301082
work/quiz_and_wiki/bert-base/metrics_test.json, 0.6308479532163743, 0.6838970407407884
work/quiz_to_wiki-fp/bert-base/metrics_test.json, 0.574317738791423, 0.6370133146440309
work/quiz_to_wiki/bert-base/metrics_test.json, 0.44615009746588696, 0.5238663297415245
work/wiki-fp/bert-base/metrics_test.json, 0.07212475633528265, 0.12898080130098624
work/wiki-fp_to_quiz/bert-base/metrics_test.json, 0.6500974658869396, 0.6997446648442489
work/wiki/bert-base/metrics_test.json, 0.23757309941520469, 0.3126578214224319
work/wiki_to_quiz/bert-base/metrics_test.json, 0.6649610136452242, 0.7101871188853451
```

## Prediction

```sh
# question-level evaluation
$ allennlp predict work/quiz/bert-base/model.tar.gz work/dataset/dev_question.json --output-file work/quiz/bert-base/prediction_dev_question.json --silent --cuda-device 0 --use-dataset-reader --predictor quizbowl --include-package modules
$ allennlp predict work/quiz/bert-base/model.tar.gz work/dataset/test_question.json --output-file work/quiz/bert-base/prediction_test_question.json --silent --cuda-device 0 --use-dataset-reader --predictor quizbowl --include-package modules
$ allennlp predict work/wiki_to_quiz/bert-base/model.tar.gz work/dataset/dev_question.json --output-file work/wiki_to_quiz/bert-base/prediction_dev_question.json --silent --cuda-device 0 --use-dataset-reader --predictor quizbowl --include-package modules
$ allennlp predict work/wiki_to_quiz/bert-base/model.tar.gz work/dataset/test_question.json --output-file work/wiki_to_quiz/bert-base/prediction_test_question.json --silent --cuda-device 0 --use-dataset-reader --predictor quizbowl --include-package modules

# sentence-level evaluation
allennlp predict work/quiz/bert-base/model.tar.gz work/dataset/dev_sentence.json --output-file work/quiz/bert-base/prediction_dev_sentence.json --silent --cuda-device 0 --use-dataset-reader --predictor quizbowl --include-package modules
allennlp predict work/quiz/bert-base/model.tar.gz work/dataset/test_sentence.json --output-file work/quiz/bert-base/prediction_test_sentence.json --silent --cuda-device 0 --use-dataset-reader --predictor quizbowl --include-package modules
allennlp predict work/wiki_to_quiz/bert-base/model.tar.gz work/dataset/dev_sentence.json --output-file work/wiki_to_quiz/bert-base/prediction_dev_sentence.json --silent --cuda-device 0 --use-dataset-reader --predictor quizbowl --include-package modules
allennlp predict work/wiki_to_quiz/bert-base/model.tar.gz work/dataset/test_sentence.json --output-file work/wiki_to_quiz/bert-base/prediction_test_sentence.json --silent --cuda-device 0 --use-dataset-reader --predictor quizbowl --include-package modules
```
