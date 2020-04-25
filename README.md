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
```

## Make vocabulary

```sh
$ allennlp train --dry-run --serialization-dir work/vocabulary --include-package modules configs/quiz/bert-base.json
```

## Training

```sh
$ mkdir work/quiz
$ qsub -v CONFIG_FILE=configs/quiz/bert-base.json,SERIALIZATION_DIR=work/quiz/bert-base -N bert-base train_raiden.sh

$ mkdir work/wiki
$ qsub -v CONFIG_FILE=configs/wiki/bert-base.json,SERIALIZATION_DIR=work/wiki/bert-base -N bert-base train_raiden.sh

$ mkdir work/wiki-fp
$ qsub -v CONFIG_FILE=configs/wiki-fp/bert-base.json,SERIALIZATION_DIR=work/wiki-fp/bert-base -N bert-base train_raiden.sh

$ mkdir work/quiz_to_wiki
$ qsub -v CONFIG_FILE=configs/quiz_to_wiki/bert-base.json,SERIALIZATION_DIR=work/quiz_to_wiki/bert-base -N bert-base train_raiden.sh

$ mkdir work/wiki_to_quiz
$ qsub -v CONFIG_FILE=configs/wiki_to_quiz/bert-base.json,SERIALIZATION_DIR=work/wiki_to_quiz/bert-base -N bert-base train_raiden.sh

$ mkdir work/quiz_to_wiki-fp
$ qsub -v CONFIG_FILE=configs/quiz_to_wiki-fp/bert-base.json,SERIALIZATION_DIR=work/quiz_to_wiki-fp/bert-base -N bert-base train_raiden.sh

$ mkdir work/wiki-fp_to_quiz
$ qsub -v CONFIG_FILE=configs/wiki-fp_to_quiz/bert-base.json,SERIALIZATION_DIR=work/wiki-fp_to_quiz/bert-base -N bert-base train_raiden.sh

$ mkdir work/quiz_and_wiki
$ qsub -v CONFIG_FILE=configs/quiz_and_wiki/bert-base.json,SERIALIZATION_DIR=work/quiz_and_wiki/bert-base -N bert-base train_raiden.sh

$ mkdir work/quiz_and_wiki-fp
$ qsub -v CONFIG_FILE=configs/quiz_and_wiki-fp/bert-base.json,SERIALIZATION_DIR=work/quiz_and_wiki-fp/bert-base -N bert-base train_raiden.sh
```

---

## Evaluation

```sh
$ allennlp evaluate --output-file work/quiz/bert-base/metrics_train.json --cuda-device 0 --include-package modules work/quiz/bert-base/model.tar.gz work/dataset/train_question.json
$ allennlp evaluate --output-file work/quiz/bert-base/metrics_dev.json --cuda-device 0 --include-package modules work/quiz/bert-base/model.tar.gz work/dataset/dev_question.json
$ allennlp evaluate --output-file work/quiz/bert-base/metrics_test.json --cuda-device 0 --include-package modules work/quiz/bert-base/model.tar.gz work/dataset/test_question.json

$ allennlp evaluate --output-file work/wiki/bert-base/metrics_dev.json --cuda-device 0 --include-package modules work/wiki/bert-base/model.tar.gz work/dataset/dev_question.json
$ allennlp evaluate --output-file work/wiki/bert-base_ppl-50/metrics_dev.json --cuda-device 0 --include-package modules work/wiki/bert-base_ppl-50/model.tar.gz work/dataset/dev_question.json
$ allennlp evaluate --output-file work/wiki/bert-base_ppl-100/metrics_dev.json --cuda-device 0 --include-package modules work/wiki/bert-base_ppl-100/model.tar.gz work/dataset/dev_question.json
$ allennlp evaluate --output-file work/wiki/bert-base_ppl-150/metrics_dev.json --cuda-device 0 --include-package modules work/wiki/bert-base_ppl-150/model.tar.gz work/dataset/dev_question.json
$ allennlp evaluate --output-file work/wiki/bert-base/metrics_test.json --cuda-device 0 --include-package modules work/wiki/bert-base/model.tar.gz work/dataset/test_question.json

$ allennlp evaluate --output-file work/wiki-fp/bert-base/metrics_dev.json --cuda-device 0 --include-package modules work/wiki-fp/bert-base/model.tar.gz work/dataset/dev_question.json
$ allennlp evaluate --output-file work/wiki-fp/bert-base/metrics_test.json --cuda-device 0 --include-package modules work/wiki-fp/bert-base/model.tar.gz work/dataset/test_question.json

$ allennlp evaluate --output-file work/wiki_quiz/bert-base/metrics_dev.json --cuda-device 0 --include-package modules work/wiki_quiz/bert-base/model.tar.gz work/dataset/dev_question.json
$ allennlp evaluate --output-file work/wiki_quiz/bert-base_ppl-50/metrics_dev.json --cuda-device 0 --include-package modules work/wiki_quiz/bert-base_ppl-50/model.tar.gz work/dataset/dev_question.json
$ allennlp evaluate --output-file work/wiki_quiz/bert-base_ppl-100/metrics_dev.json --cuda-device 0 --include-package modules work/wiki_quiz/bert-base_ppl-100/model.tar.gz work/dataset/dev_question.json
$ allennlp evaluate --output-file work/wiki_quiz/bert-base_ppl-150/metrics_dev.json --cuda-device 0 --include-package modules work/wiki_quiz/bert-base_ppl-150/model.tar.gz work/dataset/dev_question.json
$ allennlp evaluate --output-file work/wiki_quiz/bert-base/metrics_test.json --cuda-device 0 --include-package modules work/wiki_quiz/bert-base/model.tar.gz work/dataset/test_question.json

$ allennlp evaluate --output-file work/wiki-fp_quiz/bert-base/metrics_dev.json --cuda-device 0 --include-package modules work/wiki-fp_quiz/bert-base/model.tar.gz work/dataset/dev_question.json
$ allennlp evaluate --output-file work/wiki-fp_quiz/bert-base/metrics_test.json --cuda-device 0 --include-package modules work/wiki-fp_quiz/bert-base/model.tar.gz work/dataset/test_question.json

$ allennlp evaluate --output-file work/wiki_quiz_mixed/bert-base/metrics_dev.json --cuda-device 0 --include-package modules work/wiki_quiz_mixed/bert-base/model.tar.gz work/dataset/dev_question.json
$ allennlp evaluate --output-file work/wiki_quiz_mixed/bert-base/metrics_test.json --cuda-device 0 --include-package modules work/wiki_quiz_mixed/bert-base/model.tar.gz work/dataset/test_question.json

$ allennlp evaluate --output-file work/wiki-fp_quiz_mixed/bert-base/metrics_dev.json --cuda-device 0 --include-package modules work/wiki-fp_quiz_mixed/bert-base/model.tar.gz work/dataset/dev_question.json
$ allennlp evaluate --output-file work/wiki-fp_quiz_mixed/bert-base/metrics_test.json --cuda-device 0 --include-package modules work/wiki-fp_quiz_mixed/bert-base/model.tar.gz work/dataset/test_question.json
```

```sh
# dev
$ allennlp print-results work -k acc mrr -m metrics_dev.json
...
model_run, acc, mrr
work/quiz/bert-base/metrics_dev.json, 0.6949458483754513, 0.7361946579351322
work/wiki-fp/bert-base/metrics_dev.json, 0.0898014440433213, 0.15248111764554081
work/wiki-fp_quiz/bert-base/metrics_dev.json, 0.7080324909747292, 0.7500329727730596
work/wiki-fp_quiz_mixed/bert-base/metrics_dev.json, 0.6389891696750902, 0.6966216080025215
work/wiki/bert-base/metrics_dev.json, 0.2739169675090253, 0.3521658570542663
work/wiki/bert-base_ppl-100/metrics_dev.json, 0.18907942238267147, 0.2665763878219825
work/wiki/bert-base_ppl-150/metrics_dev.json, 0.12364620938628158, 0.1991181737450809
work/wiki/bert-base_ppl-50/metrics_dev.json, 0.2621841155234657, 0.3393070915115439
work/wiki_quiz/bert-base/metrics_dev.json, 0.7260830324909747, 0.762442942560795
work/wiki_quiz/bert-base_ppl-100/metrics_dev.json, 0.7134476534296029, 0.753419260686055
work/wiki_quiz/bert-base_ppl-150/metrics_dev.json, 0.7066787003610109, 0.747829943357392
work/wiki_quiz/bert-base_ppl-50/metrics_dev.json, 0.7265342960288809, 0.7605760975434892
work/wiki_quiz_mixed/bert-base/metrics_dev.json, 0.6692238267148014, 0.7213154775140949

# test
$ allennlp print-results work -k acc mrr -m metrics_test.json
...
model_run, acc, mrr
work/quiz/bert-base/metrics_test.json, 0.631578947368421, 0.6830452806071231
work/wiki-fp/bert-base/metrics_test.json, 0.07943469785575048, 0.13582232366470334
work/wiki-fp_quiz/bert-base/metrics_test.json, 0.6515594541910331, 0.6999382718264708
work/wiki-fp_quiz_mixed/bert-base/metrics_test.json, 0.6016081871345029, 0.6592202992931909
work/wiki/bert-base/metrics_test.json, 0.2387914230019493, 0.3148093954396759
work/wiki_quiz/bert-base/metrics_test.json, 0.6754385964912281, 0.7150998591911956
work/wiki_quiz_mixed/bert-base/metrics_test.json, 0.6220760233918129, 0.67730507039652
```

## Prediction

```sh
# question-level evaluation
$ allennlp predict work/quiz/bert-base/model.tar.gz work/dataset/dev_question.json --output-file work/quiz/bert-base/prediction_dev_question.json --silent --cuda-device 0 --use-dataset-reader --predictor quizbowl --include-package modules
$ allennlp predict work/quiz/bert-base/model.tar.gz work/dataset/test_question.json --output-file work/quiz/bert-base/prediction_test_question.json --silent --cuda-device 0 --use-dataset-reader --predictor quizbowl --include-package modules
$ allennlp predict work/wiki_quiz/bert-base/model.tar.gz work/dataset/dev_question.json --output-file work/wiki_quiz/bert-base/prediction_dev_question.json --silent --cuda-device 0 --use-dataset-reader --predictor quizbowl --include-package modules
$ allennlp predict work/wiki_quiz/bert-base/model.tar.gz work/dataset/test_question.json --output-file work/wiki_quiz/bert-base/prediction_test_question.json --silent --cuda-device 0 --use-dataset-reader --predictor quizbowl --include-package modules

# sentence-level evaluation
$ allennlp predict work/quiz/bert-base/model.tar.gz work/dataset/dev_sentence.json --output-file work/quiz/bert-base/prediction_dev_sentence.json --silent --cuda-device 0 --use-dataset-reader --predictor quizbowl --include-package modules
$ allennlp predict work/quiz/bert-base/model.tar.gz work/dataset/test_sentence.json --output-file work/quiz/bert-base/prediction_test_sentence.json --silent --cuda-device 0 --use-dataset-reader --predictor quizbowl --include-package modules
$ allennlp predict work/wiki_quiz/bert-base/model.tar.gz work/dataset/dev_sentence.json --output-file work/wiki_quiz/bert-base/prediction_dev_sentence.json --silent --cuda-device 0 --use-dataset-reader --predictor quizbowl --include-package modules
$ allennlp predict work/wiki_quiz/bert-base/model.tar.gz work/dataset/test_sentence.json --output-file work/wiki_quiz/bert-base/prediction_test_sentence.json --silent --cuda-device 0 --use-dataset-reader --predictor quizbowl --include-package modules
```
