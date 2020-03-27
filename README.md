# Quizbowl as a Testbed of Entity Knowledge

- All the experiments were conducted on RAIDEN.

## Install requirements

```sh
$ pip install -r requirements.txt
$ python -m spacy download en_core_web_sm
```

## Make vocabulary

```sh
$ allennlp make-vocab --serialization-dir work/vocab --include-package modules configs/make_vocab.json
```

## Training

```sh
$ mkdir -p work/quiz-ep10
$ for config_file in `ls configs/quiz-ep10`; do config=`basename $config_file .json`; qsub -v CONFIG_FILE=configs/quiz-ep10/$config.json,SERIALIZATION_DIR=work/quiz-ep10/$config -N $config train_raiden.sh; done

$ mkdir -p work/wiki-ep5
$ for config_file in `ls configs/wiki-ep5`; do config=`basename $config_file .json`; qsub -v CONFIG_FILE=configs/wiki-ep5/$config.json,SERIALIZATION_DIR=work/wiki-ep5/$config -N $config train_raiden.sh; done

$ mkdir -p work/wiki-fp-ep10
$ for config_file in `ls configs/wiki-fp-ep10`; do config=`basename $config_file .json`; qsub -v CONFIG_FILE=configs/wiki-fp-ep10/$config.json,SERIALIZATION_DIR=work/wiki-fp-ep10/$config -N $config train_raiden.sh; done

$ mkdir -p work/wiki-ep5_quiz-ep10
$ for config_file in `ls configs/quiz-ep10`; do config=`basename $config_file .json`; qsub -v MODEL_ARCHIVE=work/wiki-ep5/$config/model.tar.gz,CONFIG_FILE=configs/quiz-ep10/$config.json,SERIALIZATION_DIR=work/wiki-ep5_quiz-ep10/$config -N $config fine-tune_raiden.sh; done

$ mkdir -p work/wiki-fp-ep10_quiz-ep10
$ for config_file in `ls configs/quiz-ep10`; do config=`basename $config_file .json`; qsub -v MODEL_ARCHIVE=work/wiki-fp-ep10/$config/model.tar.gz,CONFIG_FILE=configs/quiz-ep10/$config.json,SERIALIZATION_DIR=work/wiki-fp-ep10_quiz-ep10/$config -N $config fine-tune_raiden.sh; done
```

## Evaluation

```sh
$ for dirname in `ls -d work/quiz-ep10/*`; do allennlp evaluate --output-file $dirname/metrics_train.json --cuda-device 0 --include-package modules $dirname/model.tar.gz ~/data/qanta/2018/qanta.train.2018.04.18.json; done
$ for dirname in `ls -d work/quiz-ep10/*`; do allennlp evaluate --output-file $dirname/metrics_test.json --cuda-device 0 --include-package modules $dirname/model.tar.gz ~/data/qanta/2018/qanta.test.2018.04.18.json; done
$ for dirname in `ls -d work/wiki-ep5/*`; do allennlp evaluate --output-file $dirname/metrics_test.json --cuda-device 0 --include-package modules $dirname/model.tar.gz ~/data/qanta/2018/qanta.test.2018.04.18.json; done
$ for dirname in `ls -d work/wiki-fp-ep10/*`; do allennlp evaluate --output-file $dirname/metrics_test.json --cuda-device 0 --include-package modules $dirname/model.tar.gz ~/data/qanta/2018/qanta.test.2018.04.18.json; done
$ for dirname in `ls -d work/wiki-ep5_quiz-ep10/*`; do allennlp evaluate --output-file $dirname/metrics_test.json --cuda-device 0 --include-package modules $dirname/model.tar.gz ~/data/qanta/2018/qanta.test.2018.04.18.json; done
$ for dirname in `ls -d work/wiki-fp-ep10_quiz-ep10/*`; do allennlp evaluate --output-file $dirname/metrics_test.json --cuda-device 0 --include-package modules $dirname/model.tar.gz ~/data/qanta/2018/qanta.test.2018.04.18.json; done
```

```sh
$ allennlp print-results work/quiz-ep10 -k acc mrr -m metrics_train.json
...
model_run, acc, mrr
work/quiz-ep10/bert-base/metrics_train.json, 0.9983086418659842, 0.9991528450589241
work/quiz-ep10/bert-plain/metrics_train.json, 0.9816518635932948, 0.9896869642027022
work/quiz-ep10/roberta-base/metrics_train.json, 0.998007562407573, 0.9989964018334462
work/quiz-ep10/xlnet-base/metrics_train.json, 0.9991764591284635, 0.9995867536901637

$ allennlp print-results work/quiz-ep10 -k acc mrr -m metrics_test.json
...
model_run, acc, mrr
work/quiz-ep10/bert-base/metrics_test.json, 0.6128167641325536, 0.6653197468140437
work/quiz-ep10/bert-plain/metrics_test.json, 0.37451267056530213, 0.4569582708804463
work/quiz-ep10/roberta-base/metrics_test.json, 0.5716374269005848, 0.6327016924092179
work/quiz-ep10/xlnet-base/metrics_test.json, 0.642056530214425, 0.6900900767328214

$ allennlp print-results work/wiki-ep5 -k acc mrr -m metrics_test.json
...
model_run, acc, mrr
work/wiki-ep5/bert-base/metrics_test.json, 0.24317738791423002, 0.32111390582766913
work/wiki-ep5/bert-plain/metrics_test.json, 0.1895711500974659, 0.2602922100188904
work/wiki-ep5/roberta-base/metrics_test.json, 0.2395224171539961, 0.31546673108959755
work/wiki-ep5/xlnet-base/metrics_test.json, 0.2548732943469786, 0.3338834507714006

$ allennlp print-results work/wiki-ep5_quiz-ep10 -k acc mrr -m metrics_test.json
...
model_run, acc, mrr
work/wiki-ep5_quiz-ep10/bert-base/metrics_test.json, 0.6408382066276803, 0.6883744886744092
work/wiki-ep5_quiz-ep10/bert-plain/metrics_test.json, 0.5633528265107213, 0.6218274796915333
work/wiki-ep5_quiz-ep10/roberta-base/metrics_test.json, 0.6437621832358674, 0.6927164067998964
work/wiki-ep5_quiz-ep10/xlnet-base/metrics_test.json, 0.675682261208577, 0.7157083809027198

$ allennlp print-results work/wiki-fp-ep10 -k acc mrr -m metrics_test.json
...
model_run, acc, mrr
work/wiki-fp-ep10/bert-base/metrics_test.json, 0.06700779727095517, 0.12070249195335901
work/wiki-fp-ep10/bert-plain/metrics_test.json, 0.0, 0.0002779683636476015
work/wiki-fp-ep10/roberta-base/metrics_test.json, 0.06505847953216375, 0.12057401480962891
work/wiki-fp-ep10/xlnet-base/metrics_test.json, 0.08503898635477583, 0.13603812194707102

$ allennlp print-results work/wiki-fp-ep10_quiz-ep10 -k acc mrr -m metrics_test.json
...
model_run, acc, mrr
work/wiki-fp-ep10_quiz-ep10/bert-base/metrics_test.json, 0.6362085769980507, 0.6858038216771206
work/wiki-fp-ep10_quiz-ep10/bert-plain/metrics_test.json, 0.0026803118908382065, 0.00808503616512272
work/wiki-fp-ep10_quiz-ep10/roberta-base/metrics_test.json, 0.5977095516569201, 0.6556458677697136
work/wiki-fp-ep10_quiz-ep10/xlnet-base/metrics_test.json, 0.6649610136452242, 0.7068973945827745
```

## Prediction

```sh
$ allennlp predict work/quiz-ep10/xlnet-base/model.tar.gz ~/data/qanta/2018/qanta.dev.2018.04.18.json --output-file work/quiz-ep10/xlnet-base/prediction_dev.json --silent --cuda-device 0 --use-dataset-reader --predictor quizbowl --include-package modules
$ allennlp predict work/quiz-ep10/xlnet-base/model.tar.gz ~/data/qanta/2018/qanta.test.2018.04.18.json --output-file work/quiz-ep10/xlnet-base/prediction_test.json --silent --cuda-device 0 --use-dataset-reader --predictor quizbowl --include-package modules
$ allennlp predict work/wiki-ep5_quiz-ep10/xlnet-base/model.tar.gz ~/data/qanta/2018/qanta.dev.2018.04.18.json --output-file work/wiki-ep5_quiz-ep10/xlnet-base/prediction_dev.json --silent --cuda-device 0 --use-dataset-reader --predictor quizbowl --include-package modules
$ allennlp predict work/wiki-ep5_quiz-ep10/xlnet-base/model.tar.gz ~/data/qanta/2018/qanta.test.2018.04.18.json --output-file work/wiki-ep5_quiz-ep10/xlnet-base/prediction_test.json --silent --cuda-device 0 --use-dataset-reader --predictor quizbowl --include-package modules

$ allennlp predict work/quiz-ep10/bert-base/model.tar.gz ~/data/qanta/2018/qanta.dev.2018.04.18.json --output-file work/quiz-ep10/bert-base/prediction_dev_split.json --silent --cuda-device 0 --use-dataset-reader --overrides '{"validation_dataset_reader": {"text_unit": "sentence"}}' --predictor quizbowl --include-package modules
$ allennlp predict work/quiz-ep10/xlnet-base/model.tar.gz ~/data/qanta/2018/qanta.dev.2018.04.18.json --output-file work/quiz-ep10/xlnet-base/prediction_dev_split.json --silent --cuda-device 0 --use-dataset-reader --overrides '{"validation_dataset_reader": {"text_unit": "sentence"}}' --predictor quizbowl --include-package modules
$ allennlp predict work/quiz-ep10/xlnet-base/model.tar.gz ~/data/qanta/2018/qanta.test.2018.04.18.json --output-file work/quiz-ep10/xlnet-base/prediction_test_split.json --silent --cuda-device 0 --use-dataset-reader --overrides '{"validation_dataset_reader": {"text_unit": "sentence"}}' --predictor quizbowl --include-package modules
$ allennlp predict work/wiki-ep5_quiz-ep10/xlnet-base/model.tar.gz ~/data/qanta/2018/qanta.dev.2018.04.18.json --output-file work/wiki-ep5_quiz-ep10/xlnet-base/prediction_dev_split.json --silent --cuda-device 0 --use-dataset-reader --overrides '{"validation_dataset_reader": {"text_unit": "sentence"}}' --predictor quizbowl --include-package modules
$ allennlp predict work/wiki-ep5_quiz-ep10/xlnet-base/model.tar.gz ~/data/qanta/2018/qanta.test.2018.04.18.json --output-file work/wiki-ep5_quiz-ep10/xlnet-base/prediction_test_split.json --silent --cuda-device 0 --use-dataset-reader --overrides '{"validation_dataset_reader": {"text_unit": "sentence"}}' --predictor quizbowl --include-package modules
```
