# Quizbowl as a Testbed of Entity Knowledge

- All the experiments were conducted on RAIDEN.

## Make datasets

### Wiki datasets

```sh
$ ~/Repos/wikiextractor/WikiExtractor.py --output work/wikiextractor --bytes 100G --json --namespaces 0 --no_templates --processes 4 ~/data/qanta/2018/enwiki-20180420-pages-articles-multistream.xml.bz2
```

### Quizbowl datasets

```sh
$ python make_quizbowl_dataset.py --dataset_file ~/data/qanta/2018/qanta.train.2018.04.18.json --output_file work/dataset/quizbowl/train_question.json --text_unit question
$ python make_quizbowl_dataset.py --dataset_file ~/data/qanta/2018/qanta.train.2018.04.18.json --output_file work/dataset/quizbowl/train_sentence.json --text_unit sentence
$ python make_quizbowl_dataset.py --dataset_file ~/data/qanta/2018/qanta.dev.2018.04.18.json --output_file work/dataset/quizbowl/dev_question.json --text_unit question
$ python make_quizbowl_dataset.py --dataset_file ~/data/qanta/2018/qanta.test.2018.04.18.json --output_file work/dataset/quizbowl/test_question.json --text_unit question

$ cat work/dataset/quizbowl/train_question.json|jq -r '.entity'|sort|uniq > work/dataset/quizbowl/train_entities.txt
$ python make_wiki_dataset.py --dataset_file work/wikiextractor/AA/wiki_00 --title_list_file work/dataset/quizbowl/train_entities.txt --output_file work/dataset/quizbowl/wiki_sentence_blingfire.json --text_unit sentence --sent_splitter blingfire
$ cat work/dataset/quizbowl/train_sentence.json work/dataset/quizbowl/wiki_sentence_blingfire.json > work/dataset/quizbowl/train_sentence_wiki_sentence_blingfire.json
```

### TriviaQA datasets

```sh
$ python make_triviaqa_dataset.py --dataset_file ~/data/triviaqa/triviaqa-unfiltered/unfiltered-web-train.json --output_file work/dataset/triviaqa/train_question.json
$ python make_triviaqa_dataset.py --dataset_file ~/data/triviaqa/qa/wikipedia-dev.json --output_file work/dataset/triviaqa/dev_question.json
$ python make_triviaqa_dataset.py --dataset_file ~/data/triviaqa/qa/wikipedia-dev.json --output_file work/dataset/triviaqa/eval_question.json --ignore_answer
$ python make_triviaqa_dataset.py --dataset_file ~/data/triviaqa/qa/wikipedia-test-without-answers.json --output_file work/dataset/triviaqa/test_question.json --ignore_answer

$ cat work/dataset/triviaqa/train_question.json|jq -r '.entity'|sort|uniq > work/dataset/triviaqa/train_entities.txt
$ python make_wiki_dataset.py --dataset_file work/wikiextractor/AA/wiki_00 --title_list_file work/dataset/triviaqa/train_entities.txt --output_file work/dataset/triviaqa/wiki_sentence_blingfire.json --text_unit sentence --sent_splitter blingfire
$ cat work/dataset/triviaqa/train_question.json work/dataset/triviaqa/wiki_sentence_blingfire.json > work/dataset/triviaqa/train_question_wiki_sentence_blingfire.json
$ cat work/dataset/triviaqa/train_question.json work/dataset/triviaqa/wiki-fp_sentence_blingfire.json > work/dataset/triviaqa/train_question_wiki-fp_sentence_blingfire.json
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

```
$ mkdir -p work/triviaqa/vocab
$ allennlp train --dry-run --serialization-dir work/triviaqa/vocab/bert-base --include-package modules configs/triviaqa/make_vocab/bert-base.json
$ allennlp train --dry-run --serialization-dir work/triviaqa/vocab/roberta-base --include-package modules configs/triviaqa/make_vocab/roberta-base.json
$ allennlp train --dry-run --serialization-dir work/triviaqa/vocab/xlnet-base --include-package modules configs/triviaqa/make_vocab/xlnet-base.json
```

## Training

### Quizbowl

```sh
$ mkdir work/quizbowl/quiz
$ qsub -v CONFIG_FILE=configs/quizbowl/quiz/bert-base.json,SERIALIZATION_DIR=work/quizbowl/quiz/bert-base -N quiz train_raiden.sh
$ qsub -v CONFIG_FILE=configs/quizbowl/quiz/roberta-base.json,SERIALIZATION_DIR=work/quizbowl/quiz/roberta-base -N quiz train_raiden.sh
$ qsub -v CONFIG_FILE=configs/quizbowl/quiz/xlnet-base.json,SERIALIZATION_DIR=work/quizbowl/quiz/xlnet-base -N quiz train_raiden.sh

$ mkdir work/quizbowl/wiki
$ qsub -v CONFIG_FILE=configs/quizbowl/wiki/bert-base.json,SERIALIZATION_DIR=work/quizbowl/wiki/bert-base -N wiki train_raiden.sh
$ qsub -v CONFIG_FILE=configs/quizbowl/wiki/roberta-base.json,SERIALIZATION_DIR=work/quizbowl/wiki/roberta-base -N wiki train_raiden.sh
$ qsub -v CONFIG_FILE=configs/quizbowl/wiki/xlnet-base.json,SERIALIZATION_DIR=work/quizbowl/wiki/xlnet-base -N wiki -jc gpu-container_g1.72h train_raiden.sh

$ mkdir work/quizbowl/quiz_and_wiki
$ qsub -v CONFIG_FILE=configs/quizbowl/quiz_and_wiki/bert-base.json,SERIALIZATION_DIR=work/quizbowl/quiz_and_wiki/bert-base -N quiz_and_wiki -jc gpu-container_g1.72h train_raiden.sh
$ qsub -v CONFIG_FILE=configs/quizbowl/quiz_and_wiki/roberta-base.json,SERIALIZATION_DIR=work/quizbowl/quiz_and_wiki/roberta-base -N quiz_and_wiki -jc gpu-container_g1.72h train_raiden.sh
$ qsub -v CONFIG_FILE=configs/quizbowl/quiz_and_wiki/xlnet-base.json,SERIALIZATION_DIR=work/quizbowl/quiz_and_wiki/xlnet-base -N quiz_and_wiki -jc gpu-container_g1.72h train_raiden.sh

$ mkdir work/quizbowl/quiz_to_wiki
$ qsub -v CONFIG_FILE=configs/quizbowl/quiz_to_wiki/bert-base.json,SERIALIZATION_DIR=work/quizbowl/quiz_to_wiki/bert-base -N quiz_to_wiki train_raiden.sh
$ qsub -v CONFIG_FILE=configs/quizbowl/quiz_to_wiki/roberta-base.json,SERIALIZATION_DIR=work/quizbowl/quiz_to_wiki/roberta-base -N quiz_to_wiki train_raiden.sh
$ qsub -v CONFIG_FILE=configs/quizbowl/quiz_to_wiki/xlnet-base.json,SERIALIZATION_DIR=work/quizbowl/quiz_to_wiki/xlnet-base -N quiz_to_wiki -jc gpu-container_g1.72h train_raiden.sh

$ mkdir work/quizbowl/wiki_to_quiz
$ qsub -v CONFIG_FILE=configs/quizbowl/wiki_to_quiz/bert-base.json,SERIALIZATION_DIR=work/quizbowl/wiki_to_quiz/bert-base -N wiki_to_quiz train_raiden.sh
$ qsub -v CONFIG_FILE=configs/quizbowl/wiki_to_quiz/roberta-base.json,SERIALIZATION_DIR=work/quizbowl/wiki_to_quiz/roberta-base -N wiki_to_quiz train_raiden.sh
$ qsub -v CONFIG_FILE=configs/quizbowl/wiki_to_quiz/xlnet-base.json,SERIALIZATION_DIR=work/quizbowl/wiki_to_quiz/xlnet-base -N wiki_to_quiz train_raiden.sh
```

### TriviaQA

```sh
$ mkdir work/triviaqa/quiz
qsub -v CONFIG_FILE=configs/triviaqa/quiz/bert-base.json,SERIALIZATION_DIR=work/triviaqa/quiz/bert-base -N quiz train_raiden.sh
qsub -v CONFIG_FILE=configs/triviaqa/quiz/roberta-base.json,SERIALIZATION_DIR=work/triviaqa/quiz/roberta-base -N quiz train_raiden.sh
qsub -v CONFIG_FILE=configs/triviaqa/quiz/xlnet-base.json,SERIALIZATION_DIR=work/triviaqa/quiz/xlnet-base -N quiz train_raiden.sh

$ mkdir work/triviaqa/wiki
$ qsub -v CONFIG_FILE=configs/triviaqa/wiki/bert-base.json,SERIALIZATION_DIR=work/triviaqa/wiki/bert-base -N wiki train_raiden.sh
$ qsub -v CONFIG_FILE=configs/triviaqa/wiki/roberta-base.json,SERIALIZATION_DIR=work/triviaqa/wiki/roberta-base -N wiki train_raiden.sh
$ qsub -v CONFIG_FILE=configs/triviaqa/wiki/xlnet-base.json,SERIALIZATION_DIR=work/triviaqa/wiki/xlnet-base -N wiki -jc gpu-container_g1.72h train_raiden.sh

$ mkdir work/triviaqa/quiz_and_wiki
$ qsub -v CONFIG_FILE=configs/triviaqa/quiz_and_wiki/bert-base.json,SERIALIZATION_DIR=work/triviaqa/quiz_and_wiki/bert-base -N quiz_and_wiki train_raiden.sh
$ qsub -v CONFIG_FILE=configs/triviaqa/quiz_and_wiki/roberta-base.json,SERIALIZATION_DIR=work/triviaqa/quiz_and_wiki/roberta-base -N quiz_and_wiki train_raiden.sh
$ qsub -v CONFIG_FILE=configs/triviaqa/quiz_and_wiki/xlnet-base.json,SERIALIZATION_DIR=work/triviaqa/quiz_and_wiki/xlnet-base -N quiz_and_wiki -jc gpu-container_g1.72h train_raiden.sh

$ mkdir work/triviaqa/quiz_to_wiki
$ qsub -v CONFIG_FILE=configs/triviaqa/quiz_to_wiki/bert-base.json,SERIALIZATION_DIR=work/triviaqa/quiz_to_wiki/bert-base -N quiz_to_wiki train_raiden.sh
$ qsub -v CONFIG_FILE=configs/triviaqa/quiz_to_wiki/roberta-base.json,SERIALIZATION_DIR=work/triviaqa/quiz_to_wiki/roberta-base -N quiz_to_wiki train_raiden.sh
$ qsub -v CONFIG_FILE=configs/triviaqa/quiz_to_wiki/xlnet-base.json,SERIALIZATION_DIR=work/triviaqa/quiz_to_wiki/xlnet-base -N quiz_to_wiki -jc gpu-container_g1.72h train_raiden.sh

$ mkdir work/triviaqa/wiki_to_quiz
$ qsub -v CONFIG_FILE=configs/triviaqa/wiki_to_quiz/bert-base.json,SERIALIZATION_DIR=work/triviaqa/wiki_to_quiz/bert-base -N wiki_to_quiz train_raiden.sh
$ qsub -v CONFIG_FILE=configs/triviaqa/wiki_to_quiz/roberta-base.json,SERIALIZATION_DIR=work/triviaqa/wiki_to_quiz/roberta-base -N wiki_to_quiz train_raiden.sh
$ qsub -v CONFIG_FILE=configs/triviaqa/wiki_to_quiz/xlnet-base.json,SERIALIZATION_DIR=work/triviaqa/wiki_to_quiz/xlnet-base -N wiki_to_quiz train_raiden.sh
```

---

## Prediction

### Quizbowl

```sh
# make prediction on test files
$ for dir in work/quizbowl/*/*; do allennlp predict $dir/model.tar.gz work/dataset/quizbowl/test_question.json --output-file $dir/prediction_test_question.json --silent --cuda-device 0 --use-dataset-reader --predictor quiz --include-package modules; done

$ for dir in work/quizbowl/*/*; do echo $dir; python print_result.py --input_file $dir/prediction_test_question.json; echo ''; done
work/quizbowl/quiz/bert-base
# All
Acc: 0.633 (upper bound: 0.842)
MRR: 0.683

work/quizbowl/quiz/roberta-base
# All
Acc: 0.585 (upper bound: 0.842)
MRR: 0.644

work/quizbowl/quiz/xlnet-base
# All
Acc: 0.650 (upper bound: 0.842)
MRR: 0.696

work/quizbowl/quiz_and_wiki/bert-base
# All
Acc: 0.634 (upper bound: 0.842)
MRR: 0.686

work/quizbowl/quiz_and_wiki/roberta-base
# All
Acc: 0.571 (upper bound: 0.842)
MRR: 0.634

work/quizbowl/quiz_and_wiki/xlnet-base
# All
Acc: 0.653 (upper bound: 0.842)
MRR: 0.699

work/quizbowl/quiz_to_wiki/bert-base
# All
Acc: 0.428 (upper bound: 0.842)
MRR: 0.510

work/quizbowl/quiz_to_wiki/roberta-base
# All
Acc: 0.394 (upper bound: 0.842)
MRR: 0.480

work/quizbowl/quiz_to_wiki/xlnet-base
# All
Acc: 0.469 (upper bound: 0.842)
MRR: 0.545

work/quizbowl/wiki/roberta-base
# All
Acc: 0.191 (upper bound: 0.842)
MRR: 0.264

work/quizbowl/wiki/xlnet-base
# All
Acc: 0.274 (upper bound: 0.842)
MRR: 0.351

work/quizbowl/wiki_to_quiz/bert-base
# All
Acc: 0.663 (upper bound: 0.842)
MRR: 0.707

work/quizbowl/wiki_to_quiz/roberta-base
# All
Acc: 0.650 (upper bound: 0.842)
MRR: 0.695

work/quizbowl/wiki_to_quiz/xlnet-base
# All
Acc: 0.690 (upper bound: 0.842)
MRR: 0.727
```

### TriviaQA

```sh
$ allennlp predict work/triviaqa/wiki_to_quiz/bert-base/model.tar.gz work/dataset/triviaqa/dev_question.json --output-file work/triviaqa/wiki_to_quiz/bert-base/prediction_dev_question.json --silent --cuda-device 0 --use-dataset-reader --predictor quiz --include-package modules
$ python print_result.py --input_file work/triviaqa/wiki_to_quiz/bert-base/prediction_dev_question.json
# All
Acc: 0.436 (upper bound: 0.798)
MRR: 0.497

# make prediction on evaluation files
$ for dir in work/triviaqa/*/*; do allennlp predict $dir/model.tar.gz work/dataset/triviaqa/eval_question.json --output-file $dir/prediction_eval_question.json --silent --cuda-device 0 --use-dataset-reader --predictor quiz --include-package modules; done

# convert evaluation results to TriviaQA format
$ for dir in work/triviaqa/*/*; do echo $dir; python convert_triviaqa_prediction.py --input_file $dir/prediction_eval_question.json --output_file $dir/prediction_eval_question_converted.json --postprocess_answers; done

# print evaluation results
$ cd ~/Repos/triviaqa
$ for dir in ~/Projects/quizbowl_beta/work/triviaqa/*/*; do echo $dir; python -m evaluation.triviaqa_evaluation --dataset_file ~/data/triviaqa/qa/wikipedia-dev.json --prediction_file $dir/prediction_eval_question_converted.json|grep -v "em=0:"; done
/uge_mnt/home/m-suzuki/Projects/quizbowl_beta/work/triviaqa/quiz/bert-base
{'exact_match': 25.534842987614162, 'f1': 28.416237370355127, 'common': 7993, 'denominator': 7993, 'pred_len': 7993, 'gold_len': 7993}
/uge_mnt/home/m-suzuki/Projects/quizbowl_beta/work/triviaqa/quiz/roberta-base
{'exact_match': 22.95758788940323, 'f1': 25.729875029262033, 'common': 7993, 'denominator': 7993, 'pred_len': 7993, 'gold_len': 7993}
/uge_mnt/home/m-suzuki/Projects/quizbowl_beta/work/triviaqa/quiz/xlnet-base
{'exact_match': 30.78944076066558, 'f1': 33.70652360831268, 'common': 7993, 'denominator': 7993, 'pred_len': 7993, 'gold_len': 7993}
/uge_mnt/home/m-suzuki/Projects/quizbowl_beta/work/triviaqa/quiz_and_wiki/bert-base
{'exact_match': 38.39609658451145, 'f1': 41.182639199061434, 'common': 7993, 'denominator': 7993, 'pred_len': 7993, 'gold_len': 7993}
/uge_mnt/home/m-suzuki/Projects/quizbowl_beta/work/triviaqa/quiz_and_wiki/roberta-base
{'exact_match': 37.320155135743775, 'f1': 40.16680044172847, 'common': 7993, 'denominator': 7993, 'pred_len': 7993, 'gold_len': 7993}
/uge_mnt/home/m-suzuki/Projects/quizbowl_beta/work/triviaqa/quiz_and_wiki/xlnet-base
{'exact_match': 37.345177029901166, 'f1': 40.35863200050341, 'common': 7993, 'denominator': 7993, 'pred_len': 7993, 'gold_len': 7993}
/uge_mnt/home/m-suzuki/Projects/quizbowl_beta/work/triviaqa/quiz_to_wiki/bert-base
{'exact_match': 14.800450394094833, 'f1': 17.725529054817198, 'common': 7993, 'denominator': 7993, 'pred_len': 7993, 'gold_len': 7993}
/uge_mnt/home/m-suzuki/Projects/quizbowl_beta/work/triviaqa/quiz_to_wiki/roberta-base
{'exact_match': 14.124859251845365, 'f1': 17.468389924506425, 'common': 7993, 'denominator': 7993, 'pred_len': 7993, 'gold_len': 7993}
/uge_mnt/home/m-suzuki/Projects/quizbowl_beta/work/triviaqa/quiz_to_wiki/xlnet-base
{'exact_match': 16.0890779432003, 'f1': 19.02875687499735, 'common': 7993, 'denominator': 7993, 'pred_len': 7993, 'gold_len': 7993}
/uge_mnt/home/m-suzuki/Projects/quizbowl_beta/work/triviaqa/wiki/bert-base
{'exact_match': 11.334918053296635, 'f1': 13.264109703494162, 'common': 7993, 'denominator': 7993, 'pred_len': 7993, 'gold_len': 7993}
/uge_mnt/home/m-suzuki/Projects/quizbowl_beta/work/triviaqa/wiki/roberta-base
{'exact_match': 11.535093206555736, 'f1': 13.895030626545694, 'common': 7993, 'denominator': 7993, 'pred_len': 7993, 'gold_len': 7993}
/uge_mnt/home/m-suzuki/Projects/quizbowl_beta/work/triviaqa/wiki/xlnet-base
{'exact_match': 13.311647691730265, 'f1': 15.719322217397755, 'common': 7993, 'denominator': 7993, 'pred_len': 7993, 'gold_len': 7993}
/uge_mnt/home/m-suzuki/Projects/quizbowl_beta/work/triviaqa/wiki_to_quiz/bert-base
{'exact_match': 42.674840485424745, 'f1': 46.222435348545645, 'common': 7993, 'denominator': 7993, 'pred_len': 7993, 'gold_len': 7993}
/uge_mnt/home/m-suzuki/Projects/quizbowl_beta/work/triviaqa/wiki_to_quiz/roberta-base
{'exact_match': 42.462154385086954, 'f1': 45.93530032724163, 'common': 7993, 'denominator': 7993, 'pred_len': 7993, 'gold_len': 7993}
/uge_mnt/home/m-suzuki/Projects/quizbowl_beta/work/triviaqa/wiki_to_quiz/xlnet-base
{'exact_match': 41.87413987238834, 'f1': 45.241961633178896, 'common': 7993, 'denominator': 7993, 'pred_len': 7993, 'gold_len': 7993}
```

```sh
# make prediction on test set
$ cd ~/Projects/quizbowl_beta/
$ allennlp predict work/triviaqa/wiki_to_quiz/bert-base/model.tar.gz work/dataset/triviaqa/test_question.json --output-file work/triviaqa/wiki_to_quiz/bert-base/prediction_test_question.json --silent --cuda-device 0 --use-dataset-reader --predictor quiz --include-package modules

# convert the test result to TriviaQA format
$ python convert_triviaqa_prediction.py --input_file work/triviaqa/wiki_to_quiz/bert-base/prediction_test_question.json --output_file work/triviaqa/wiki_to_quiz/bert-base/prediction_test_question_converted.json --postprocess_answers

# prepare a submission file for TriviaQA leader's board
$ cd work/triviaqa/wiki_to_quiz/bert-base
$ cp prediction_test_question_converted.json predictions.json && zip -j submission.zip predictions.json && rm predictions.json
```
