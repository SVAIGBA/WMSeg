# WMSeg

This is the implementation of [Improving  Chinese  Word  Segmentation  with  Wordhood  Memory  Networks](https://www.aclweb.org/anthology/2020.acl-main.734/) at ACL2020.

Please contact us at `yhtian@uw.edu` if you have any questions.

**Visit our [homepage](https://github.com/synlp/.github) to find more our recent research and softwares for NLP (e.g., pre-trained LM, POS tagging, NER, sentiment analysis, relation extraction, datasets, etc.).**

## Citation

If you use or extend our work, please cite our paper at ACL2020.

```
@inproceedings{tian-etal-2020-improving,
    title = "Improving Chinese Word Segmentation with Wordhood Memory Networks",
    author = "Tian, Yuanhe and Song, Yan and Xia, Fei and Zhang, Tong and Wang, Yonggang",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    pages = "8274--8285",
}
```

## Requirements

Our code works with the following environment.
* `python=3.6`
* `pytorch=1.1`

## Downloading BERT, ZEN and WMSeg

In our paper, we use BERT ([paper](https://www.aclweb.org/anthology/N19-1423/)) and ZEN ([paper](https://arxiv.org/abs/1911.00720)) as the encoder.

For BERT, please download pre-trained BERT-Base Chinese from [Google](https://github.com/google-research/bert) or from [HuggingFace](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz). If you download it from Google, you need to convert the model from TensorFlow version to PyTorch version.

For ZEN, you can download the pre-trained model from [here](https://github.com/sinovation/ZEN).

For WMSeg, you can download the models we trained in our experiments from [here](https://github.com/SVAIGBA/WMSeg/tree/master/models).

## Run on Sample Data

Run `run_sample.sh` to train a model on the small sample data under the `sample_data` directory.

## Datasets

We use [SIGHAN2005](http://sighan.cs.uchicago.edu/bakeoff2005/) and [CTB6](https://catalog.ldc.upenn.edu/LDC2007T36) in our paper.

To obtain and pre-process the data, please go to `data_preprocessing` directory and run `getdata.sh`. This script will download and process the official data from SIGHAN2005. For CTB6, you need to obtain the official data first, and then put the `LDC07T36` folder under the `data_preprocessing` directory.

All processed data will appear in `data` directory.

## Training and Testing

You can find the command lines to train and test models on a specific dataset in `run.sh`.

Here are some important parameters:

* `--do_train`: train the model.
* `--do_test`: test the model.
* `--use_bert`: use BERT as encoder.
* `--use_zen`: use ZEN as encoder.
* `--bert_model`: the directory of pre-trained BERT/ZEN model.
* `--use_memory`: use key-value memory networks.
* `--decoder`: use `crf` or `softmax` as the decoder.
* `--ngram_flag`: use `av`, `dlg`, or `pmi` to construct the lexicon N.
* `--av_threshold`: when using `av` to construct the lexicon N, n-grams whose AV score is lower than the threshold will be excluded from the lexicon N.
* `--ngram_num_threshold`: n-grams whose frequency is lower than the threshold will be excluded from the lexicon N. Note that, when the threshold is set to 1, no n-gram is filtered out by its frequency. We therefore **DO NOT** recommend you to use 1 as the n-gram frequency threshold.
* `--model_name`: the name of model to save.

## Predicting

`run_sample.sh` contains the command line to segment the sentences in an input file (`./sample_data/sentence.txt`).

Here are some important parameters:

* `--do_predict`: segment the sentences using a pre-trained WMSeg model.
* `--input_file`: the file contains sentences to be segmented. Each line contains one sentence; you can refer to [a sample input file](./sample_data/sentence.txt) for the input format.
* `--output_file`: the path of the output file. Words are segmented by a space.
* `--eval_model`: the pre-trained WMSeg model to be used to segment the sentences in the input file.

## To-do List

* Release a toolkit using WMSeg with necessary APIs

You can leave comments in the `Issues` section, if you want us to implement any functions.

You can check our updates at [updates.md](./updates.md).
