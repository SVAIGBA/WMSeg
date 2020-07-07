# WMSeg

This is the implementation of [Improving  Chinese  WordSegmentation  with  Wordhood  Memory  Networks](https://www.aclweb.org/anthology/2020.acl-main.734/) at ACL2020.

We will keep updating this repository these days.

## Citation

If you use or extend our work, please cite our paper at ACL2020.

```
@inproceedings{tian-etal-2020-joint,
    title = "Joint Chinese Word Segmentation and Part-of-speech Tagging via Two-way Attentions of Auto-analyzed Knowledge",
    author = "Tian, Yuanhe and Song, Yan and Ao, Xiang and Xia, Fei and Quan, Xiaojun and Zhang, Tong and Wang, Yonggang",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    pages = "8286--8296",
}
```

## Requirements

Our code works with the following environment.
* `python=3.6`
* `pytorch=1.1`

## Downloading BERT and ZEN

In our paper, we use BERT ([paper](https://www.aclweb.org/anthology/N19-1423/)) and ZEN ([paper](https://arxiv.org/abs/1911.00720)) as the encoder.

For BERT, please download pre-trained BERT-Base Chinese from [Google](https://github.com/google-research/bert) or from [HuggingFace](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz). If you download it from Google, you need to convert the model from TensorFlow version to PyTorch version.

For ZEN, you can download the pre-trained model form [here](https://github.com/sinovation/ZEN).

## Run on Sample Data

Run `run_sample.sh` to train a model on the small sample data under the `sample_data` directory.

## Datasets

We use [SIGHAN2005](http://sighan.cs.uchicago.edu/bakeoff2005/) and [CTB6](https://catalog.ldc.upenn.edu/LDC2007T36) in our paper.

To obtain and pre-process the data, please go to `data_preprocessing` directory and run `getdata.sh`. This script will download and process the official data from SIGHAN2005. For CTB6, you need to obtain the official data first, and then put the `LDC07T36` folder under the `data_preprocessing` directory.

All processed data will appear in `data` directory.

## Training and Testing

You can find the command lines to train and test model on a specific dataset in `run.sh`.

Here are some important parameters:

* `--do_train`: train the model
* `--do_test`: test the model
* `--use_bert`: use BERT as encoder
* `--use_zen`: use ZEN as encoder
* `--bert_model`: the directory of pre-trained BERT/ZEN model
* `--use_memory`: use memory
* `--decoder`: use `crf` or `softmax` as the decoder
* `--ngram_flag`: use `av`, `dlg`, or `pmi` to construct the lexicon N
* `--model_name`: the name of model to save 

