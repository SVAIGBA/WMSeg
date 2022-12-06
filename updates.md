# Important Updates

* Dec 5, 2022: Visit [here](https://github.com/synlp/WMSeg) for upgrades of WMSeg.
* July 18, 2020: Change the operation of `--ngram_num_threshold` and `--av_threshold` to filter out n-grams. Past: n-grams whose frequency or AV score is lower than or equal to the threshold will be excluded from the lexicon N; now: n-grams whose frequency or AV score is lower than the threshold will be excluded from the lexicon N. You can check [readme](./README.md) for detailed instruction to use the two hyper-parameters. The hyper-parameter settings are also changed in [run.sh](./run.sh) to reflect this update. This update only affects the training process; it does not affect the testing and predicting process.
* July 11, 2020: Implement the `predict` function in `wmseg_main.py`. You can use that function to segment the sentences in an input file with a pre-trained WMSeg model. See [run_sample.sh](./run_sample.sh) for the usage, and [./sample_data/sentences.txt](./sample_data/sentence.txt) for the input format.
* July 7, 2020: The release of [pre-trained WMSeg models](./models).
