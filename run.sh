mkdir logs

# important parameters
# do_train: train the model
# do_test: test the model
# use_bert: use BERT as encoder
# use_zen: use ZEN as encoder
# bert_model: the directory of BERT/ZEN model
# use_memory: use memory
# decoder: crf or softmax
# ngram_flag: use av, dlg, or pmi to construct the lexicon N
# model_name: the name of model to save

# train
# Updates July 18, 2020
# We update the hyper-parameters of '--ngram_num_threshold' and '--av_threshold' in all command lines of training a model to reflect the updates on July 18, 2020 (you can check the detials in updates.md). This change does not affect the testing and predicting process.

# msr bert
python wmseg_main.py --do_train --train_data_path=./data/msr/train.tsv --eval_data_path=./data/msr/test.tsv --use_bert --bert_model=/path/to/bert/model --use_memory --decoder=crf --max_seq_length=300 --max_ngram_size=300 --train_batch_size=16 --eval_batch_size=16 --num_train_epochs=50 --warmup_proportion=0.1 --learning_rate=1e-5 --ngram_num_threshold=5 --patient=10 --ngram_flag=av --av_threshold=6 --model_name=msr_bert_memory_crf

# msr zen
python wmseg_main.py --do_train --train_data_path=./data/msr/train.tsv --eval_data_path=./data/msr/test.tsv --use_zen --bert_model=/path/to/zen/model --use_memory --decoder=crf --max_seq_length=300 --max_ngram_size=300 --train_batch_size=16 --eval_batch_size=16 --num_train_epochs=50 --warmup_proportion=0.1 --learning_rate=1e-5 --ngram_num_threshold=5 --patient=10 --ngram_flag=av --av_threshold=6 --model_name=msr_zen_memory_crf


# pku bert
python wmseg_main.py --do_train --train_data_path=./data/pku/train.tsv --eval_data_path=./data/pku/test.tsv --use_bert --bert_model=/path/to/bert/model --use_memory --decoder=crf --max_seq_length=300 --max_ngram_size=300 --train_batch_size=16 --eval_batch_size=16 --num_train_epochs=50 --warmup_proportion=0.1 --learning_rate=1e-5 --ngram_num_threshold=3 --patient=10 --ngram_flag=av --av_threshold=3 --model_name=pku_bert_memory_crf

# pku zen
python wmseg_main.py --do_train --train_data_path=./data/pku/train.tsv --eval_data_path=./data/pku/test.tsv --use_zen --bert_model=/path/to/zen/model --use_memory --decoder=crf --max_seq_length=300 --max_ngram_size=300 --train_batch_size=16 --eval_batch_size=16 --num_train_epochs=50 --warmup_proportion=0.1 --learning_rate=1e-5 --ngram_num_threshold=3 --patient=10 --ngram_flag=av --av_threshold=3 --model_name=pku_zen_memory_crf


# as bert
python wmseg_main.py --do_train --train_data_path=./data/as/train.tsv --eval_data_path=./data/as/test.tsv --use_bert --bert_model=/path/to/bert/model --use_memory --decoder=crf --max_seq_length=300 --max_ngram_size=300 --train_batch_size=16 --eval_batch_size=16 --num_train_epochs=50 --warmup_proportion=0.1 --learning_rate=1e-5 --ngram_num_threshold=9 --patient=10 --ngram_flag=av --av_threshold=6 --model_name=as_bert_memory_crf

# as zen
python wmseg_main.py --do_train --train_data_path=./data/as/train.tsv --eval_data_path=./data/as/test.tsv --use_zen --bert_model=/path/to/zen/model --use_memory --decoder=crf --max_seq_length=300 --max_ngram_size=300 --train_batch_size=16 --eval_batch_size=16 --num_train_epochs=50 --warmup_proportion=0.1 --learning_rate=1e-5 --ngram_num_threshold=9 --patient=10 --ngram_flag=av --av_threshold=6 --model_name=as_zen_memory_crf


# cityu bert
python wmseg_main.py --do_train --train_data_path=./data/cityu/train.tsv --eval_data_path=./data/cityu/test.tsv --use_bert --bert_model=/path/to/bert/model --use_memory --decoder=crf --max_seq_length=300 --max_ngram_size=300 --train_batch_size=16 --eval_batch_size=16 --num_train_epochs=50 --warmup_proportion=0.1 --learning_rate=1e-5 --ngram_num_threshold=3 --patient=10 --ngram_flag=av --av_threshold=3 --model_name=cityu_bert_memory_crf

# cityu zen
python wmseg_main.py --do_train --train_data_path=./data/cityu/train.tsv --eval_data_path=./data/cityu/test.tsv --use_zen --bert_model=/path/to/zen/model --use_memory --decoder=crf --max_seq_length=300 --max_ngram_size=300 --train_batch_size=16 --eval_batch_size=16 --num_train_epochs=50 --warmup_proportion=0.1 --learning_rate=1e-5 --ngram_num_threshold=3 --patient=10 --ngram_flag=av --av_threshold=3 --model_name=cityu_zen_memory_crf


# ctb6 bert
python wmseg_main.py --do_train --train_data_path=./data/ctb6/train.tsv --eval_data_path=./data/ctb6/test.tsv --use_bert --bert_model=/path/to/bert/model --use_memory --decoder=crf --max_seq_length=300 --max_ngram_size=300 --train_batch_size=16 --eval_batch_size=16 --num_train_epochs=50 --warmup_proportion=0.1 --learning_rate=1e-5 --ngram_num_threshold=2 --patient=10 --ngram_flag=av --av_threshold=3 --model_name=ctb6_bert_memory_crf

# ctb6 zen
python wmseg_main.py --do_train --train_data_path=./data/ctb6/train.tsv --eval_data_path=./data/ctb6/test.tsv --use_zen --bert_model=/path/to/zen/model --use_memory --decoder=crf --max_seq_length=300 --max_ngram_size=300 --train_batch_size=16 --eval_batch_size=16 --num_train_epochs=50 --warmup_proportion=0.1 --learning_rate=1e-5 --ngram_num_threshold=2 --patient=10 --ngram_flag=av --av_threshold=3 --model_name=ctb6_zen_memory_crf


# test
python wmseg_main.py --do_test --eval_data_path=./data/dataset_name/test.tsv --eval_model=./models/model_name/model.pt

