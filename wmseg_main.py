from __future__ import absolute_import, division, print_function

import argparse
import json
import logging
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

from tqdm import tqdm, trange
from seqeval.metrics import classification_report
from wmseg_helper import get_word2id, get_gram2id
from wmseg_eval import eval_sentence, cws_evaluate_word_PRF, cws_evaluate_OOV
from wmseg_model import WMSeg
import datetime


def train(args):

    if args.use_bert and args.use_zen:
        raise ValueError('We cannot use both BERT and ZEN')

    if not os.path.exists('./logs'):
        os.mkdir('./logs')

    now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    log_file_name = './logs/log-' + now_time
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        filename=log_file_name,
                        filemode='w',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)

    logger = logging.getLogger(__name__)

    logger.info(vars(args))

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not os.path.exists('./models'):
        os.mkdir('./models')

    if args.model_name is None:
        raise Warning('model name is not specified, the model will NOT be saved!')
    output_model_dir = os.path.join('./models', args.model_name + '_' + now_time)

    word2id = get_word2id(args.train_data_path)
    logger.info('# of word in train: %d: ' % len(word2id))

    if args.use_memory:
        if args.ngram_num_threshold <= 1:
            raise Warning('The threshold of n-gram frequency is set to %d. '
                          'No n-grams will be filtered out by frequency. '
                          'We only filter out n-grams whose frequency is lower than that threshold!'
                          % args.ngram_num_threshold)
        gram2id = get_gram2id(args.train_data_path, args.eval_data_path,
                              args.ngram_num_threshold, args.ngram_flag, args.av_threshold)
        logger.info('# of n-gram in memory: %d' % len(gram2id))
    else:
        gram2id = None

    label_list = ["O", "B", "I", "E", "S", "[CLS]", "[SEP]"]
    label_map = {label: i for i, label in enumerate(label_list, 1)}

    hpara = WMSeg.init_hyper_parameters(args)
    seg_model = WMSeg(word2id, gram2id, label_map, hpara, args)

    train_examples = seg_model.load_data(args.train_data_path)
    eval_examples = seg_model.load_data(args.eval_data_path)
    num_labels = seg_model.num_labels
    convert_examples_to_features = seg_model.convert_examples_to_features
    feature2input = seg_model.feature2input

    total_params = sum(p.numel() for p in seg_model.parameters() if p.requires_grad)
    logger.info('# of trainable parameters: %d' % total_params)

    num_train_optimization_steps = int(
        len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    if args.fp16:
        seg_model.half()
    seg_model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        seg_model = DDP(seg_model)
    elif n_gpu > 1:
        seg_model = torch.nn.DataParallel(seg_model)

    param_optimizer = list(seg_model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
        warmup_linear = WarmupLinearSchedule(warmup=args.warmup_proportion,
                                             t_total=num_train_optimization_steps)

    else:
        # num_train_optimization_steps=-1
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    best_epoch = -1
    best_p = -1
    best_r = -1
    best_f = -1
    best_oov = -1

    history = {'epoch': [], 'p': [], 'r': [], 'f': [], 'oov': []}
    num_of_no_improvement = 0
    patient = args.patient

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            np.random.shuffle(train_examples)
            seg_model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, start_index in enumerate(tqdm(range(0, len(train_examples), args.train_batch_size))):
                seg_model.train()
                batch_examples = train_examples[start_index: min(start_index +
                                                                 args.train_batch_size, len(train_examples))]
                if len(batch_examples) == 0:
                    continue
                train_features = convert_examples_to_features(batch_examples)
                input_ids, input_mask, l_mask, label_ids, matching_matrix, ngram_ids, ngram_positions, \
                segment_ids, valid_ids, word_ids, word_mask = feature2input(device, train_features)

                loss, _ = seg_model(input_ids, segment_ids, input_mask, label_ids, valid_ids, l_mask, word_ids,
                                    matching_matrix, word_mask, ngram_ids, ngram_positions)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear(global_step / num_train_optimization_steps,
                                                                          args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            seg_model.to(device)

            if args.local_rank == -1 or torch.distributed.get_rank() == 0:
                seg_model.eval()
                eval_loss, eval_accuracy = 0, 0
                nb_eval_steps, nb_eval_examples = 0, 0
                y_true = []
                y_pred = []
                label_map = {i: label for i, label in enumerate(label_list, 1)}
                for start_index in range(0, len(eval_examples), args.eval_batch_size):
                    eval_batch_examples = eval_examples[start_index: min(start_index + args.eval_batch_size,
                                                                         len(eval_examples))]
                    eval_features = convert_examples_to_features(eval_batch_examples)

                    input_ids, input_mask, l_mask, label_ids, matching_matrix, ngram_ids, ngram_positions, \
                    segment_ids, valid_ids, word_ids, word_mask = feature2input(device, eval_features)

                    with torch.no_grad():
                        _, tag_seq = seg_model(input_ids, segment_ids, input_mask, labels=label_ids,
                                                   valid_ids=valid_ids, attention_mask_label=l_mask,
                                                   word_seq=word_ids, label_value_matrix=matching_matrix,
                                                   word_mask=word_mask,
                                                   input_ngram_ids=ngram_ids, ngram_position_matrix=ngram_positions)

                    # logits = torch.argmax(F.log_softmax(logits, dim=2),dim=2)
                    # logits = logits.detach().cpu().numpy()
                    logits = tag_seq.to('cpu').numpy()
                    label_ids = label_ids.to('cpu').numpy()
                    input_mask = input_mask.to('cpu').numpy()

                    for i, label in enumerate(label_ids):
                        temp_1 = []
                        temp_2 = []
                        for j, m in enumerate(label):
                            if j == 0:
                                continue
                            elif label_ids[i][j] == num_labels - 1:
                                y_true.append(temp_1)
                                y_pred.append(temp_2)
                                break
                            else:
                                temp_1.append(label_map[label_ids[i][j]])
                                temp_2.append(label_map[logits[i][j]])

                y_true_all = []
                y_pred_all = []
                sentence_all = []
                for y_true_item in y_true:
                    y_true_all += y_true_item
                for y_pred_item in y_pred:
                    y_pred_all += y_pred_item
                for example, y_true_item in zip(eval_examples, y_true):
                    sen = example.text_a
                    sen = sen.strip()
                    sen = sen.split(' ')
                    if len(y_true_item) != len(sen):
                        sen = sen[:len(y_true_item)]
                    sentence_all.append(sen)
                p, r, f = cws_evaluate_word_PRF(y_pred_all, y_true_all)
                oov = cws_evaluate_OOV(y_pred, y_true, sentence_all, word2id)
                logger.info('OOV: %f' % oov)
                history['epoch'].append(epoch)
                history['p'].append(p)
                history['r'].append(r)
                history['f'].append(f)
                history['oov'].append(oov)
                logger.info("=======entity level========")
                logger.info("\nEpoch: %d, P: %f, R: %f, F: %f, OOV: %f", epoch + 1, p, r, f, oov)
                logger.info("=======entity level========")
                # the evaluation method of NER
                report = classification_report(y_true, y_pred, digits=4)

                if args.model_name is not None:
                    if not os.path.exists(output_model_dir):
                        os.mkdir(output_model_dir)

                    output_eval_file = os.path.join(args.model_name, "eval_results.txt")

                    if os.path.exists(output_eval_file):
                        with open(output_eval_file, "a") as writer:
                            logger.info("***** Eval results *****")
                            logger.info("=======token level========")
                            logger.info("\n%s", report)
                            logger.info("=======token level========")
                            writer.write(report)

                if f > best_f:
                    best_epoch = epoch + 1
                    best_p = p
                    best_r = r
                    best_f = f
                    best_oov = oov
                    num_of_no_improvement = 0

                    if args.model_name:
                        with open(os.path.join(output_model_dir, 'CWS_result.txt'), "w") as writer:
                            for i in range(len(y_pred)):
                                sentence = eval_examples[i].text_a
                                seg_true_str, seg_pred_str = eval_sentence(y_pred[i], y_true[i], sentence, word2id)
                                # logger.info("true: %s", seg_true_str)
                                # logger.info("pred: %s", seg_pred_str)
                                writer.write('True: %s\n' % seg_true_str)
                                writer.write('Pred: %s\n\n' % seg_pred_str)

                        best_eval_model_path = os.path.join(output_model_dir, 'model.pt')

                        if n_gpu > 1:
                            torch.save({
                                'spec': seg_model.module.spec,
                                'state_dict': seg_model.module.state_dict(),
                                # 'trainer': optimizer.state_dict(),
                            }, best_eval_model_path)
                        else:
                            torch.save({
                                'spec': seg_model.spec,
                                'state_dict': seg_model.state_dict(),
                                # 'trainer': optimizer.state_dict(),
                            }, best_eval_model_path)
                else:
                    num_of_no_improvement += 1

            if num_of_no_improvement >= patient:
                logger.info('\nEarly stop triggered at epoch %d\n' % epoch)
                break

        logger.info("\n=======best f entity level========")
        logger.info("\nEpoch: %d, P: %f, R: %f, F: %f, OOV: %f\n", best_epoch, best_p, best_r, best_f, best_oov)
        logger.info("\n=======best f entity level========")

        if os.path.exists(output_model_dir):
            with open(os.path.join(output_model_dir, 'history.json'), 'w', encoding='utf8') as f:
                json.dump(history, f)
                f.write('\n')


def test(args):

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    print("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    seg_model_checkpoint = torch.load(args.eval_model)
    seg_model = WMSeg.from_spec(seg_model_checkpoint['spec'], seg_model_checkpoint['state_dict'], args)

    eval_examples = seg_model.load_data(args.eval_data_path)
    convert_examples_to_features = seg_model.convert_examples_to_features
    feature2input = seg_model.feature2input
    num_labels = seg_model.num_labels
    word2id = seg_model.word2id
    label_map = {v: k for k, v in seg_model.labelmap.items()}

    if args.fp16:
        seg_model.half()
    seg_model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        seg_model = DDP(seg_model)
    elif n_gpu > 1:
        seg_model = torch.nn.DataParallel(seg_model)

    seg_model.to(device)

    seg_model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    y_true = []
    y_pred = []

    for start_index in tqdm(range(0, len(eval_examples), args.eval_batch_size)):
        eval_batch_examples = eval_examples[start_index: min(start_index + args.eval_batch_size,
                                                             len(eval_examples))]
        eval_features = convert_examples_to_features(eval_batch_examples)

        input_ids, input_mask, l_mask, label_ids, matching_matrix, ngram_ids, ngram_positions, \
        segment_ids, valid_ids, word_ids, word_mask = feature2input(device, eval_features)

        with torch.no_grad():
            _, tag_seq = seg_model(input_ids, segment_ids, input_mask, labels=label_ids,
                                       valid_ids=valid_ids, attention_mask_label=l_mask,
                                       word_seq=word_ids, label_value_matrix=matching_matrix,
                                       word_mask=word_mask,
                                       input_ngram_ids=ngram_ids, ngram_position_matrix=ngram_positions)

        # logits = torch.argmax(F.log_softmax(logits, dim=2),dim=2)
        # logits = logits.detach().cpu().numpy()
        logits = tag_seq.to('cpu').numpy()
        label_ids = label_ids.to('cpu').numpy()
        input_mask = input_mask.to('cpu').numpy()

        for i, label in enumerate(label_ids):
            temp_1 = []
            temp_2 = []
            for j, m in enumerate(label):
                if j == 0:
                    continue
                elif label_ids[i][j] == num_labels - 1:
                    y_true.append(temp_1)
                    y_pred.append(temp_2)
                    break
                else:
                    temp_1.append(label_map[label_ids[i][j]])
                    temp_2.append(label_map[logits[i][j]])

    y_true_all = []
    y_pred_all = []
    sentence_all = []
    for y_true_item in y_true:
        y_true_all += y_true_item
    for y_pred_item in y_pred:
        y_pred_all += y_pred_item
    for example, y_true_item in zip(eval_examples, y_true):
        sen = example.text_a
        sen = sen.strip()
        sen = sen.split(' ')
        if len(y_true_item) != len(sen):
            sen = sen[:len(y_true_item)]
        sentence_all.append(sen)
    p, r, f = cws_evaluate_word_PRF(y_pred_all, y_true_all)
    oov = cws_evaluate_OOV(y_pred, y_true, sentence_all, word2id)

    print(args.eval_data_path)
    print("\nP: %f, R: %f, F: %f, OOV: %f" % (p, r, f, oov))


def predict(args):
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    print("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    seg_model_checkpoint = torch.load(args.eval_model)
    seg_model = WMSeg.from_spec(seg_model_checkpoint['spec'], seg_model_checkpoint['state_dict'], args)

    eval_examples = seg_model.load_data(args.input_file, do_predict=True)
    convert_examples_to_features = seg_model.convert_examples_to_features
    feature2input = seg_model.feature2input
    num_labels = seg_model.num_labels
    word2id = seg_model.word2id
    label_map = {v: k for k, v in seg_model.labelmap.items()}

    if args.fp16:
        seg_model.half()
    seg_model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        seg_model = DDP(seg_model)
    elif n_gpu > 1:
        seg_model = torch.nn.DataParallel(seg_model)

    seg_model.to(device)

    seg_model.eval()
    y_pred = []

    for start_index in tqdm(range(0, len(eval_examples), args.eval_batch_size)):
        eval_batch_examples = eval_examples[start_index: min(start_index + args.eval_batch_size,
                                                             len(eval_examples))]
        eval_features = convert_examples_to_features(eval_batch_examples)

        input_ids, input_mask, l_mask, label_ids, matching_matrix, ngram_ids, ngram_positions, \
        segment_ids, valid_ids, word_ids, word_mask = feature2input(device, eval_features)

        with torch.no_grad():
            _, tag_seq = seg_model(input_ids, segment_ids, input_mask, labels=label_ids,
                                       valid_ids=valid_ids, attention_mask_label=l_mask,
                                       word_seq=word_ids, label_value_matrix=matching_matrix,
                                       word_mask=word_mask,
                                       input_ngram_ids=ngram_ids, ngram_position_matrix=ngram_positions)

        logits = tag_seq.to('cpu').numpy()
        label_ids = label_ids.to('cpu').numpy()

        for i, label in enumerate(label_ids):
            temp = []
            for j, m in enumerate(label):
                if j == 0:
                    continue
                elif label_ids[i][j] == num_labels - 1:
                    y_pred.append(temp)
                    break
                else:
                    temp.append(label_map[logits[i][j]])

    print('write results to %s' % str(args.output_file))
    with open(args.output_file, 'w', encoding='utf8') as writer:
        for i in range(len(y_pred)):
            sentence = eval_examples[i].text_a
            _, seg_pred_str = eval_sentence(y_pred[i], None, sentence, word2id)
            writer.write('%s\n' % seg_pred_str)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_test",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_predict",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--train_data_path",
                        default=None,
                        type=str,
                        help="The training data path. Should contain the .tsv files for the task.")
    parser.add_argument("--eval_data_path",
                        default=None,
                        type=str,
                        help="The eval/testing data path. Should contain the .tsv files for the task.")
    parser.add_argument("--input_file",
                        default=None,
                        type=str,
                        help="The data path containing the sentences to be segmented")
    parser.add_argument("--output_file",
                        default=None,
                        type=str,
                        help="The output path of segmented file")
    parser.add_argument("--use_bert",
                        action='store_true',
                        help="Whether to use BERT.")
    parser.add_argument("--use_zen",
                        action='store_true',
                        help="Whether to use ZEN.")
    parser.add_argument("--bert_model", default=None, type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--eval_model", default=None, type=str,
                        help="")
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_ngram_size",
                        default=128,
                        type=int,
                        help="The maximum candidate word size used by attention. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--patient', type=int, default=3, help="Patient for the early stop.")
    parser.add_argument('--ngram_num_threshold', type=int, default=0, help="The threshold of n-gram frequency")
    parser.add_argument('--av_threshold', type=int, default=5, help="av threshold")
    parser.add_argument('--max_ngram_length', type=int, default=5,
                        help="The maximum length of n-grams to be considered.")
    parser.add_argument('--model_name', type=str, default=None, help="")
    parser.add_argument("--use_memory",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument('--decoder', type=str, default='crf',
                        help="the decoder to be used, should be one of softmax and crf.")
    parser.add_argument('--ngram_flag', type=str, default='av', help="")
    parser.add_argument('--save_top',
                        type=int,
                        default=1,
                        help="")

    args = parser.parse_args()

    if args.do_train:
        train(args)
    elif args.do_test:
        test(args)
    elif args.do_predict:
        predict(args)
    else:
        raise ValueError('At least one of `do_train`, `do_eval`, `do_predict` must be True.')


if __name__ == "__main__":
    main()
