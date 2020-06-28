import argparse
import re
import numpy as np
import json
from tqdm import tqdm
from os import path
from collections import defaultdict  # defaultdict是经过封装的dict，它能够让我们设定默认值
from math import log
import copy
import operator
import pandas as pd

DATA_DIR = './data/CWS'
punctuation = ['。', '，', '、', '：', '？', '！', '（', '）', '“', '”', '【', '】']


class Find_Words:
    def __init__(self, min_count=10, max_count=10000000, min_pmi=0):
        self.min_count = min_count
        self.min_pmi = min_pmi
        self.chars, self.pairs = defaultdict(int), defaultdict(int) #如果键不存在，那么就用int函数
                                                                  #初始化一个值，int()的默认结果为0
        self.total = 0.
        self.max_count = max_count

    def text_filter(self, texts): #预切断句子，以免得到太多无意义（不是中文、英文、数字）的字符串
        for a in tqdm(texts):
            for t in re.split(u'[^\u4e00-\u9fa50-9a-zA-Z]+', a): #这个正则表达式匹配的是任意非中文、
                                                              #非英文、非数字，因此它的意思就是用任
                                                              #意非中文、非英文、非数字的字符断开句子
                if t:
                    yield t

    def count(self, texts): #计数函数，计算单字出现频数、相邻两字出现的频数
        mi_list = []
        for text in self.text_filter(texts):
            self.chars[text[0]] += 1
            for i in range(len(text)-1):
                self.chars[text[i+1]] += 1
                self.pairs[text[i:i+2]] += 1
                self.total += 1
        self.chars = {i:j for i,j in self.chars.items() if 100 * self.max_count > j > self.min_count} #最少频数过滤
        self.pairs = {i:j for i,j in self.pairs.items() if self.max_count > j > self.min_count} #最少频数过滤
        # self.chars = {i:j for i,j in self.chars.items() if j >= 1} #最少频数过滤
        # self.pairs = {i:j for i,j in self.pairs.items() if j >= 1} #最少频数过滤
        self.strong_segments = set()
        for i,j in self.pairs.items(): #根据互信息找出比较“密切”的邻字
            if i[0] in self.chars and i[1] in self.chars:
                mi = log(self.total*j/(self.chars[i[0]]*self.chars[i[1]]))
                mi_list.append(mi)
                if mi >= self.min_pmi:
                    self.strong_segments.add(i)
        print('min mi: %.4f' % min(mi_list))
        print('max mi: %.4f' % max(mi_list))
        print('remaining: %d / %d (%.4f)' % (len(self.strong_segments), len(mi_list), len(self.strong_segments)/len(mi_list)))

    def find_words(self, texts): #根据前述结果来找词语
        self.words = defaultdict(int)
        for text in self.text_filter(texts):
            s = text[0]
            for i in range(len(text)-1):
                if text[i:i+2] in self.strong_segments: #如果比较“密切”则不断开
                    s += text[i+1]
                else:
                    self.words[s] += 1 #否则断开，前述片段作为一个词来统计
                    s = text[i+1]
        self.words = {i:j for i,j in self.words.items() if j > self.min_count and 6 > len(i) > 1} #最后再次根据频数过滤
        # self.words = {i: j for i, j in self.words.items() if j >= self.min_count}  # 最后再次根据频数过滤


def read_tsv(file_path):
    sentence_list = []
    label_list = []
    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        sentence = []
        labels = []
        for line in lines:
            line = line.strip()
            if line == '':
                if len(sentence) > 0:
                    sentence_list.append(sentence)
                    label_list.append(labels)
                    sentence = []
                    labels = []
                continue
            items = re.split('\\s+', line)
            character = items[0]
            label = items[-1]
            sentence.append(character)
            labels.append(label)

            if character in ['，', '。', '？', '！', '：', '；', '（', '）', '、'] and len(sentence) > 64:
                sentence_list.append(sentence)
                label_list.append(labels)
                sentence = []
                labels = []

    return sentence_list, label_list


def count_n_gram(data_dir):
    print('count ngram')
    train_sentences, _ = read_tsv(path.join(data_dir, 'train.tsv'))
    test_sentences, _ = read_tsv(path.join(data_dir, 'test.tsv'))

    all_sentences = train_sentences + test_sentences

    n_gram_dict = extract_ngram(all_sentences)

    with open(path.join(data_dir, 'ngram_count.json'), 'w', encoding='utf8') as f:
        json.dump(n_gram_dict, f, ensure_ascii=False)
        f.write('\n')
    with open(path.join(data_dir, 'ngram_count'), 'w', encoding='utf8') as f:
        for n_gram, value in n_gram_dict.items():
            f.write('%s\t%d\n' % (n_gram, value))

    print('# of ngram (freq) %d' % len(n_gram_dict))

    return n_gram_dict


def dlg(data_dir, min_freq):
    print('dlg')
    train_sentences, _ = read_tsv(path.join(data_dir, 'train.tsv'))
    test_sentences, _ = read_tsv(path.join(data_dir, 'test.tsv'))

    all_sentences = train_sentences + test_sentences

    n_gram_dict = extract_ngram(all_sentences, 0)
    corpus_size = 0
    for gram, count in n_gram_dict.items():
        if len(gram) == 1:
            corpus_size += count
    print('%s corpus size: %d' % (data_dir, corpus_size))

    min_dlg = np.inf
    max_dlg = -np.inf

    min_dlg_2 = np.inf
    max_dlg_2 = -np.inf

    print('processing dlg ...')

    n_gram_dlg_dict = {}
    num_small_dlg = 0
    skip_num = 0

    for gram, c_gram in tqdm(n_gram_dict.items()):
        if len(gram) == 1 or c_gram < 2:
            skip_num += 1
            continue
        new_corpus_size = corpus_size - c_gram * (len(gram) - 1) + len(gram) + 1
        dlg = c_gram * np.log10(c_gram) + corpus_size * np.log10(corpus_size) - new_corpus_size * np.log10(new_corpus_size)
        if dlg > max_dlg_2:
            max_dlg_2 = dlg
        if dlg < min_dlg_2:
            min_dlg_2 = dlg
        if dlg > 200000:
            print('%s %d' % (gram, c_gram))
        char_in_gram = list(set(gram))
        for character in char_in_gram:
            c_character = n_gram_dict[character]
            new_c_character = c_character - (c_gram - 1) * gram.count(character)
            # if not new_c_character > 0:
            #     print('gram: %s' % gram)
            #     print('# of new c character: %d' % new_c_character)
            #     raise ValueError()
            new_character_item = new_c_character * np.log10(new_c_character) if new_c_character > 0 else 0
            dlg += new_character_item - c_character * np.log10(c_character)
        if dlg > 0:
            n_gram_dlg_dict[gram] = dlg / c_gram
        else:
            num_small_dlg += 1
        if dlg > max_dlg:
            max_dlg = dlg
        if dlg < min_dlg:
            min_dlg = dlg

    print('max dlg 2: ', max_dlg_2)
    print('min dlg 2: ', min_dlg_2)

    print('max dlg: ', max_dlg)
    print('min dlg: ', min_dlg)
    # print('# of dlg < 0: %d' % num_small_dlg)
    # print('n-gram-count: %d / %d' % (len(n_gram_dlg_dict), len(n_gram_dict) - skip_num))

    new_dlg_dict = {}
    new_all_sentences = []
    for sen in all_sentences:
        str_sen = ''.join(sen)
        new_sen = re.split(u'[^\u4e00-\u9fa50-9a-zA-Z]+', str_sen)
        for s in new_sen:
            if len(s) > 0:
                new_all_sentences.append(s)
    for sentence in tqdm(new_all_sentences):
        n_gram_list = vitbi(sentence, n_gram_dlg_dict)
        for gram in n_gram_list:
            if gram not in new_dlg_dict:
                new_dlg_dict[gram] = 0
            else:
                new_dlg_dict[gram] += 1

    new_dlg_dict_2 = {gram: c for gram, c in new_dlg_dict.items() if c > min_freq}

    print('dlg size: %d' % len(new_dlg_dict_2))

    # with open(path.join(data_dir, 'ngram_dlg.json'), 'w', encoding='utf8') as f:
    #     json.dump(new_dlg_dict_2, f, ensure_ascii=False)
    #     f.write('\n')
    # with open(path.join(data_dir, 'ngram_dlg'), 'w', encoding='utf8') as f:
    #     for n_gram, value in new_dlg_dict_2.items():
    #         f.write('%s\t%f\n' % (n_gram, value))

    return new_dlg_dict_2



def dlg_3(data_dir, min_freq):
    print('dlg')
    train_sentences, _ = read_tsv(path.join(data_dir, 'train.tsv'))
    test_sentences, _ = read_tsv(path.join(data_dir, 'test.tsv'))

    all_sentences = train_sentences + test_sentences

    n_gram_dict = extract_ngram(all_sentences, 0)
    corpus_size = get_corpus_size(all_sentences)

    char_dict = extract_characters(all_sentences)
    print('%s corpus size: %d' % (data_dir, corpus_size))

    min_dlg = np.inf
    max_dlg = -np.inf

    print('processing dlg ...')

    n_gram_dlg_dict = {}
    num_small_dlg = 0
    skip_num = 0

    for gram, c_gram in tqdm(n_gram_dict.items()):
        if len(gram) == 1 or c_gram < 2:
            skip_num += 1
            continue
        new_corpus_size = corpus_size - c_gram * (len(gram) - 1) + len(gram) + 1
        dlg = corpus_size * np.log(corpus_size) - new_corpus_size * np.log(new_corpus_size)
        # if dlg > max_dlg_2:
        #     max_dlg_2 = dlg
        # if dlg < min_dlg_2:
        #     min_dlg_2 = dlg
        # if dlg > 200000:
        #     print('%s %d' % (gram, c_gram))
        char_in_gram = list(set(gram))
        for character in char_in_gram:
            c_character = char_dict[character]
            new_c_character = c_character - (c_gram - 1) * gram.count(character)
            # if not new_c_character > 0:
            #     print('gram: %s' % gram)
            #     print('# of new c character: %d' % new_c_character)
            #     raise ValueError()
            new_character_item = new_c_character * np.log(new_c_character) if new_c_character > 0 else 0
            dlg += new_character_item - c_character * np.log(c_character)
        adlg = dlg / c_gram
        if adlg > 0:
            n_gram_dlg_dict[gram] = adlg
        else:
            num_small_dlg += 1
        if adlg > max_dlg:
            max_dlg = adlg
        if adlg < min_dlg:
            min_dlg = adlg

    # print('max dlg 2: ', max_dlg_2)
    # print('min dlg 2: ', min_dlg_2)

    print('max dlg: ', max_dlg)
    print('min dlg: ', min_dlg)
    # print('# of dlg < 0: %d' % num_small_dlg)
    # print('n-gram-count: %d / %d' % (len(n_gram_dlg_dict), len(n_gram_dict) - skip_num))

    new_dlg_dict = {}
    new_all_sentences = []
    for sen in all_sentences:
        str_sen = ''.join(sen)
        new_sen = re.split(u'[^\u4e00-\u9fa50-9a-zA-Z]+', str_sen)
        for s in new_sen:
            if len(s) > 0:
                new_all_sentences.append(s)
    for sentence in tqdm(new_all_sentences):
        n_gram_list = vitbi(sentence, n_gram_dlg_dict)
        for gram in n_gram_list:
            if gram not in new_dlg_dict:
                new_dlg_dict[gram] = 0
            else:
                new_dlg_dict[gram] += 1

    new_dlg_dict_2 = {gram: c for gram, c in new_dlg_dict.items() if c > min_freq}

    print('dlg size: %d' % len(new_dlg_dict_2))

    # with open(path.join(data_dir, 'ngram_dlg.json'), 'w', encoding='utf8') as f:
    #     json.dump(new_dlg_dict_2, f, ensure_ascii=False)
    #     f.write('\n')
    # with open(path.join(data_dir, 'ngram_dlg'), 'w', encoding='utf8') as f:
    #     for n_gram, value in new_dlg_dict_2.items():
    #         f.write('%s\t%f\n' % (n_gram, value))

    return new_dlg_dict_2


def get_corpus_size(all_sentences):
    corpus_size = 0
    for sen in all_sentences:
        corpus_size += len(sen)
    return corpus_size


def dl(corpus_size, vocab):
    dl = 0
    for char, count in vocab.items():
        dl -= count * np.log10(count / corpus_size)
    return dl


def replace_corpus(all_sentence, n_gram):
    l = len(n_gram)
    new_sentence_list = []
    for sen in all_sentence:
        sen_str = ''.join(sen)
        new_sen = []
        current_index = 0
        while current_index < len(sen_str):
            word = sen_str[current_index: current_index + l]
            if n_gram == word:
                current_index += l
            else:
                new_sen.append(sen_str[current_index])
                current_index += 1
        new_sentence_list.append(new_sen)
    return new_sentence_list


def get_new_char_dict(char_dict, ngram, ngram_count):
    new_char_dict = copy.deepcopy(char_dict)
    checked_char_list = []
    for char in ngram:
        if char in checked_char_list:
            continue
        new_number = new_char_dict[char] - ngram_count
        new_char_dict[char] = max(0, new_number)
        checked_char_list.append(char)
    return new_char_dict


def dlg_2(data_dir, min_freq):
    print('dlg 2')
    train_sentences, _ = read_tsv(path.join(data_dir, 'train.tsv'))
    test_sentences, _ = read_tsv(path.join(data_dir, 'test.tsv'))

    all_sentences = train_sentences + test_sentences

    n_gram_dict = extract_ngram(all_sentences, min_freq)

    data_size = get_corpus_size(all_sentences)

    char_dict = extract_characters(all_sentences)

    data_dl = dl(data_size, char_dict)

    print('%s corpus dl: %f' % (data_dir, data_dl))

    min_dlg = np.inf
    max_dlg = -np.inf

    min_dlg_2 = np.inf
    max_dlg_2 = -np.inf

    print('processing dlg ...')

    n_gram_dlg_dict = {}
    num_small_dlg = 0
    skip_num = 0
    #
    for gram, c_gram in tqdm(n_gram_dict.items()):
        if len(gram) == 1 or c_gram < 2:
            skip_num += 1
            continue
        new_char_dict = get_new_char_dict(char_dict, gram, c_gram)

        new_data_size = data_size - len(gram) * c_gram

        if new_data_size < 1:
            print('new data size < 0: %s', gram)
            continue

        dlg = dl(new_data_size, new_char_dict)
        if dlg > 0:
            n_gram_dlg_dict[gram] = dlg / c_gram
        else:
            num_small_dlg += 1
        if dlg > max_dlg:
            max_dlg = dlg
        if dlg < min_dlg:
            min_dlg = dlg

    # print('max dlg 2: ', max_dlg_2)
    # print('min dlg 2: ', min_dlg_2)

    print('max dlg: ', max_dlg)
    print('min dlg: ', min_dlg)
    # print('# of dlg < 0: %d' % num_small_dlg)
    # print('n-gram-count: %d / %d' % (len(n_gram_dlg_dict), len(n_gram_dict) - skip_num))

    # new_dlg_dict = {}
    # new_all_sentences = []
    # for sen in all_sentences:
    #     str_sen = ''.join(sen)
    #     new_sen = re.split(u'[^\u4e00-\u9fa50-9a-zA-Z]+', str_sen)
    #     for s in new_sen:
    #         if len(s) > 0:
    #             new_all_sentences.append(s)
    # for sentence in tqdm(new_all_sentences):
    #     n_gram_list = vitbi(sentence, n_gram_dlg_dict)
    #     for gram in n_gram_list:
    #         if gram not in new_dlg_dict:
    #             new_dlg_dict[gram] = 0
    #         else:
    #             new_dlg_dict[gram] += 1
    #
    # new_dlg_dict_2 = {gram: c for gram, c in new_dlg_dict.items() if c > min_freq}

    # print('dlg size: %d' % len(new_dlg_dict_2))

    # with open(path.join(data_dir, 'ngram_dlg.json'), 'w', encoding='utf8') as f:
    #     json.dump(new_dlg_dict_2, f, ensure_ascii=False)
    #     f.write('\n')
    # with open(path.join(data_dir, 'ngram_dlg'), 'w', encoding='utf8') as f:
    #     for n_gram, value in new_dlg_dict_2.items():
    #         f.write('%s\t%f\n' % (n_gram, value))

    # return new_dlg_dict_2


def vitbi(sentence, ngram_dict):
    score = [0 for i in range(len(sentence))]
    n_gram = [[] for i in range(len(sentence))]
    word = sentence[0]
    n_gram[0].append(word)
    for i in range(1, len(score)):
        tmp_score_list = [score[i-1], -1, -1, -1, -1]
        for n in range(2, 6):
            if i - n < -1:
                break
            word = ''.join(sentence[i - n + 1: i + 1])
            if word in ngram_dict:
                tmp_score_list[n-1] = score[i-n] + ngram_dict[word] if i-n >= 0 else ngram_dict[word]
        max_score = max(tmp_score_list)
        max_score_index = tmp_score_list.index(max(tmp_score_list))
        word = ''.join(sentence[i-max_score_index: i+1])
        score[i] = max_score
        if i-(max_score_index+1) >= 0:
            n_gram[i].extend(n_gram[i - (max_score_index + 1)])
        n_gram[i].append(word)
    return n_gram[-1]


def pmi(data_dir, min_freq):
    print('pmi')
    train_sentences, _ = read_tsv(path.join(data_dir, 'train.tsv'))
    test_sentences, _ = read_tsv(path.join(data_dir, 'test.tsv'))
    all_sentences = []
    for sentence in train_sentences + test_sentences:
        all_sentences.append(''.join(sentence))

    fw = Find_Words(min_freq, 1000000000000, 0)
    fw.count(all_sentences)
    fw.find_words(all_sentences)
    words = fw.words

    print('# of ngrams: %d' % (len(words)))

    # with open(path.join(data_dir, 'ngram_pmi.json'), 'w', encoding='utf8') as f:
    #     json.dump(words, f)
    #     f.write('\n')
    # with open(path.join(data_dir, 'ngram_pmi'), 'w', encoding='utf8') as f:
    #     for word, count in words.items():
    #         f.write('%s\t%d\n' % (word, count))
    print('pmi size: %d' % len(words))
    return words


#
# def mi_stat(data_dir, min_count, max_count, pmi):
#     train_sentences, _ = read_tsv(path.join(data_dir, 'train.tsv'))
#     test_sentences, _ = read_tsv(path.join(data_dir, 'test.tsv'))
#     all_sentences = []
#     for sentence in train_sentences + test_sentences:
#         all_sentences.append(''.join(sentence))
#
#     fw = Find_Words(min_count, max_count, pmi)
#     fw.count(all_sentences)
#     fw.find_words(all_sentences)
#     ngram2count = fw.words
#
#     word2count = {}
#     word = ''
#
#     with open(path.join(data_dir, "train.tsv"), 'r', encoding='utf8') as f:
#         for line in tqdm(f.readlines()):
#             line = line.strip()
#             if len(line) == 0:
#                 continue
#             splits = line.split('\t')
#             character = splits[0]
#             label = splits[-1]
#             word += character
#             if label in ['S', 'E']:
#                 if len(word) > 5:
#                     word = ''
#                     continue
#                 if word not in word2count:
#                     word2count[word] = 1
#                 else:
#                     word2count[word] += 1
#                 word = ''
#
#     new_word2count = {}
#     for word, count in word2count.items():
#         if 20 > count > 1:
#             new_word2count[word] = count
#     print('word size (length longer than 1): %d' % len(new_word2count))
#
#     new_n_gram_dict = ngram2count
#
#     overlap = 0
#     for ngram in new_n_gram_dict.keys():
#         if ngram in new_word2count:
#             overlap += 1
#
#     p = overlap/len(new_n_gram_dict)
#     r = overlap/len(new_word2count)
#     f = 2 * p * r / (p + r)
#
#     print('ngram size with min count of %d and pmi of %d: %d' % (min_count, pmi, len(ngram2count)))
#     print('# of overlaps between ngram and words: %d (P: %f, R: %f, F: %f)\n'
#           % (overlap, p, r, f))


def get_word2id(data_dir):
    word2id_path = path.join(data_dir, 'word2id.json')
    word2count_path = path.join(data_dir, 'word2count.json')
    word2id = {'<PAD>': 0}
    word2count = {}
    word = ''
    index = 1
    with open(path.join(data_dir, "train.tsv"), 'r', encoding='utf8') as f:
        for line in tqdm(f.readlines()):
            line = line.strip()
            if len(line) == 0:
                continue
            splits = line.split('\t')
            character = splits[0]
            label = splits[-1]
            word += character
            if label in ['S', 'E']:
                if word not in word2id:
                    word2id[word] = index
                    word2count[word] = 1
                    index += 1
                else:
                    word2count[word] += 1
                word = ''
    with open(word2id_path, 'w', encoding='utf8') as f:
        json.dump(word2id, f, ensure_ascii=False)
        f.write('\n')
    with open(path.join(data_dir, 'word2id'), 'w', encoding='utf8') as f:
        for w, v in word2id.items():
            f.write('%s\t%d\n' % (w, v))

    with open(word2count_path, 'w', encoding='utf8') as f:
        json.dump(word2count, f, ensure_ascii=False)
        f.write('\n')
    with open(path.join(data_dir, 'word2count'), 'w', encoding='utf8') as f:
        for w, v in word2count.items():
            f.write('%s\t%d\n' % (w, v))


def train_word(data_dir, min_freq):
    word2count = {}
    word = ''
    with open(path.join(data_dir, "train.tsv"), 'r', encoding='utf8') as f:
        for line in tqdm(f.readlines()):
            line = line.strip()
            if len(line) == 0:
                continue
            splits = line.split('\t')
            character = splits[0]
            label = splits[-1]
            word += character
            if label in ['S', 'E']:
                if word not in word2count:
                    word2count[word] = 1
                else:
                    word2count[word] += 1
                word = ''
    new_word2count = {word: c for word, c in word2count.items() if c > min_freq}
    print('word size: %d' % len(new_word2count))
    return new_word2count


def av(data_dir, min_freq, lower_threshold=5, flag='train+test'):
    print('av')
    train_sentences, _ = read_tsv(path.join(data_dir, 'train.tsv'))
    test_sentences, _ = read_tsv(path.join(data_dir, 'test.tsv'))

    # if data_dir.find('msr') > 0 or data_dir.find('as') > 0:
    #     lower_threshold = 5
    # else:
    #     lower_threshold = 3

    print('av threshold: %d' % lower_threshold)
    if flag == 'train+test':
        all_sentences = train_sentences + test_sentences
    elif flag == 'train':
        all_sentences = train_sentences
    elif flag == 'test':
        all_sentences = test_sentences
    else:
        raise ValueError()
    n_gram_dict = {}
    new_all_sentences = []

    ngram2av = {}

    for sen in all_sentences:
        str_sen = ''.join(sen)
        new_sen = re.split(u'[^\u4e00-\u9fa50-9a-zA-Z]+', str_sen)
        for s in new_sen:
            if len(s) > 0:
                new_all_sentences.append(s)

    for sentence in tqdm(new_all_sentences):
        for i in range(len(sentence)):
            for n in range(1, 6):
                if i + n > len(sentence):
                    break
                left_index = i - 1
                right_index = i + n
                n_gram = ''.join(sentence[i: i + n])
                if n_gram not in n_gram_dict:
                    n_gram_dict[n_gram] = 1
                    ngram2av[n_gram] = {'l': {}, 'r': {}}
                else:
                    n_gram_dict[n_gram] += 1
                if left_index >= 0:
                    ngram2av[n_gram]['l'][sentence[left_index]] = 1
                if right_index < len(sentence):
                    ngram2av[n_gram]['r'][sentence[right_index]] = 1
    remaining_ngram = {}
    for ngram, av_dict in ngram2av.items():
        avl = len(av_dict['l'])
        avr = len(av_dict['r'])
        av = min(avl, avr)
        if av > lower_threshold and n_gram_dict[ngram] > min_freq:
            remaining_ngram[ngram] = n_gram_dict[ngram]

    # with open(path.join(data_dir, 'ngram_av.json'), 'w', encoding='utf8') as f:
    #     json.dump(remaining_ngram, f)
    #     f.write('\n')
    # with open(path.join(data_dir, 'ngram_av'), 'w', encoding='utf8') as f:
    #     for word, count in remaining_ngram.items():
    #         f.write('%s\t%d\n' % (word, count))
    print('av size: %d' % len(remaining_ngram))
    return remaining_ngram


def extract_ngram(all_sentences, min_feq=0):
    n_gram_dict = {}

    new_all_sentences = []

    for sen in all_sentences:
        str_sen = ''.join(sen)
        new_sen = re.split(u'[^\u4e00-\u9fa50-9a-zA-Z]+', str_sen)
        for s in new_sen:
            if len(s) > 0:
                new_all_sentences.append(s)

    for sentence in new_all_sentences:
        for i in range(len(sentence)):
            for n in range(1, 6):
                if i + n > len(sentence):
                    break
                n_gram = ''.join(sentence[i: i + n])
                if n_gram not in n_gram_dict:
                    n_gram_dict[n_gram] = 1
                else:
                    n_gram_dict[n_gram] += 1
    new_ngram_dict = {gram: c for gram, c in n_gram_dict.items() if c > min_feq}
    return new_ngram_dict

def extract_characters(all_sentences):
    char_dict = defaultdict(int)

    for sentence in all_sentences:
        for char in sentence:
            char_dict[char] += 1
    new_char_dict = {gram: c for gram, c in char_dict.items() if c > 1}
    return new_char_dict

def oov_stat(data_dir):
    oov_count = 0
    word_count = 0
    word = ''
    char_count = 0
    oov_dict = {}
    char_dict = {}
    word_dict = {}

    with open(path.join(data_dir, 'word2id.json'), 'r', encoding='utf8') as f:
        word2id = json.loads(f.readline())

    with open(path.join(data_dir, "test.tsv"), 'r', encoding='utf8') as f:
        for line in tqdm(f.readlines()):
            line = line.strip()
            if len(line) == 0:
                continue
            splits = line.split('\t')
            character = splits[0]
            label = splits[-1][0]
            word += character
            char_count += 1
            char_dict[character] = 0
            if label in ['S', 'E']:
                word_count += 1
                word_dict[word] = 0
                if word not in word2id:
                    oov_dict[word] = 0
                    oov_count += 1
                word = ''

    print('# of chars %d' % char_count)
    print('# of words: %d' % word_count)
    print('# of unique char: %d' % len(char_dict))
    print('# of unique word %d' % len(word_dict))
    print('# of OOV: %d' % oov_count)
    print('# of unique OOV: %d' % len(oov_dict))
    print('% of OOV: ', (oov_count / word_count))


def word_stat(data_dir, threshold, upper_threshold=100000):
    word2count_path = path.join(data_dir, 'word2count.json')
    word2count = {}
    if path.exists(word2count_path):
        print('load word from existing file')
        with open(word2count_path, 'r', encoding='utf8') as f:
            word2count = json.loads(f.readline())

    gram2id = {'<PAD>': 0}
    index = 1
    for word, count in word2count.items():
        if count > threshold and count < upper_threshold:
            gram2id[word] = index
            index += 1
    print('# of words: %d' % len(word2count))
    print('# of words appearing more than %d times and less than %d times: %d' % (threshold, upper_threshold, len(gram2id)))
    return word2count


def ngram_stat(data_dir, lower_threshold, upper_threshold):
    train_sentences, _ = read_tsv(path.join(data_dir, 'train.tsv'))
    # test_sentences, _ = read_tsv(path.join(data_dir, 'test.tsv'))

    all_sentences = train_sentences

    n_gram_dict = extract_ngram(all_sentences)

    word2count = {}
    word = ''

    with open(path.join(data_dir, "train.tsv"), 'r', encoding='utf8') as f:
        for line in tqdm(f.readlines()):
            line = line.strip()
            if len(line) == 0:
                continue
            splits = line.split('\t')
            character = splits[0]
            label = splits[-1]
            word += character
            if label in ['S', 'E']:
                if len(word) > 5:
                    word = ''
                    continue
                if word not in word2count:
                    word2count[word] = 1
                else:
                    word2count[word] += 1
                word = ''

    new_word2count = {}
    for word, count in word2count.items():
        if count > 1:
            new_word2count[word] = count
    print('\nword size: %d\n' % len(new_word2count))

    for lt in lower_threshold:
        new_n_gram_dict = {}
        for word, value in n_gram_dict.items():
            # value = int(np.log(n_gram_dict[word]))
            if upper_threshold > value > lt:
                new_n_gram_dict[word] = value

        overlap = 0
        for ngram in new_n_gram_dict.keys():
            if ngram in new_word2count:
                overlap += 1

        print('# of n-gram appears between (%d, %d) times: %d' % (lt, upper_threshold, len(new_n_gram_dict)))
        print('# of overlaps between ngram and words: %d (%f, %f)\n'
              % (overlap, (overlap/len(new_n_gram_dict)), (overlap/len(new_word2count))))

    # for word in

    # return new_n_gram_dict


def sentence_length(data_dir):
    sentence_1, _ = read_tsv(path.join(data_dir, 'train.tsv'))
    sentence_2, _ = read_tsv(path.join(data_dir, 'test.tsv'))
    sentence = sentence_1
    ls = []
    ls_2 = []
    count = 0
    lower_count = 0
    threshold = 128
    lower_threshold = 32
    for s in tqdm(sentence):
        ls_2.append(len(s))
        if len(s) > threshold:
            count += 1
        if np.random.randn(1)[0] > 0.75 and len(s) < lower_threshold:
            lower_count += 1
        s = ''.join(s)
        s2 = re.split('[。？，！；、（）：]', s)

        for s3 in s2:
            ls.append(len(s3))

    print('max sub-sentence length of %s: %d' % (data_dir, max(ls)))
    print('max sentence length of %s: %d' % (data_dir, max(ls_2)))
    print('length larger than threshold %d: %d %.4f' % (threshold, count, (count / len(sentence) * 100)) + '%')
    print('length smaller than threshold %d: %d %.4f' % (lower_threshold, lower_count, (lower_count / len(sentence) * 100)) + '%')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")

    args = parser.parse_args()
    base_min_freq = 1
    av_threshold = 5

    if args.dataset in ['msra']:
        min_freq = 4 * base_min_freq
        av_threshold = 5
    elif args.dataset in ['pku', 'cityu']:
        min_freq = 2 * base_min_freq
        av_threshold = 2
    elif args.dataset in ['as']:
        min_freq = 8 * base_min_freq
        av_threshold = 5
    else:
        min_freq = 1 * base_min_freq
        av_threshold = 2

    print('min freq: %d' % min_freq)

    data_dir = path.join(DATA_DIR, args.dataset)

    print(data_dir)

    get_word2id(data_dir)

    # count_n_gram(data_dir)
    # dlg(data_dir, min_freq)
    # dlg_2(data_dir, min_freq)
    dlg_3(data_dir, min_freq)
    # pmi(data_dir, min_freq)
    # av(data_dir, min_freq, av_threshold)
    # train_word(data_dir, min_freq)

    # word_stat(data_dir, 1, 20)
    # oov_stat(data_dir)
    # sentence_length(data_dir)
    # ngram_stat(data_dir, range(1, 6), 20)
    # mi_stat(data_dir, 2, 3000000000, 0)

    print('')
