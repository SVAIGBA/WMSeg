import re
import numpy as np
import json
from os import path
from collections import defaultdict
from math import log
import copy

class Find_Words:
    def __init__(self, min_count=10, max_count=10000000, min_pmi=0):
        self.min_count = min_count
        self.min_pmi = min_pmi
        self.chars, self.pairs = defaultdict(int), defaultdict(int)
        self.total = 0.
        self.max_count = max_count

    def text_filter(self, texts):
        for a in texts:
            for t in re.split(u'[^\u4e00-\u9fa50-9a-zA-Z]+', a):
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
        self.words = {i:j for i,j in self.words.items() if j >= self.min_count and 6 > len(i) > 1}


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

    return n_gram_dict


def dlg(train_path, eval_path, min_freq):
    train_sentences, _ = read_tsv(train_path)
    test_sentences, _ = read_tsv(eval_path)

    all_sentences = train_sentences + test_sentences

    n_gram_dict = extract_ngram(all_sentences, 0)
    corpus_size = 0
    for gram, count in n_gram_dict.items():
        if len(gram) == 1:
            corpus_size += count

    min_dlg = np.inf
    max_dlg = -np.inf

    min_dlg_2 = np.inf
    max_dlg_2 = -np.inf

    n_gram_dlg_dict = {}
    num_small_dlg = 0
    skip_num = 0

    for gram, c_gram in n_gram_dict.items():
        if len(gram) == 1 or c_gram < 2:
            skip_num += 1
            continue
        new_corpus_size = corpus_size - c_gram * (len(gram) - 1) + len(gram) + 1
        dlg = c_gram * np.log10(c_gram) + corpus_size * np.log10(corpus_size) - new_corpus_size * np.log10(new_corpus_size)
        if dlg > max_dlg_2:
            max_dlg_2 = dlg
        if dlg < min_dlg_2:
            min_dlg_2 = dlg
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

    new_dlg_dict = {}
    new_all_sentences = []
    for sen in all_sentences:
        str_sen = ''.join(sen)
        new_sen = re.split(u'[^\u4e00-\u9fa50-9a-zA-Z]+', str_sen)
        for s in new_sen:
            if len(s) > 0:
                new_all_sentences.append(s)
    for sentence in new_all_sentences:
        n_gram_list = vitbi(sentence, n_gram_dlg_dict)
        for gram in n_gram_list:
            if gram not in new_dlg_dict:
                new_dlg_dict[gram] = 0
            else:
                new_dlg_dict[gram] += 1

    new_dlg_dict_2 = {gram: c for gram, c in new_dlg_dict.items() if c >= min_freq}

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


def pmi(train_path, eval_path, min_freq):
    train_sentences, _ = read_tsv(train_path)
    test_sentences, _ = read_tsv(eval_path)
    all_sentences = []
    for sentence in train_sentences + test_sentences:
        all_sentences.append(''.join(sentence))

    fw = Find_Words(min_freq, 1000000000000, 0)
    fw.count(all_sentences)
    fw.find_words(all_sentences)
    words = fw.words

    return words


def av(train_path, eval_path, min_freq, av_threshold=5):

    train_sentences, _ = read_tsv(train_path)
    test_sentences, _ = read_tsv(eval_path)

    all_sentences = train_sentences + test_sentences

    n_gram_dict = {}
    new_all_sentences = []

    ngram2av = {}

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
        if av >= av_threshold and n_gram_dict[ngram] >= min_freq:
            remaining_ngram[ngram] = n_gram_dict[ngram]

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



def get_word2id(train_data_path):
    word2id = {'<PAD>': 0}
    word = ''
    index = 1
    for line in open(train_data_path):
        if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
            continue
        splits = line.split('\t')
        character = splits[0]
        label = splits[-1][:-1]
        word += character
        if label in ['S', 'E']:
            if word not in word2id:
                word2id[word] = index
                index += 1
            word = ''
    return word2id


def get_gram2id(train_data_dir, eval_data_dir, threshold=0, flag='train_words', av_threshold=5):
    if flag == 'dlg':
        word2count = dlg(train_data_dir, eval_data_dir, threshold)
    elif flag == 'pmi':
        word2count = pmi(train_data_dir, eval_data_dir, threshold)
    elif flag == 'av':
        word2count = av(train_data_dir, eval_data_dir, threshold, av_threshold)
    else:
        raise ValueError()

    gram2id = {'<PAD>': 0}
    index = 1
    for word, count in word2count.items():
        gram2id[word] = index
        index += 1
    return gram2id
