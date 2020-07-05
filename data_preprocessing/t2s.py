from langconv import *
from os import path
import os
import argparse
from tqdm import tqdm

input_dir = './traditional_data/'
output_dir = './processed/'

def read_file(file_path):
    sentence_list = []
    label_list = []
    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        sentence = []
        labels = []
        for line in lines:
            line = line.strip()
            if line == '':
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

    return sentence_list, label_list


def write_file(file_path, sentence_list, label_list):
    with open(file_path, 'w', encoding='utf8') as f:
        for sentence, label in zip(sentence_list, label_list):
            for s, l in zip(sentence, label):
                f.write('%s\t%s\n' % (s, l))
            f.write('\n')


def Traditional2Simplified(sentence):
    sentence = Converter('zh-hans').convert(sentence)
    return sentence


def traditional2simplified(input_file_path, output_file_path):
    simp_sentence_list = []

    sentence_list, label_list = read_file(input_file_path)
    for sentence in tqdm(sentence_list):
        sentence_str = ''.join(sentence)
        simp_sentence_str = Traditional2Simplified(sentence_str)
        assert len(simp_sentence_str) == len(sentence)
        simp_sentence_list.append(simp_sentence_str)

    write_file(output_file_path, simp_sentence_list, label_list)
