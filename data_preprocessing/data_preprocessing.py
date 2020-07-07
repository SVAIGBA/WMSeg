import os
import re
from os import path
import argparse

OUTPUT_DIR = '../data'
if not path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

class SIGHAN2005(object):
    def __init__(self):
        self.data_dir = './icwb2-data'
        self.data_dir_dict = {'train': 'training', 'test': 'gold'}

        self.data_dict = {
            'msr': {'train': 'msr_training.utf8', 'test': 'msr_test_gold.utf8'},
            'pku': {'train': 'pku_training.utf8', 'test': 'pku_test_gold.utf8'},
            'as': {'train': 'as_training.utf8', 'test': 'as_testing_gold.utf8'},
            'cityu': {'train': 'cityu_training.utf8', 'test': 'cityu_test_gold.utf8'}
        }

    def process(self):
        for dataset in self.data_dict.keys():
            for flag in ['train', 'test']:
                file_path = path.join(self.data_dir, self.data_dir_dict[flag], self.data_dict[dataset][flag])
                print('processing: %s' % str(file_path))
                data = self._process_file(file_path)
                output_file_dir = path.join(OUTPUT_DIR, dataset)
                if not path.exists(output_file_dir):
                    os.mkdir(output_file_dir)
                if dataset in ['as', 'cityu']:
                    output_file_name = flag + '.traditional.tsv'
                else:
                    output_file_name = flag + '.tsv'
                output_file_path = path.join(output_file_dir, output_file_name)
                self._write_file(data, output_file_path)

    @staticmethod
    def translate():
        from t2s import traditional2simplified
        for dataset in ['as', 'cityu']:
            for flag in ['train', 'test']:
                input_file_path = path.join(OUTPUT_DIR, dataset, flag + '.traditional.tsv')
                output_file_path = path.join(OUTPUT_DIR, dataset, flag + '.tsv')
                print('translating: %s' % str(input_file_path))
                traditional2simplified(input_file_path, output_file_path)
        return

    @staticmethod
    def _process_file(file_path):
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            data = []
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line == '':
                    continue
                word_list = re.split('\\s+', line)
                for word in word_list:
                    if len(word) == 1:
                        data.append('%s\t%s' % (word, 'S'))
                    elif len(word) == 2:
                        data.append('%s\t%s' % (word[0], 'B'))
                        data.append('%s\t%s' % (word[1], 'E'))
                    else:
                        data.append('%s\t%s' % (word[0], 'B'))
                        for i in range(1, len(word) - 1):
                            data.append('%s\t%s' % (word[i], 'I'))
                        data.append('%s\t%s' % (word[-1], 'E'))
                data.append('\n')
        return data

    @staticmethod
    def _write_file(data, file_path):
        with open(file_path, 'w', encoding='utf8') as f:
            for line in data:
                f.write(line)
                f.write('\n')


class CTB6(object):
    def __init__(self):
        self.raw_data_dir = './LDC07T36/data/utf8'

        # This split follows https://www.aclweb.org/anthology/P12-1083/ (Table 1)
        dev_index = [i for i in range(41, 81)]
        dev_index.extend([i for i in range(1120, 1130)])
        dev_index.extend([i for i in range(2140, 2160)])
        dev_index.extend([i for i in range(2280, 2295)])
        dev_index.extend([i for i in range(2550, 2570)])
        dev_index.extend([i for i in range(2775, 2800)])
        dev_index.extend([i for i in range(3080, 3110)])

        self.dev_index = set(dev_index)

        test_index = [i for i in range(1, 41)]
        test_index.extend([i for i in range(901, 932)])
        test_index.extend([1018, 1020, 1036, 1044, 1060, 1061, 1072, 1118, 1119, 1132, 1141, 1142, 1148])
        test_index.extend([i for i in range(2165, 2181)])
        test_index.extend([i for i in range(2295, 2311)])
        test_index.extend([i for i in range(2570, 2603)])
        test_index.extend([i for i in range(2800, 2820)])
        test_index.extend([i for i in range(3110, 3146)])

        self.test_index = set(test_index)

    def process(self):
        input_dir = path.join(self.raw_data_dir, 'segmented')
        output_dir = path.join(OUTPUT_DIR, 'ctb6')
        if not path.exists(output_dir):
            os.mkdir(output_dir)
        train = []
        dev = []
        test = []
        input_file_list = os.listdir(input_dir)
        input_file_list.sort()

        for file_name in input_file_list:
            if not file_name.endswith('.seg'):
                continue
            file_path = path.join(input_dir, file_name)

            file_index = int(file_name[file_name.find('_') + 1: file_name.rfind('.')])

            with open(file_path, 'r', encoding='utf8') as f:
                data = []
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if line == '' or line.startswith('<'):
                        continue
                    word_list = re.split('\\s+', line)
                    for word in word_list:
                        if len(word) == 1:
                            data.append('%s\t%s' % (word, 'S'))
                        elif len(word) == 2:
                            data.append('%s\t%s' % (word[0], 'B'))
                            data.append('%s\t%s' % (word[1], 'E'))
                        else:
                            data.append('%s\t%s' % (word[0], 'B'))
                            for i in range(1, len(word) - 1):
                                data.append('%s\t%s' % (word[i], 'I'))
                            data.append('%s\t%s' % (word[-1], 'E'))
                    data.append('\n')
                if file_index in self.test_index:
                    test.extend(data)
                elif file_index in self.dev_index:
                    dev.extend(data)
                else:
                    train.extend(data)
        output_train_path = path.join(output_dir, 'train.tsv')
        output_dev_path = path.join(output_dir, 'dev.tsv')
        output_test_path = path.join(output_dir, 'test.tsv')

        with open(output_train_path, 'w', encoding='utf8') as f:
            for line in train:
                f.write(line)
                f.write('\n')
        with open(output_test_path, 'w', encoding='utf8') as f:
            for line in test:
                f.write(line)
                f.write('\n')
        with open(output_dev_path, 'w', encoding='utf8') as f:
            for line in dev:
                f.write(line)
                f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset",
                        default=None,
                        type=str,
                        required=True,
                        help="The dataset. Should be one of \'sighan2005\' and \'ctb6\'.")

    parser.add_argument("--translate",
                        action='store_true',
                        help="Whether translate traditional Chinese into simplified Chinese.")

    args = parser.parse_args()

    print(vars(args))

    if args.dataset == 'sighan2005':
        processor = SIGHAN2005()
        processor.process()
        if args.translate:
            processor.translate()
    elif args.dataset == 'ctb6':
        if not path.exists('./LDC07T36'):
            raise FileNotFoundError('Do not find CTB6 dataset')
        processor = CTB6()
        processor.process()
