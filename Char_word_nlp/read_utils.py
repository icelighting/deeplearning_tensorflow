import numpy as np
import copy
import time
import tensorflow as tf
import collections
import pickle


def batch_generator(arr, num_seqs, num_steps):
    '''
    :param arr: training target text  which has been transformed into int
    :param num_seqs: length of sequence
    :param num_steps: length of steps
    :return:
    '''
    text = copy.deepcopy(arr)
    batch_size = num_seqs * num_steps
    #assert len(text) % batch_size == 0
    num_batch = len(text) // batch_size ## 这里的 batch_size 已经给定了二，所以每一次batch生成器会自动的生成所需要的训练数据，而不需要在model中获取，
    text = text[:num_batch * batch_size]
    text = text.reshape((num_seqs, -1))## 这里表示第一个维度等同于 num_seqs 剩下的维度则自动补全
    while True:
        np.random.shuffle(text)
        for i in range(0, text.shape[1], num_steps):
            x = text[:, i:i+ num_steps]
            y = np.zeros_like(x)
            # 目标值向下滚动一个字母，目标值的最后一列为 样本特征batch的第一列， 不会影响精度
            y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
            yield x, y

class TextConverter(object):
    def __init__(self, text=None, max_vocab=5000, filename=None):
        if filename is not None:
            with open(filename, 'rb') as f:
                self.vocab = pickle.load(f)
        else:
            vocab = set(text)
            print(len(vocab))
            # max_vocab_process
            vocab_count = collections.Counter(text)
            count = [['UNK',-1]]
            count.extend(vocab_count.most_common(max_vocab-1))
            dictionary = dict()
            for word,_ in count:
                dictionary[word] = len(dictionary)
            data = list()
            unk_count = 0
            for word in text:
                if word in dictionary:
                    index = dictionary[word]
                else:
                    index = 0
                    unk_count += 1
                data.append(index)
            count[0][1] = unk_count
            count.sort(key=lambda x:x[1], reverse=True)
            vocab = [x[0] for x in count]
            #for word in vocab:
            #    vocab_count[word] = 0
            #for word in text:
            #    vocab_count[word] += 1
            #vocab_count_list = []
            #for word in vocab_count:
            #   vocab_count_list.append((word, vocab_count[word]))
            #vocab_count_list.sort(key=lambda x: x[1], reverse=True)
            #if len(vocab_count_list) > max_vocab:
            #    vocab_count_list = vocab_count_list[:max_vocab]
            #vocab = [x[0] for x in vocab_count_list]
            self.vocab = vocab

        self.word_to_int_table = {c: i for i, c in enumerate(self.vocab)}
        self.int_to_word_table = dict(enumerate(self.vocab))

    @property
    def vocab_size(self):
        return len(self.vocab) + 1

    def word_to_int(self, word):
        if word in self.word_to_int_table:
            return self.word_to_int_table[word]
        else:
            return len(self.vocab)

    def int_to_word(self, index):
        if index == len(self.vocab):
            return '<unk>'
        elif index < len(self.vocab):
            return self.int_to_word_table[index]
        else:
            raise Exception('Unknown index!')

    def text_to_arr(self, text):
        arr = []
        for word in text:
            arr.append(self.word_to_int(word))
        return np.array(arr)

    def arr_to_text(self, arr):
        words = []
        for index in arr:
            words.append(self.int_to_word(index))
        return "".join(words)

    def save_to_file(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.vocab, f)
